#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/sundials/ida.h>

#include <fstream>
#include <iostream>

struct CahnHilliardParameters
{
  double epsilon = std::sqrt(1.5);
  double M = 1.0;
  double t_end = 100.0;
  double dt_initial = 1e-5;
  unsigned int n_refinements = 6;
  unsigned int degree = 1;
};

template<int dim>
class InitialCondition : public dealii::Function<dim>
{
public:
  InitialCondition()
    : dealii::Function<dim>(2) {};

  void vector_value([[maybe_unused]] const dealii::Point<dim>& p,
                    dealii::Vector<double>& values) const override
  {
    values[0] = 0.05 * (2.0 * (double)rand() / RAND_MAX - 1.0); // c
    values[1] = 0.0;                                            // mu
  }
};

template<int dim>
class CahnHilliardIDA
{
public:
  CahnHilliardIDA(const CahnHilliardParameters& prm, MPI_Comm comm);

  void run();

private:
  void setup_mesh();
  void setup_dofs();

  void assemble_jacobian(const dealii::TrilinosWrappers::MPI::Vector& y,
                         double alpha);

  int assemble_residual(double t,
                        const dealii::TrilinosWrappers::MPI::Vector& y,
                        const dealii::TrilinosWrappers::MPI::Vector& y_dot,
                        dealii::TrilinosWrappers::MPI::Vector& res);

  int output_step(double t,
                  const dealii::TrilinosWrappers::MPI::Vector& y,
                  const dealii::TrilinosWrappers::MPI::Vector& y_dot,
                  unsigned int step);

  MPI_Comm mpi_comm;
  dealii::ConditionalOStream pcout;

  dealii::parallel::distributed::Triangulation<dim> tria;
  dealii::FESystem<dim> fe;
  dealii::DoFHandler<dim> dof_handler;
  dealii::AffineConstraints<double> constraints;
  dealii::IndexSet locally_owned_dofs;
  dealii::IndexSet locally_relevant_dofs;

  dealii::TrilinosWrappers::MPI::Vector y_ghosted;
  dealii::TrilinosWrappers::MPI::Vector y_dot_ghosted;

  dealii::TrilinosWrappers::SparseMatrix jacobian_matrix;

  dealii::TrilinosWrappers::PreconditionILU preconditioner;

  CahnHilliardParameters prm;
};

template<int dim>
CahnHilliardIDA<dim>::CahnHilliardIDA(const CahnHilliardParameters& prm,
                                      MPI_Comm comm)
  : mpi_comm(comm)
  , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0)
  , tria(comm)
  , fe(dealii::FE_Q<dim>(prm.degree), 2)
  , dof_handler(tria)
  , prm(prm)
{
}

template<int dim>
void
CahnHilliardIDA<dim>::setup_mesh()
{
  dealii::GridGenerator::hyper_cube(tria, 0.0, 100.0, true);
  tria.refine_global(prm.n_refinements);
}

template<int dim>
void
CahnHilliardIDA<dim>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);
  dealii::DoFRenumbering::component_wise(dof_handler);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs =
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  y_ghosted.reinit(locally_relevant_dofs, mpi_comm);
  y_dot_ghosted.reinit(locally_relevant_dofs, mpi_comm);

  dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  dealii::SparsityTools::distribute_sparsity_pattern(
    dsp, locally_owned_dofs, mpi_comm, locally_relevant_dofs);

  jacobian_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);

  pcout << "  DoFs: " << dof_handler.n_dofs() << "\n";
}

// ============================================================
//  Residual & Jacobian assembly
//
//  We have a system of equations
//
//  y(t) = [c(t), μ(t)]
//  ẏ(t) = [ċ(t)]
//
//  The IDA residual (weak form, per cell) is:
//
//    R_c  = (ċ, φ_c) + M·(∇μ, ∇φ_c)           = 0
//    R_μ  = (μ − f′(c), φ_μ) − ε²·(∇c, ∇φ_μ)  = 0
//
//  with F = [R_c, R_μ]
//
//  For the Jacobian we need to provide
//
//    J = ∂F/∂y + α ∂F/∂ẏ
//
//    J_c_c   = α·(δc, φ_c)
//    J_c_μ   = M·(∇δμ, ∇φ_c)
//    J_μ_c   = (−f''(c)·δc, φ_μ) − ε²·(∇δc, ∇φ_μ)
//    J_μ_μ   = (δμ, φ_μ)
//
//  f'(c)  = c^3 - c
//  f''(c) = 3c^2 - 1
// ============================================================
template<int dim>
void
CahnHilliardIDA<dim>::assemble_jacobian(
  const dealii::TrilinosWrappers::MPI::Vector& y,
  double alpha)
{
  jacobian_matrix = 0.0;

  const dealii::QGauss<dim> quadrature(prm.degree + 2);
  dealii::FEValues<dim> fe_values(fe,
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q = quadrature.size();

  const dealii::FEValuesExtractors::Scalar c_field(0);
  const dealii::FEValuesExtractors::Scalar mu_field(1);

  dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  y_ghosted = y;

  std::vector<double> c_vals(n_q);

  for (const auto& cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);
    cell_matrix = 0.0;

    fe_values[c_field].get_function_values(y_ghosted, c_vals);

    for (unsigned int q = 0; q < n_q; ++q) {
      const double c = c_vals[q];
      const double d2f_dc = 12.0 * c * (c - 1.0) + 2.0;
      const double JxW = fe_values.JxW(q);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        const double phi_c_i = fe_values[c_field].value(i, q);
        const double phi_mu_i = fe_values[mu_field].value(i, q);
        const auto gphi_c_i = fe_values[c_field].gradient(i, q);
        const auto gphi_mu_i = fe_values[mu_field].gradient(i, q);

        const bool i_is_c = fe.system_to_component_index(i).first == 0;
        const bool i_is_mu = fe.system_to_component_index(i).first == 1;

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          const double phi_c_j = fe_values[c_field].value(j, q);
          const double phi_mu_j = fe_values[mu_field].value(j, q);
          const auto gphi_c_j = fe_values[c_field].gradient(j, q);
          const auto gphi_mu_j = fe_values[mu_field].gradient(j, q);

          const bool j_is_c = fe.system_to_component_index(j).first == 0;
          const bool j_is_mu = fe.system_to_component_index(j).first == 1;

          // J_c_c
          if (i_is_c && j_is_c)
            cell_matrix(i, j) += alpha * phi_c_j * phi_c_i * JxW;

          // J_c_μ
          if (i_is_c && j_is_mu)
            cell_matrix(i, j) += prm.M * (gphi_mu_j * gphi_c_i) * JxW;

          // J_μ_c
          if (i_is_mu && j_is_c)
            cell_matrix(i, j) +=
              (-d2f_dc * phi_c_j * phi_mu_i -
               prm.epsilon * prm.epsilon * (gphi_c_j * gphi_mu_i)) *
              JxW;

          // J_μ_μ
          if (i_is_mu && j_is_mu)
            cell_matrix(i, j) += phi_mu_j * phi_mu_i * JxW;
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
      cell_matrix, local_dof_indices, jacobian_matrix);
  }

  jacobian_matrix.compress(dealii::VectorOperation::add);

  dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
  preconditioner.initialize(jacobian_matrix, ilu_data);
}

template<int dim>
int
CahnHilliardIDA<dim>::assemble_residual(
  [[maybe_unused]] double t,
  const dealii::TrilinosWrappers::MPI::Vector& y,
  const dealii::TrilinosWrappers::MPI::Vector& y_dot,
  dealii::TrilinosWrappers::MPI::Vector& res)
{
  res = 0.0;

  const dealii::QGauss<dim> quadrature(prm.degree + 2);
  dealii::FEValues<dim> fe_values(fe,
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q = quadrature.size();

  const dealii::FEValuesExtractors::Scalar c_field(0);
  const dealii::FEValuesExtractors::Scalar mu_field(1);

  dealii::Vector<double> cell_rhs(dofs_per_cell);
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  y_ghosted = y;
  y_dot_ghosted = y_dot;

  std::vector<double> c_vals(n_q), c_dot_vals(n_q), mu_vals(n_q);
  std::vector<dealii::Tensor<1, dim, double>> grad_c_vals(n_q),
    grad_mu_vals(n_q);

  for (const auto& cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);
    cell_rhs = 0.0;

    fe_values[c_field].get_function_values(y_ghosted, c_vals);
    fe_values[c_field].get_function_values(y_dot_ghosted, c_dot_vals);
    fe_values[mu_field].get_function_values(y_ghosted, mu_vals);
    fe_values[c_field].get_function_gradients(y_ghosted, grad_c_vals);
    fe_values[mu_field].get_function_gradients(y_ghosted, grad_mu_vals);

    for (unsigned int q = 0; q < n_q; ++q) {
      const double c = c_vals[q];
      const double c_dot = c_dot_vals[q];
      const double mu = mu_vals[q];
      const double df_dc = 4.0 * (c - 1.0) * (c - 0.5) * c;
      const double JxW = fe_values.JxW(q);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        const double phi_c = fe_values[c_field].value(i, q);
        const double phi_mu = fe_values[mu_field].value(i, q);
        const auto gphi_c = fe_values[c_field].gradient(i, q);
        const auto gphi_mu = fe_values[mu_field].gradient(i, q);

        // R_c
        cell_rhs[i] +=
          (c_dot * phi_c + prm.M * (grad_mu_vals[q] * gphi_c)) * JxW;

        // R_mu
        cell_rhs[i] += ((mu - df_dc) * phi_mu - prm.epsilon * prm.epsilon *
                                                  (grad_c_vals[q] * gphi_mu)) *
                       JxW;
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(cell_rhs, local_dof_indices, res);
  }

  res.compress(dealii::VectorOperation::add);
  return 0;
}

template<int dim>
int
CahnHilliardIDA<dim>::output_step(
  double t,
  const dealii::TrilinosWrappers::MPI::Vector& y,
  [[maybe_unused]] const dealii::TrilinosWrappers::MPI::Vector& y_dot,
  unsigned int step)
{
  pcout << "  Step " << step << "  t = " << t << "\n";

  y_ghosted = y;
  y_dot_ghosted = y_dot;

  std::vector<std::string> names = { "c", "mu" };
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    interp(2, dealii::DataComponentInterpretation::component_is_scalar);

  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(
    y_ghosted, names, dealii::DataOut<dim>::type_dof_data, interp);

  std::vector<std::string> names_2 = { "c_dot", "mu_dot" };
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    interp_2(2, dealii::DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(
    y_dot_ghosted, names_2, dealii::DataOut<dim>::type_dof_data, interp_2);

  dealii::Vector<float> subdomain(tria.n_active_cells());
  for (auto& s : subdomain)
    s = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");
  data_out.build_patches();

  dealii::DataOutBase::VtkFlags flags;
  flags.cycle = step;
  flags.time = t;
  data_out.set_flags(flags);

  const std::string filename = "ch_solution_";
  data_out.write_vtu_with_pvtu_record("./", filename, step, mpi_comm, 2, 8);
  return 0;
}

template<int dim>
void
CahnHilliardIDA<dim>::run()
{
  pcout << "=== Cahn-Hilliard IDA (dim=" << dim << ") ===\n";
  setup_mesh();
  setup_dofs();

  using VectorType = dealii::TrilinosWrappers::MPI::Vector;

  dealii::SUNDIALS::IDA<VectorType>::AdditionalData ida_data;
  ida_data.initial_time = 0.0;
  ida_data.final_time = prm.t_end;
  ida_data.initial_step_size = prm.dt_initial;
  ida_data.output_period = prm.t_end / 20.0;
  ida_data.ic_type =
    dealii::SUNDIALS::IDA<VectorType>::AdditionalData::use_y_diff;
  ida_data.reset_type =
    dealii::SUNDIALS::IDA<VectorType>::AdditionalData::use_y_diff;

  dealii::SUNDIALS::IDA<VectorType> ida(ida_data, mpi_comm);

  ida.differential_components = [&]() -> dealii::IndexSet {
    const auto dofs_per_component =
      dealii::DoFTools::count_dofs_per_fe_component(dof_handler);
    const dealii::types::global_dof_index n_c = dofs_per_component[0];

    // Return only the locally owned c-component DoFs (differential)
    // mu-component DoFs are omitted => algebraic
    dealii::IndexSet diff_set(dof_handler.n_dofs());
    for (const auto i : locally_owned_dofs)
      if (i < n_c)
        diff_set.add_index(i);
    diff_set.compress();
    return diff_set;
  };

  ida.reinit_vector = [&](VectorType& v) {
    v.reinit(locally_owned_dofs, mpi_comm);
  };

  ida.residual = [&](double t,
                     const VectorType& y,
                     const VectorType& y_dot,
                     VectorType& res) -> int {
    return assemble_residual(t, y, y_dot, res);
  };

  // ---- setup_jacobian: assemble J = alpha*M + K(y) and factor ILU ----
  ida.setup_jacobian = [&](double t,
                           const VectorType& y,
                           const VectorType& y_dot,
                           double alpha) -> int {
    (void)t;
    (void)y_dot;
    assemble_jacobian(y, alpha);
    return 0;
  };

  // ---- solve_with_jacobian: one preconditioned GMRES solve ----
  // IDA calls this with the current Newton RHS; tolerance is provided
  // by the integrator and should be met for tight Newton convergence.
  ida.solve_with_jacobian =
    [&](const VectorType& src, VectorType& dst, double tol) -> int {
    dealii::SolverControl sc(1000, tol);
    dealii::SolverGMRES<VectorType> gmres(sc);
    gmres.solve(jacobian_matrix, dst, src, preconditioner);
    return 0;
  };

  ida.output_step = [&](double t,
                        const VectorType& y,
                        const VectorType& y_dot,
                        unsigned int step) -> int {
    return output_step(t, y, y_dot, step);
  };

  // ---- Initial condition ----
  VectorType y(locally_owned_dofs, mpi_comm);
  VectorType y_dot(locally_owned_dofs, mpi_comm);

  {
    dealii::TrilinosWrappers::MPI::Vector y_tmp(
      locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    srand(dealii::Utilities::MPI::this_mpi_process(mpi_comm) + 42);
    dealii::VectorTools::interpolate(
      dof_handler, InitialCondition<dim>(), y_tmp);
    constraints.distribute(y_tmp);
    y = y_tmp;
  }

  y_dot = 0.0;
  y.compress(dealii::VectorOperation::insert);
  y_dot.compress(dealii::VectorOperation::insert);

  ida.solve_dae(y, y_dot);
  pcout << "Done.\n";
}

int
main(int argc, char* argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  try {
    CahnHilliardParameters prm;
    CahnHilliardIDA<2> solver(prm, MPI_COMM_WORLD);
    solver.run();
  } catch (std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << "\n";
    return 1;
  }
  return 0;
}
