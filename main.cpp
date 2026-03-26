#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_ts.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

constexpr int problem_degree = 1;
constexpr int problem_dim = 2;

using VectorType = dealii::PETScWrappers::MPI::Vector;
using BlockVectorType = dealii::PETScWrappers::MPI::BlockVector;
using MatrixType = dealii::PETScWrappers::MPI::SparseMatrix;
using Preconditioner = dealii::PETScWrappers::PreconditionSSOR;

template<int dim>
class InitialCondition : public dealii::Function<dim>
{
public:
  InitialCondition()
    : dealii::Function<dim>(2) {};

  void vector_value([[maybe_unused]] const dealii::Point<dim>& p,
                    dealii::Vector<double>& values) const override
  {
    values[0] = 0.5 + 0.05 * (2.0 * (double)rand() / RAND_MAX - 1.0); // c
    values[1] = 0.0;                                                  // mu
  }
};

template<int dim>
class ImplicitCahnHilliard : public dealii::ParameterAcceptor
{
public:
  ImplicitCahnHilliard(const MPI_Comm comm);
  void run();

private:
  void setup_system();

  void output_results(const double t,
                      const unsigned int step,
                      const VectorType& y);

  void implicit_function(const double t,
                         const VectorType& y,
                         const VectorType& y_dot,
                         VectorType& res);

  void assemble_implicit_jacobian(const double t,
                                  const VectorType& y,
                                  const VectorType& y_dot,
                                  double alpha);

  void solve_with_jacobian(const VectorType& src, VectorType& res);

  const MPI_Comm mpi_comm;
  dealii::ConditionalOStream pcout;
  dealii::TimerOutput timer;

  dealii::parallel::distributed::Triangulation<dim> tria;
  dealii::FESystem<dim> fe;
  dealii::DoFHandler<dim> dof_handler;
  dealii::AffineConstraints<double> constraints;
  dealii::IndexSet locally_owned_dofs;
  dealii::IndexSet locally_relevant_dofs;

  VectorType y_ghosted;
  VectorType y_dot_ghosted;
  MatrixType jacobian_matrix;

  Preconditioner preconditioner;

  dealii::PETScWrappers::TimeStepperData time_stepper_data;

  unsigned int n_refinements = 0;
  unsigned int n_outputs = 0;

  double M = 0.0;
  double epsilon = 0.0;
};

template<int dim>
ImplicitCahnHilliard<dim>::ImplicitCahnHilliard(MPI_Comm comm)
  : dealii::ParameterAcceptor("/Implicit Cahn Hilliard/")
  , mpi_comm(comm)
  , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0)
  , timer(comm,
          pcout,
          dealii::TimerOutput::summary,
          dealii::TimerOutput::wall_times)
  , tria(comm)
  , fe(dealii::FE_Q<dim>(problem_degree), 2)
  , dof_handler(tria)
  , time_stepper_data("", "beuler", 0.0, 1.0, 1.0e-5)
{
  enter_subsection("Time stepper");
  {
    enter_my_subsection(this->prm);
    {
      time_stepper_data.add_parameters(this->prm);
    }
    leave_my_subsection(this->prm);
  }
  leave_subsection();

  add_parameter("n refinements",
                n_refinements,
                "Number of times the mesh is refined globally before starting "
                "the time stepping.");
  add_parameter("n outputs",
                n_outputs,
                "Number of outputs over the course of the simulation");

  add_parameter("mobility", M);
  add_parameter("gradient energy", epsilon);
}

template<int dim>
void
ImplicitCahnHilliard<dim>::setup_system()
{
  dealii::TimerOutput::Scope local_timer(timer, "setup system");

  dealii::GridGenerator::hyper_cube(tria, 0.0, 100.0, true);
  tria.refine_global(n_refinements);

  dof_handler.distribute_dofs(fe);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs =
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  y_ghosted.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  y_dot_ghosted.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);

  dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  dealii::SparsityTools::distribute_sparsity_pattern(
    dsp, locally_owned_dofs, mpi_comm, locally_relevant_dofs);

  jacobian_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);

  pcout << std::endl
        << "Number of active cells: " << tria.n_active_cells() << std::endl
        << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl
        << std::endl;
}

template<int dim>
void
ImplicitCahnHilliard<dim>::output_results(const double t,
                                          const unsigned int step,
                                          const VectorType& y)
{
  dealii::TimerOutput::Scope local_timer(timer, "output results");

  pcout << "  Step " << step << "  t = " << t << "\n";

  y_ghosted = y;

  std::vector<std::string> names = { "c", "mu" };
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    interp(2, dealii::DataComponentInterpretation::component_is_scalar);

  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(
    y_ghosted, names, dealii::DataOut<dim>::type_dof_data, interp);

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
ImplicitCahnHilliard<dim>::implicit_function([[maybe_unused]] double t,
                                             const VectorType& y,
                                             const VectorType& y_dot,
                                             VectorType& res)
{
  dealii::TimerOutput::Scope local_timer(timer, "implicit function");

  res = 0.0;

  const dealii::QGauss<dim> quadrature(problem_degree + 2);
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
        cell_rhs[i] += (c_dot * phi_c + M * (grad_mu_vals[q] * gphi_c)) * JxW;

        // R_mu
        cell_rhs[i] += ((mu - df_dc) * phi_mu -
                        epsilon * epsilon * (grad_c_vals[q] * gphi_mu)) *
                       JxW;
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(cell_rhs, local_dof_indices, res);
  }

  res.compress(dealii::VectorOperation::add);
}

template<int dim>
void
ImplicitCahnHilliard<dim>::assemble_implicit_jacobian(
  [[maybe_unused]] const double t,
  const VectorType& y,
  [[maybe_unused]] const VectorType& y_dot,
  double alpha)
{
  dealii::TimerOutput::Scope local_timer(timer, "assemble implicit Jacobian");

  jacobian_matrix = 0.0;

  const dealii::QGauss<dim> quadrature(problem_degree + 2);
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
            cell_matrix(i, j) += M * (gphi_mu_j * gphi_c_i) * JxW;

          // J_μ_c
          if (i_is_mu && j_is_c)
            cell_matrix(i, j) += (-d2f_dc * phi_c_j * phi_mu_i -
                                  epsilon * epsilon * (gphi_c_j * gphi_mu_i)) *
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
}

template<int dim>
void
ImplicitCahnHilliard<dim>::solve_with_jacobian(const VectorType& src,
                                               VectorType& res)
{
  dealii::TimerOutput::Scope local_timer(timer, "solve with Jacobian");

  preconditioner.initialize(jacobian_matrix);

  dealii::SolverControl solver_control(1000, 1.0e-8 * src.l2_norm());
  dealii::PETScWrappers::SolverGMRES gmres(solver_control);
  gmres.set_prefix("user_");

  gmres.solve(jacobian_matrix, res, src, preconditioner);
}

template<int dim>
void
ImplicitCahnHilliard<dim>::run()
{
  pcout << "Implicit Cahn-Hilliard (dim=" << dim << ")\n";

  setup_system();

  dealii::PETScWrappers::TimeStepper<VectorType, MatrixType> petsc_ts(
    time_stepper_data);

  petsc_ts.set_matrices(jacobian_matrix, jacobian_matrix);

  petsc_ts.implicit_function = [&](const double time,
                                   const VectorType& y,
                                   const VectorType& y_dot,
                                   VectorType& res) {
    this->implicit_function(time, y, y_dot, res);
  };

  petsc_ts.setup_jacobian = [&](const double time,
                                const VectorType& y,
                                const VectorType& y_dot,
                                const double alpha) {
    this->assemble_implicit_jacobian(time, y, y_dot, alpha);
  };

  petsc_ts.solve_with_jacobian = [&](const VectorType& src, VectorType& dst) {
    this->solve_with_jacobian(src, dst);
  };

  petsc_ts.algebraic_components = [&]() {
    dealii::IndexSet algebraic_set(dof_handler.n_dofs());

    const dealii::IndexSet mu_dofs =
      dealii::DoFTools::locally_owned_dofs_per_component(dof_handler)[1];
    algebraic_set.add_indices(mu_dofs);
    algebraic_set.compress();
    return algebraic_set;
  };

  petsc_ts.monitor = [&](const double time,
                         const VectorType& y,
                         const unsigned int step_number) {
    const double output_interval = time_stepper_data.final_time / n_outputs;

    if (step_number == 0 || std::fmod(time, output_interval) <
                              time_stepper_data.initial_step_size) {
      pcout << "Time step " << step_number << " at t=" << time << std::endl;
      this->output_results(time, step_number, y);
    }
  };

  VectorType solution(locally_owned_dofs, mpi_comm);
  srand(dealii::Utilities::MPI::this_mpi_process(mpi_comm) + 42);
  dealii::VectorTools::interpolate(
    dof_handler, InitialCondition<dim>(), solution);

  petsc_ts.solve(solution);
}

int
main(int argc, char** argv)
{
  try {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    ImplicitCahnHilliard<problem_dim> problem(MPI_COMM_WORLD);

    const std::string input_filename = (argc > 1 ? argv[1] : "parameters.prm");
    dealii::ParameterAcceptor::initialize(input_filename,
                                          "parameters_used.prm");
    problem.run();
  } catch (std::exception& exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
