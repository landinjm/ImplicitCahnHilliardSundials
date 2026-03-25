#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/sundials/ida.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

int
main(int argc, char* argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPI_Comm communicator = MPI_COMM_WORLD;

  using VectorType = dealii::TrilinosWrappers::MPI::Vector;

  dealii::IndexSet locally_owned(2);
  locally_owned.add_range(0, 2);

  VectorType y, y_dot;
  y.reinit(locally_owned, communicator);
  y_dot.reinit(locally_owned, communicator);

  double kappa = 1.0;
  dealii::FullMatrix<double> A(2, 2);
  A(0, 1) = -1.0;
  A(1, 0) = kappa * kappa;

  dealii::FullMatrix<double> J(2, 2);
  dealii::FullMatrix<double> J_inv(2, 2);

  dealii::SUNDIALS::IDA<VectorType>::AdditionalData time_stepper_data;
  time_stepper_data.final_time = 10.0;

  dealii::SUNDIALS::IDA<VectorType> time_stepper(time_stepper_data,
                                                 communicator);

  time_stepper.reinit_vector = [&](VectorType& v) {
    v.reinit(locally_owned, communicator);
  };

  time_stepper.residual = [&](const double t,
                              const VectorType& y,
                              const VectorType& y_dot,
                              VectorType& residual) -> int {
    residual = y_dot;
    {
      dealii::Vector<double> tmp_in(2), tmp_out(2);
      for (unsigned int i = 0; i < 2; ++i)
        tmp_in[i] = y[i];
      A.vmult(tmp_out, tmp_in);
      for (unsigned int i = 0; i < 2; ++i)
        residual[i] += tmp_out[i];
    }
    return 0;
  };

  time_stepper.setup_jacobian = [&](const double t,
                                    const VectorType& y,
                                    const VectorType& y_dot,
                                    const double alpha) -> int {
    J = A;
    J(0, 0) += alpha;
    J(1, 1) += alpha;
    J_inv.invert(J);
    return 0;
  };

  time_stepper.solve_with_jacobian =
    [&](const VectorType& src, VectorType& dst, const double tolerance) -> int {
    dealii::Vector<double> tmp_src(2), tmp_dst(2);
    for (unsigned int i = 0; i < 2; ++i)
      tmp_src[i] = src[i];
    J_inv.vmult(tmp_dst, tmp_src);
    for (unsigned int i = 0; i < 2; ++i)
      dst[i] = tmp_dst[i];
    return 0;
  };

  time_stepper.output_step = [&](const double t,
                                 const VectorType& y,
                                 const VectorType& y_dot,
                                 const unsigned int step_number) -> int {
    std::cout << "Step " << step_number << " time " << t << "\n";
    return 0;
  };

  y[1] = kappa;
  y_dot[0] = kappa;

  y.compress(dealii::VectorOperation::insert);
  y_dot.compress(dealii::VectorOperation::insert);

  time_stepper.solve_dae(y, y_dot);
  return 0;
}
