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
main()
{
  using VectorType = dealii::Vector<double>;

  VectorType y(2);
  VectorType y_dot(2);

  double kappa = 1.0;

  dealii::FullMatrix<double> A(2, 2);
  A(0, 1) = -1.0;
  A(1, 0) = kappa * kappa;

  dealii::FullMatrix<double> J(2, 2);
  dealii::FullMatrix<double> J_inv(2, 2);

  dealii::SUNDIALS::IDA<VectorType>::AdditionalData time_stepper_data;
  time_stepper_data.final_time = 10.0;

  dealii::SUNDIALS::IDA<VectorType> time_stepper(time_stepper_data);

  time_stepper.reinit_vector = [&](VectorType& v) { v.reinit(2); };

  time_stepper.residual = [&](const double t,
                              const VectorType& y,
                              const VectorType& y_dot,
                              VectorType& residual) {
    residual = y_dot;
    A.vmult_add(residual, y);
  };

  time_stepper.setup_jacobian = [&](const double t,
                                    const VectorType& y,
                                    const VectorType& y_dot,
                                    const double alpha) {
    J = A;
    J(0, 0) += alpha;
    J(1, 1) += alpha;

    J_inv.invert(J);
  };

  time_stepper.solve_with_jacobian =
    [&](const VectorType& src, VectorType& dst, const double tolerance) {
      J_inv.vmult(dst, src);
    };

  time_stepper.output_step = [&](const double t,
                                 const VectorType& y,
                                 const VectorType& y_dot,
                                 const unsigned int step_number) {
    std::cout << "Step " << step_number << " time " << t << "\n";
  };

  y[1] = kappa;
  y_dot[0] = kappa;

  time_stepper.solve_dae(y, y_dot);

  return 0;
}
