#include <compute_lambda.h>
#include <kernels.h>


void compute_lambda(
    Eigen::MatrixXd &positions,
    double pho0,
    double mass,
    double epsilon,
    double h,
    Eigen::VectorXd &lambdas,
    int i
) {
    Eigen::Vector3d p_i = positions.row(i);

    double pho = 0.0;
    Eigen::Vector3d grad_i;
    grad_i.setZero();
    double grad_sum = 0.0;
    for (int j = 0; j < positions.rows(); j++) {
        Eigen::Vector3d p_j = positions.row(j);
        Eigen::Vector3d diff = p_i - p_j;
        pho += mass * poly6(diff.norm(), h);

        Eigen::Vector3d local_grad;
        spiky_grad(diff, h, local_grad);
        grad_i += local_grad;

        if (j != i) {
            grad_sum += (1.0/pho0)*(1.0/pho0) * local_grad.squaredNorm();
        }
    }
    double C_i = (pho/pho0) - 1.0;
    double denominator = grad_sum + (1.0/pho0)*(1.0/pho0)*grad_i.squaredNorm();
    lambdas(i) =-C_i / (denominator+epsilon);
}