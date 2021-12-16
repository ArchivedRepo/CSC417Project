#include <compute_delta_position.h>
#include <kernels.h>

void compute_delta_position(
    Eigen::MatrixXd &positions,
    double pho0,
    double h,
    Eigen::VectorXd &lambdas,
    Eigen::MatrixXd &delta_positions,
    int i
) {
    Eigen::Vector3d p_i = positions.row(i);

    Eigen::Vector3d tmp;
    tmp.setZero();
    for (int j=0; j < positions.rows(); j++) {
        Eigen::Vector3d p_j = positions.row(j);
        Eigen::Vector3d diff = p_i - p_j;
        
        Eigen::Vector3d local_grad;
        spiky_grad(diff, h, local_grad);

        double s_corr = -0.1 * pow(poly6(diff.norm(), h) / poly6(0.1 * h, h), 4.0);
        // double s_corr = 0.0;

        tmp += (lambdas(i)+lambdas(j) + s_corr) * local_grad;
    }
    delta_positions.row(i) = (1/pho0) * tmp;
}
