#include <gravity_matrix.h>


void gravity_matrix(
    Eigen::MatrixXd &g_m,
    Eigen::Vector3d &g_v,
    int N
) {
    Eigen::MatrixXd ones(N, 1);
    ones.setOnes();
    g_m = ones * g_v.transpose();
}