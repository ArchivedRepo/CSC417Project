#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <kernels.h>

void compute_lambda(
    Eigen:MatrixXd &positions,
    std:vector<int> &grid_result,
    double cube_s,
    Eigen::Vector3d &bot_left,
    EIgen::Vector3d &up_right,
    std::vector<std::tuple<int, int>> &grid_indices,
    Eigen::VectorXd &lambdas,
    double pho0,
    Eigen::VectorXd &phos,
    double h,
    double sigma,
    int i // indicate which particle to compute
) {
    
}