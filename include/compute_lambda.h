#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <kernels.h>

void compute_lambda(
    Eigen::MatrixXd &positions,
    std::vector<int> &grid_result,
    double cube_s,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    std::vector<std::tuple<int, int>> &grid_indices,
    Eigen::VectorXd &lambdas, // assume caller has already resize this
    double pho0,
    double h_kernel, // smooth kernel parameter
    double epsilon,
    double mass, // mass of each particle
    int i // indicate which particle to compute
);