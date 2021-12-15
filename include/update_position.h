#include <Eigen/Dense>
#include <Eigen/Core>

#include <build_grid.h>
#include <kernels.h>

void update_position(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &new_positions,
    std::vector<int> grid_result,
    double cube_s,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    std::vector<std::tuple<int, int>> &grid_indices,
    Eigen::VectorXd &lambdas,
    Eigen::MatrixXd &velocity,
    double pho0,
    double h_kernels,
    double epsilon,
    double k, // k for tensile instability
    double delta_q,// delta_q for tensile instability
    double n_coor, // n for compute tensile instability
    double i
);