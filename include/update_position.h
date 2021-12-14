#include <Eigen/Dense>
#include <Eigen/Core>

#include <build_grid.h>
#include <kernels.h>

#define BOUND_LIMIT 1e-3

// Apply boundry condition, squeeze the particle back into the box if it goes out
// of bound.
static void apply_boundry(
    Eigen::Vector3d &result, 
    Eigen::Vector3d bot_left, 
    Eigen::Vector3d up_right
){
        result(0) = std::max(std::min(result(0), up_right(0) - BOUND_LIMIT), bot_left(0) + BOUND_LIMIT);
        result(1) = std::max(std::min(result(1), up_right(1) - BOUND_LIMIT), bot_left(1) + BOUND_LIMIT);
        result(2) = std::max(std::min(result(2), up_right(2) - BOUND_LIMIT), bot_left(2) + BOUND_LIMIT);
}

void update_position(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &new_positions,
    std::vector<int> grid_result,
    double cube_s,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    std::vector<std::tuple<int, int>> &grid_indices,
    Eigen::VectorXd &lambdas,
    double pho0,
    double h_kernels,
    double epsilon,
    double k, // k for tensile instability
    double delta_q,// delta_q for tensile instability
    double n_coor, // n for compute tensile instability
    double i
);