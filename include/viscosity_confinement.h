#include <EigenTypes.h>

void viscosity_confinement(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::Vector3d &up_right,
    Eigen::Vector3d &bot_left,
    Eigen::MatrixXd &delta_positions,
    std::vector<int> &grid_result,
    std::vector<std::tuple<int, int>> &grid_indices,
    double cube_s,
    double h, 
    int index
);