#include <EigenTypes.h>

void viscosity_confinement(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    double h,
    std::vector<int> &grid_result,
    std::vector<std::tuple<int, int>> &grid_indices,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    double cube_s, 
    int &index
);