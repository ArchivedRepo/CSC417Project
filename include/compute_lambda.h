#include <EigenTypes.h>

void compute_lambda(
    Eigen::MatrixXd &positions,
    double pho0,
    double mass,
    double epsilon,
    double h,
    Eigen::VectorXd &lambdas,
    std::vector<int> &grid_result,
    std::vector<std::tuple<int, int>> &grid_indices,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    double cube_s,
    int i
);