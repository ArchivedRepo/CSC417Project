#include <EigenTypes.h>

int compute_index(
    Eigen::Vector3d position,
    Eigen::Vector3d &bot_left,
    int L, int W, int H, double cube_s
);

void index_to_xyh(int index, int L, int W, int H, int &x, int &y, int &h);

void xyh_to_index(int x, int y, int h, int L, int W, int H, int &index);

void build_grid(
    Eigen::MatrixXd &positions,
    std::vector<int> &result,
    double cube_s,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    std::vector<std::tuple<int, int>> &grid_indices // tuple<grid_index, particle_index>
);