#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>


/*
Build a grid of particles based on particle location.

positions: position of particle, N * 3

indices: index into positions, of size N, this will be sorted based on grid index

result: result[i-1], result[i] is the start and end index into indices to
indicate the segment of particles in a grid 

cube_s: size of single cuba

bot_left, up_right: the position of bot_left and up_right of the whole cuba in
which the particles move

*/
void build_grid(
    Eigen::MatrixXd &positions,
    std::vector<int> &result,
    double cube_s,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    std::vector<std::tuple<int, int>> &grid_indices
);