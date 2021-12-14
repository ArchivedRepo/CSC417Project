#include <build_grid.h>
#include <iostream>
#include <assert.h>

int compute_index(
    Eigen::Vector3d position,
    Eigen::Vector3d &bot_left,
    int L, int W, int H, double cube_s
) {
    Eigen::Vector3d relative_pos = position - bot_left;
    // TODO: THIS COULD CAUSE CASTING ERROR
    Eigen::Vector3i int_pos;
    for (int i = 0; i < 3; i++) {
        int_pos(i) = (int)(relative_pos(i) / cube_s);
    }
    int h = int_pos(2) * (W * L);
    int w = int_pos(1) * L;
    return h + int_pos(0);
}


void index_to_xyh(int index, int L, int W, int H, int &x, int &y, int &h) {
    h = index / (L * W);
    int plane_coor = h % (L * W);
    x = plane_coor % L;
    y = plane_coor / L;
}

void xyh_to_index(int x, int y, int h, int L, int W, int H, int &index) {
    index = h * (W * L) + y * L + x;
}

void build_grid(
    Eigen::MatrixXd &positions,
    std::vector<int> &result,
    double cube_s,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    std::vector<std::tuple<int, int>> &grid_indices // tuple<grid_index, particle_index>
) {
    int L = up_right(0) - bot_left(0);
    int W = up_right(1) - bot_left(1);
    int H = up_right(2) - bot_left(2);

    int N = positions.rows();
    grid_indices.clear();
    for (int i =0 ; i < N; i++) {
        grid_indices.push_back(
            std::make_tuple(
            compute_index(positions.row(i), bot_left, L, W, H, cube_s),
            i
            )
        );
    }
    std::sort(grid_indices.begin(), grid_indices.end());
    result.clear();

    int cur_grid = 0;
    for (int i = 0; i < N; i++) {
        std::tuple<int, int> this_tuple = grid_indices[i];
        int grid_index = std::get<0>(this_tuple);
        int particle_index = std::get<0>(this_tuple);
        
        assert(particle_index >= 0);
        // grid index
        // cur_grid = 5
        // 0, 0, 0, 1, 4, 4, 5
        // 0, 1, 2, 3, 4, 5, 6
        // 3, 4, 4, 6
        if (grid_index == cur_grid + 1) {
            result.push_back(i);
            cur_grid += 1;
        } else if (grid_index != cur_grid) {
            // grid_index > cur_grid+1
            while (grid_index != cur_grid + 1) {
                result.push_back(i);
                cur_grid += 1;
            }
            result.push_back(i);
            cur_grid += 1;
        }
        if (i == N - 1) {
            while (result.size() < L * W * H) {
                result.push_back(N);
            }
        }
    }
}