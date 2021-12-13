#include <update_position.h>
#include <cmath>
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
        result(1) = std::max(std::min(result(1), up_right(0) - BOUND_LIMIT), bot_left(1) + BOUND_LIMIT);
        result(2) = std::max(std::min(result(2), up_right(2) - BOUND_LIMIT), bot_left(2) + BOUND_LIMIT);
}

void update_position(
    Eigen::Matrix3d &positions,
    Eigen::Matrix3d &new_positions,
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
) {
    int L = up_right(0) - bot_left(0);
    int W = up_right(1) - bot_left(1);
    int H = up_right(2) - bot_left(2);

    int this_index = compute_index(positions.row(i), bot_left,
    L, W, H, cube_s);
    int x, y, h;
    index_to_xyh(this_index, L, W, H, x, y, h);

    Eigen::Vector3d delta_position;
    delta_position.setZero();

    for (int x_offset=-1; x_offset < 2; x_offset++ ) {
        for (int y_offset=-1; y_offset<2; y_offset++) {
            for (int h_offset=-1; h_offset<2; h_offset++) {
                int cur_x, cur_y, cur_h, cur_index;
                cur_x = x + x_offset;
                cur_y = y + y_offset;
                cur_h = h + h_offset;

                if (cur_x < 0 || cur_y < 0 || cur_h < 0 || cur_x >= L || cur_y >= W || cur_h >= H) {
                    continue;
                }

                xyh_to_index(cur_x, cur_y, cur_h, L, W, H, cur_index);
                int start, end;
                if (cur_index == 0) {
                    start = 0;
                    end = grid_result[0];
                } else {
                    start = grid_result[cur_index - 1];
                    end = grid_result[cur_index];
                }

                for (int j = start; start < end; start++) {
                    int cur_particle_idx = std::get<1>(grid_indices[j]);
                    Eigen::Vector3d r = positions.row(i)-positions.row(j);
                    Eigen::Vector3d local_grad;
                    spiky_grad(r, h, local_grad);
                    Eigen::Vector3d tmp;
                    tmp << delta_q, 0.0, 0.0;
                    double s_corr = -k * std::pow(poly6(r, h)/poly6(tmp, h), n_coor);
                    delta_position += (1/pho0) * (s_corr + lambdas(i)+lambdas(j)) * local_grad;
                }

                Eigen::Vector3d tmp;
                tmp = (Eigen::Vector3d)positions.row(i) + delta_position;
                apply_boundry(tmp, bot_left, up_right);
                new_positions.row(i) = tmp;
                
            }
        }
    }
}