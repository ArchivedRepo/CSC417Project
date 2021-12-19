#include <viscosity_confinement.h>
#include <build_grid.h>
#include <kernels.h>

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
){

    int num_L = ceil((up_right(0) - bot_left(0)) / cube_s);
    int num_W = ceil((up_right(1) - bot_left(1)) / cube_s);
    int num_H = ceil((up_right(2) - bot_left(2)) / cube_s);

    Eigen::Vector3d curr_position = positions.row(index);
    Eigen::Vector3d curr_v = velocity.row(index);

    int this_index = compute_index(curr_position, bot_left, num_L, num_W, num_H, cube_s);

    int x_index, y_index, h_index;

    index_to_xyh(this_index, num_L, num_W, num_H, x_index, y_index, h_index);

    Eigen::Vector3d accu;
    accu.setZero();

    for (int x_offset = -1 ; x_offset < 2; x_offset++ ) {
        for (int y_offset = -1; y_offset < 2; y_offset++) {
            for (int h_offset = -1; h_offset < 2; h_offset++) {

                int cur_x, cur_y, cur_h, cur_index;

                cur_x = x_index + x_offset;
                cur_y = y_index + y_offset;
                cur_h = h_index + h_offset;

                if (cur_x < 0 || cur_y < 0 || cur_h < 0 || cur_x >= num_L || cur_y >= num_W || cur_h >= num_H) {
                    continue;
                }

                xyh_to_index(cur_x, cur_y, cur_h, num_L, num_W, num_H, cur_index);

                int start, end;

                if (cur_index == 0) {
                    start = 0;
                    end = grid_result[0];
                } else {
                    start = grid_result[cur_index - 1];
                    end = grid_result[cur_index];
                }

                for (int j = start; j < end; j++){
                    int target_id = std::get<1>(grid_indices[j]);
                    Eigen::Vector3d delta_p = curr_position - (Eigen::Vector3d) positions.row(target_id);

                    Eigen::Vector3d dv = (Eigen::Vector3d) velocity.row(target_id) - curr_v;
                    
                    double kernel_value = poly6(delta_p.norm(), h);
                    accu += dv * kernel_value;
                }
            }
        }
    }

    velocity.row(index) = (Eigen::Vector3d) velocity.row(index) + 0.001 * accu;
}