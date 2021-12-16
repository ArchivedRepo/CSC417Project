#include <compute_delta_position.h>
#include <kernels.h>
#include <build_grid.h>

void compute_delta_position(
    Eigen::MatrixXd &positions,
    double pho0,
    double h,
    Eigen::VectorXd &lambdas,
    Eigen::MatrixXd &delta_positions,
    std::vector<int> &grid_result,
    std::vector<std::tuple<int, int>> &grid_indices,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    double cube_s,
    int i
) {
    int num_L = ceil((up_right(0) - bot_left(0)) / cube_s);
    int num_W = ceil((up_right(1) - bot_left(1)) / cube_s);
    int num_H = ceil((up_right(2) - bot_left(2)) / cube_s);

    Eigen::Vector3d p_i = positions.row(i);

    int this_index = compute_index(p_i, bot_left, num_L, num_W, num_H, cube_s);

    int x_index, y_index, h_index;

    index_to_xyh(this_index, num_L, num_W, num_H, x_index, y_index, h_index);

    Eigen::Vector3d tmp;
    tmp.setZero();

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

                for (int j = start; j < end; j++) {
                    int target_id = std::get<1>(grid_indices[j]);

                    Eigen::Vector3d p_j = positions.row(target_id);
                    Eigen::Vector3d diff = p_i - p_j;
                    
                    Eigen::Vector3d local_grad;
                    spiky_grad(diff, h, local_grad);

                    double s_corr = -0.1 * pow(poly6(diff.norm(), h) / poly6(0.1 * h, h), 4.0);
                    // double s_corr = 0.0;

                    tmp += (lambdas(i) + lambdas(target_id) + s_corr) * local_grad;
                }
            }
        }
    }

    delta_positions.row(i) = (1/pho0) * tmp;
}
