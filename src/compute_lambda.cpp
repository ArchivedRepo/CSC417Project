#include <compute_lambda.h>
#include <kernels.h>
#include <build_grid.h>


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
) {

    int num_L = ceil((up_right(0) - bot_left(0)) / cube_s);
    int num_W = ceil((up_right(1) - bot_left(1)) / cube_s);
    int num_H = ceil((up_right(2) - bot_left(2)) / cube_s);

    Eigen::Vector3d p_i = positions.row(i);

    int this_index = compute_index(p_i, bot_left, num_L, num_W, num_H, cube_s);

    int x_index, y_index, h_index;

    index_to_xyh(this_index, num_L, num_W, num_H, x_index, y_index, h_index);

    double pho = 0.0;

    Eigen::Vector3d grad_i;
    grad_i.setZero();

    double grad_sum = 0.0;

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
                    pho += mass * poly6(diff.norm(), h);

                    Eigen::Vector3d local_grad;
                    spiky_grad(diff, h, local_grad);
                    grad_i += local_grad;

                    if (target_id != i) {
                        grad_sum += (1.0/pho0)*(1.0/pho0) * local_grad.squaredNorm();
                    }
                }
            }
        }
    }
    double C_i = (pho/pho0) - 1.0;
    double denominator = grad_sum + (1.0/pho0)*(1.0/pho0)*grad_i.squaredNorm();
    lambdas(i) =-C_i / (denominator+epsilon);
}