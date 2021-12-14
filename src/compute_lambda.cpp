#include <compute_lambda.h>
#include <build_grid.h>
#include <kernels.h>
#include <math.h>

#include <iostream>

void compute_lambda(
    Eigen::MatrixXd &positions,
    std::vector<int> &grid_result,
    double cube_s,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    std::vector<std::tuple<int, int>> &grid_indices,
    Eigen::VectorXd &lambdas, // assume caller has already resize this
    double pho0,
    double h_kernel, // smooth kernel parameter
    double epsilon,
    double mass, // mass of each particle
    int i // indicate which particle to compute
) {
    int L = ceil((up_right(0) - bot_left(0)) / cube_s);
    int W = ceil((up_right(1) - bot_left(1)) / cube_s);
    int H = ceil((up_right(2) - bot_left(2)) / cube_s);

    int this_index = compute_index(positions.row(i), bot_left,
    L, W, H, cube_s);
    int x, y, h;
    index_to_xyh(this_index, L, W, H, x, y, h);

    double pho = 0.0;
    Eigen::Vector3d gradient_i;
    double gradient_sum;

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
                    pho += mass * poly6(r, h);
                    Eigen::Vector3d local_grad;
                    spiky_grad(r, h, local_grad);
                    local_grad /= pho0;
                    gradient_i += local_grad;
                    if (j != i) {
                        gradient_sum += local_grad.squaredNorm();
                    }
                }
            }
        }
    }
    double C_i = (pho / pho0) - 1.0;
    double denominator = gradient_sum + gradient_i.squaredNorm() + epsilon;
    lambdas(i) = - C_i / denominator;

}