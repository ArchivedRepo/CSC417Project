#include <init_particles.h>

void init_particles(
    Eigen::MatrixXd &positions,
    Eigen::Vector3d bot_left,
    Eigen::Vector3d up_right,
    double step
) {

    int x_count, y_count, h_count;
    double L, W, H;
    L = up_right(0) - bot_left(0);
    W = up_right(1) - bot_left(1);
    H = up_right(2) - bot_left(2);
    x_count = (int)(L / step);
    y_count = (int)(W / step);
    h_count = (int)(H / step);
    int i = 0;
    positions.resize(x_count * y_count * h_count, 3);
    double cur_h = bot_left(2); 
    double cur_y, cur_x;
    for (int h = 0; h < h_count; h++) {
        cur_x = up_right(0);
        for (int x=0; x < x_count; x++) {
            cur_y = bot_left(1);
            for (int y =0; y < y_count; y++) {
                positions.row(i) << cur_x, cur_y, cur_h;
                cur_y += step;
                i++;
            }
            cur_x += step;
        }
        cur_h += step;
    }
}