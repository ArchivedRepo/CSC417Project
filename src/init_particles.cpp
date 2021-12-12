#include <init_particles.h>

#define STEP 0.2

void init_particles(
    Eigen::MatrixXd &positions,
    Eigen::Vector3d bot_left_pos,
    unsigned int x_count,
    unsigned int y_count,
    unsigned int h_count
) {
    int i = 0;
    positions.resize(x_count * y_count * h_count, 3);
    double cur_h = bot_left_pos(2); 
    double cur_y, cur_x;
    for (int h = 0; h < h_count; h++) {
        cur_x = bot_left_pos(0);
        for (int x=0; x < x_count; x++) {
            cur_y = bot_left_pos(1);
            for (int y =0; y < y_count; y++) {
                positions.row(i) << cur_x, cur_y, cur_h;
                cur_y += STEP;
                i++;
            }
            cur_x += STEP;
        }
        cur_h += STEP;
    }
}