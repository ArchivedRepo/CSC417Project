// #include <init_particles.h>

// void init_particles(
//     Eigen::MatrixXd &positions,
//     Eigen::Vector3d &bot_left,
//     double step,
//     int x_count, int y_count, int z_count
// ) {
//     double cur_x, cur_y, cur_z;

//     int N = x_count * y_count * z_count;
//     positions.resize(N, 3);

//     int row = 0;
//     cur_z = bot_left(2);
//     for (int z = 0; z < z_count; z++) {
//         cur_y = bot_left(1);
//         for (int y = 0; y < y_count; y++) {
//             cur_x = bot_left(0);
//             for (int x = 0; x < x_count; x++) {
//                 positions.row(row) << cur_x, cur_y, cur_z;
//                 cur_x += step;
//                 row++;
//             }
//             cur_y += step;
//         }
//         cur_z += step;
//     }
//     assert(row == N);
// }