// #include <advect.h>
// #include <update_position.h>

// void advect(
//     Eigen::MatrixXd &positions,
//     Eigen::MatrixXd &positions_star,
//     Eigen::MatrixXd &velocity,
//     Eigen::MatrixXd &accu,
//     Eigen::Vector3d &bottom_left,
//     Eigen::Vector3d &top_right,
//     double dt
// ) {
//     velocity = velocity + dt * accu;
//     positions_star = positions + dt * velocity;

//     for (int i = 0; i < velocity.rows(); i++){
//         Eigen::Vector3d tmp = positions_star.row(i);
//         Eigen::Vector3d tmp_v = velocity.row(i);
//         apply_boundry(tmp, tmp_v, bottom_left, top_right);
//         positions_star.row(i) = tmp;
//         velocity.row(i) = tmp_v;
//     }
// }