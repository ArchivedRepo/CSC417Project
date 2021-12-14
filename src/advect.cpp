#include <advect.h>

void advect(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &positions_star,
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &accu,
    double dt
) {
    velocity = velocity + dt * accu;
    positions_star = positions + dt * velocity;
}