#include <update_position.h>

void update_positions(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &delta_positions,
    double dt
) {
    positions = positions + dt * delta_positions;
}