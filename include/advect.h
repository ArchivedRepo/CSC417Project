#include <EigenTypes.h>

void advect(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &positions_star,
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &accu,
    double dt
);