#include <EigenTypes.h>

void compute_delta_position(
    Eigen::MatrixXd &positions,
    double pho0,
    double h,
    Eigen::VectorXd &lambdas,
    Eigen::MatrixXd &delta_positions,
    int i
);