#include <EigenTypes.h>

void compute_lambda(
    Eigen::MatrixXd &positions,
    double pho0,
    double mass,
    double epsilon,
    double h,
    Eigen::VectorXd lambdas,
    int i
);