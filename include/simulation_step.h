#include <EigenTypes.h>


void simulation_step(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &gravity,
    double dt,
    double h,
    double mass,
    double pho, 
    double epsilon,
    double num_iteration
);