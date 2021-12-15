#include <EigenTypes.h>


void simulation_step(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &gravity,
    Eigen::Vector3d &bottom_left,
    Eigen::Vector3d &top_right,
    double dt,
    double h,
    double mass,
    double pho0, 
    double epsilon,
    double num_iteration
);