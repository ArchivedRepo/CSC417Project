#include <EigenTypes.h>


void simulation_step(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &gravity,
    Eigen::Vector3d &sim_space_bot_left,
    Eigen::Vector3d &sim_space_top_right,
    double cube_s,
    double dt,
    double h,
    double mass,
    double pho0, 
    double epsilon,
    double num_iteration
);