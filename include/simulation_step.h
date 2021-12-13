#include <Eigen/Dense>
#include <Eigen/Core>

void simulation_step(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::Vector3d gravity,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    double cube_s,
    int num_iterations,
    double dt,
    double pho0,
    double epsilon,
    double mass,
    double h_kernel,
    double k,
    double delta_q,
    double n_coor
);