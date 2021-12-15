#include <Eigen/Dense>
#include <Eigen/Core>

void update_velocity(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::VectorXd &phos,
    double h, 
    int &index
);