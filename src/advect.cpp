#include <advect.h>
#include <iostream>

void advect(
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &p0,
    Eigen::MatrixXd &p1,
    Eigen::Vector3d &accu,
    float dt
) {
    std::cout << "BEFORE: " << velocity.rows() << " " << velocity.cols() << std::endl;
    velocity = velocity + dt * accu;
    std::cout << "AFTER: " << velocity.rows() << " " << velocity.cols() << std::endl;
    p1.resize(p0.rows(), p0.cols());
    p1 = p0 + dt * velocity;
}