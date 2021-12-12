#include <advect.h>


void advect(
    Eigen::MatrixXd velocity,
    Eigen::MatrixXd p0,
    Eigen::MatrixXd p1,
    Eigen::Vector3d accu,
    float dt
) {
    velocity = velocity + dt * accu;
    p1.resize(p0.rows(), p0.cols());
    p1 = p0 + dt * velocity;
}