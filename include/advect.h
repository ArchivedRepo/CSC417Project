#include <Eigen/Dense>
#include <Eigen/Core>

/* Compute fluid advect

velocity: velocity of particle, in-place updated, N * 3
p0: current position of particle, N*3
p1: predicted position, N*3
accu: accelaration
dt: time step
*/
void advect(
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &p0,
    Eigen::MatrixXd &p1,
    Eigen::Vector3d &accu,
    float dt
);