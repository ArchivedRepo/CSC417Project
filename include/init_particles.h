#include <Eigen/Dense>
#include <Eigen/Core>


/* 
Generate a cude of points where the bottom left corner of the cube is at
bot_left and top right corner is at up_right, place particle each step away
*/
void init_particles(
    Eigen::MatrixXd &positions,
    Eigen::Vector3d bot_left,
    Eigen::Vector3d up_right,
    double step
);