#include <Eigen/Dense>
#include <Eigen/Core>


/* 
Generate a cude of points where the bottom left corner of the cube is at
bottom_left_position.

The cude is x_count * y_count * h_count.
*/
void init_particles(
    Eigen::MatrixXd &positions,
    Eigen::Vector3d bottom_left_position,
    unsigned int x_count,
    unsigned int y_count,
    unsigned int h_count
);