#include <EigenTypes.h>

/*
Turn a gravity vector to a matrix. Need to do this because Eigen does not 
automatically broadcast!
*/
void gravity_matrix(
    Eigen::MatrixXd &g_m,
    Eigen::Vector3d &g_v,
    int N
);