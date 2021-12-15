#include <EigenTypes.h>

void viscosity_confinement(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::VectorXd &phos,
    double h, 
    int &index
);