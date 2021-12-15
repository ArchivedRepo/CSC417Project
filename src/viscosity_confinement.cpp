#include <kernels.h>

void viscosity_confinement(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::VectorXd &phos,
    double h, 
    int &index
){
    Eigen::Vector3d curr_position = positions.row(index);
    Eigen::Vector3d curr_v = velocity.row(index);
    double curr_pho = phos[index];

    Eigen::Vector3d accu;
    accu.setZero();

    for (int j = 0; j < positions.rows(); j++){
        Eigen::Vector3d delta_p = curr_position - (Eigen::Vector3d) positions.row(j);
        Eigen::Vector3d dv = (Eigen::Vector3d) velocity.row(j) - curr_v;
        double pho_sum = curr_pho + phos[j];
        double kernel_value = poly6(delta_p.norm(), h);
        accu += (dv/pho_sum)*kernel_value;
    }

    velocity.row(index) += accu;
}