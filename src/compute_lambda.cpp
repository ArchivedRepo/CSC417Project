#include <compute_lambda.h>
#include <build_grid.h>
#include <kernels.h>
#include <math.h>

#include <iostream>

void compute_lambda(
    Eigen::MatrixXd &positions,
    std::vector<int> &grid_result,
    double cube_s,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    std::vector<std::tuple<int, int>> &grid_indices,
    Eigen::VectorXd &lambdas, // assume caller has already resize this
    Eigen::VectorXd &phos,
    double pho0,
    double h_kernel, // smooth kernel parameter
    double epsilon,
    double mass, // mass of each particle
    int i // indicate which particle to compute
) {
    double pho = 0.0;

    Eigen::Vector3d curr_pos = positions.row(i);

    Eigen::Vector3d grad_C_j_not_same;
    Eigen::Vector3d grad_same;

    grad_C_j_not_same.setZero();
    grad_same.setZero();

    Eigen::Vector3d result;

    for (int j = 0; j < positions.rows(); j++){
        Eigen::Vector3d delta_r = curr_pos - (Eigen::Vector3d) positions.row(j);

        pho += poly6(delta_r, h_kernel);
        spiky_grad(delta_r, h_kernel, result);

        grad_same += result;

        if (i != j){
            spiky_grad(delta_r, h_kernel, result);
            grad_C_j_not_same -= result /pho0;
        }
    }
    grad_same /= pho0;

    double Ci = (pho /pho0) - 1;
    double lambdai = -Ci / (grad_same.norm() + grad_C_j_not_same.norm() + epsilon);

    lambdas(i) = lambdai;
    phos(i) = pho;

}