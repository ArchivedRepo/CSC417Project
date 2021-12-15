#include <update_position.h>
#include <cmath>
#include <kernels.h>
#include <iostream>

#define BOUND_LIMIT 1e-3

// Apply boundry condition, squeeze the particle back into the box if it goes out
// of bound.
static void apply_boundry(
    Eigen::MatrixXd &positions, 
    Eigen::MatrixXd &velocity,
    Eigen::Vector3d bot_left,
    Eigen::Vector3d up_right,
    int i
){
        
    double x = std::min(std::max(positions(i, 0), bot_left(0) + BOUND_LIMIT), up_right(0) - BOUND_LIMIT);
    double y = std::min(std::max(positions(i, 1), bot_left(1) + BOUND_LIMIT), up_right(1) - BOUND_LIMIT);
    double z = std::min(std::max(positions(i, 2), bot_left(2) + BOUND_LIMIT), up_right(2) - BOUND_LIMIT);

    if (x != positions(i, 0)){
        // positions(i, 0) = x;
        velocity(i, 0) = 0.0;
    }

    if (y != positions(i, 1)){
        positions(i, 1) = y;
        velocity(i, 1) = 0.0;
    }

    if (z != positions(i, 2)){
        positions(i, 2) = z;
        velocity(i, 2) = 0.0;
    }
}

void update_position(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &new_positions,
    std::vector<int> grid_result,
    double cube_s,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    std::vector<std::tuple<int, int>> &grid_indices,
    Eigen::VectorXd &lambdas,
    Eigen::MatrixXd &velocity,
    double pho0,
    double h_kernels,
    double epsilon,
    double k, // k for tensile instability
    double delta_q,// delta_q for tensile instability
    double n_coor, // n for compute tensile instability
    double i
) {
    Eigen::Vector3d delta_p;
    delta_p.setZero();

    Eigen::Vector3d curr_pos = positions.row(i);

    double curr_lambda = lambdas(i);

    Eigen::Vector3d result;
    
    Eigen::Vector3d proxy;
    proxy.setZero();
    proxy(0) = delta_q;

    for(int j = 0; j < positions.rows(); j++){
        Eigen::Vector3d delta_pos = curr_pos - (Eigen::Vector3d) positions.row(j);

        spiky_grad(delta_pos, h_kernels, result);
        if (result(0) < 1e-14 && j % 24 == 0){
            // std::cout << (Eigen::RowVector3d) curr_pos << "|"<< (Eigen::RowVector3d)  positions.row(j) << " " <<std::endl;
        }
        double s_corr = -k * pow(poly6(delta_pos, h_kernels)/poly6(proxy, h_kernels), n_coor);
        delta_p += (curr_lambda + lambdas(j) + s_corr) * result;
    }

    delta_p /= pho0;
    new_positions.row(i) = positions.row(i) + (Eigen::RowVector3d) delta_p;

    // apply_boundry(new_positions, velocity, bot_left, up_right, i);
}