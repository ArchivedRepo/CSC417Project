#include <simulation_step.h>
#include <advect.h>
#include <build_grid.h>
#include <compute_lambda.h>
#include <update_position.h>
#include <update_velocity.h>

#include <iostream>

void simulation_step(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::Vector3d gravity,
    Eigen::Vector3d &bot_left,
    Eigen::Vector3d &up_right,
    double cube_s,
    int num_iterations,
    double dt,
    double pho0,
    double epsilon,
    double mass,
    double h_kernel,
    double k,
    double delta_q,
    double n_coor
) {
    Eigen::MatrixXd new_position;
    new_position.resize(positions.rows(), 3);
    std::vector<std::tuple<int, int>> grid;
    std::vector<int> result;

    advect(velocity, positions, new_position,gravity, dt);

    Eigen::VectorXd lambdas;
    lambdas.resize(new_position.rows());
    lambdas.setZero();

    Eigen::VectorXd phos;
    phos.resize(new_position.rows());
    phos.setZero();

    for (int iteration = 0; iteration < 2; iteration++){

        std::cout << "==============================" << iteration << "==========================" << std::endl;
        std::cout << "position before 0 " << positions.row(0) << std::endl;
        for (int i = 0; i < positions.rows(); i++){
            compute_lambda(positions, result, cube_s, bot_left, up_right, grid, lambdas, phos, pho0, h_kernel, epsilon, mass, i);
        }
        std::cout << "position after 1 " << positions.row(0) << std::endl;
        for (int i = 0; i < positions.rows(); i++){
            update_position(positions, new_position, result, cube_s, bot_left, up_right, grid, lambdas, velocity, pho0, h_kernel, epsilon, k, delta_q, n_coor, i);
        }
        std::cout << "position before 2 " << new_position.row(0) << std::endl;
        for (int i = 0; i < positions.rows(); i++){
            velocity.row(i) = (new_position.row(i) - positions.row(i))/dt;
            update_velocity(positions, velocity, phos, h_kernel, i);
        }
        std::cout << "position after 3 " << new_position.row(0) << std::endl;

        positions = new_position;
    }

}