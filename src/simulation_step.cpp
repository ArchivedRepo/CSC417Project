#include <simulation_step.h>
#include <advect.h>
#include <compute_lambda.h>
#include <compute_delta_position.h>
#include <update_position.h>
#include <update_velocity.h>
#include <viscosity_confinement.h>
#include <build_grid.h>
#include <iostream>
#include <time.h>

void simulation_step(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &gravity,
    Eigen::Vector3d &sim_space_bot_left,
    Eigen::Vector3d &sim_space_top_right,
    double cube_s,
    double dt,
    double h,
    double mass,
    double pho0, 
    double epsilon,
    double num_iteration
) { 
    clock_t t = clock();

    Eigen::MatrixXd positions_star;
    positions_star.resize(positions.rows(), 3);
    
    advect(positions, positions_star, velocity, gravity, sim_space_bot_left, sim_space_top_right, dt);

    std::vector<int> result;
    result.clear();

    std::vector<std::tuple<int, int>> grid_indices;
    grid_indices.clear();

    build_grid(positions_star, result, cube_s, sim_space_bot_left, sim_space_top_right, grid_indices);

    Eigen::VectorXd lambdas;
    lambdas.resize(positions.rows());
    lambdas.setZero();

    Eigen::MatrixXd delta_positions(positions.rows(), 3);
    delta_positions.setZero();

    for (int iter = 0; iter < num_iteration; iter++) {
        for (int i = 0; i < positions.rows(); i++) {
            compute_lambda(positions_star, pho0, mass, epsilon, h, lambdas, result, 
            grid_indices, sim_space_bot_left, sim_space_top_right, cube_s, i);
        }
        for (int i=0;i <positions_star.rows(); i++) {
            compute_delta_position(positions_star, pho0, h, lambdas, delta_positions, 
            result, grid_indices, sim_space_bot_left, sim_space_top_right, cube_s, i);
        }
        update_positions(positions_star, delta_positions, velocity, sim_space_bot_left, sim_space_top_right);
    }
    update_velocity(positions, positions_star, velocity, dt);
    
    for (int i = 0; i < velocity.rows(); i++){
        viscosity_confinement(positions_star, velocity, sim_space_top_right, 
        sim_space_bot_left, delta_positions, result, grid_indices, cube_s, h, i);
    }
    positions = positions_star;

    t = clock() - t;
    double diff = (double)t / CLOCKS_PER_SEC;
    std::cout << diff << std::endl;
}