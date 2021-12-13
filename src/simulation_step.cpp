#include <simulation_step.h>
#include <advect.h>
#include <build_grid.h>
#include <compute_lambda.h>
#include <update_position.h>
#include <update_velocity.h>

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
    int N = positions.rows();
    Eigen::MatrixXd position_star;
    advect(velocity, positions, position_star, gravity, dt);

    std::vector<int> grid_result;
    std::vector<std::tuple<int, int>> grid_indices;

    build_grid(position_star, grid_result, cube_s, bot_left, up_right, grid_indices);
    Eigen::VectorXd lambdas;
    lambdas.resize(positions.rows());
    lambdas.setZero();

    Eigen::MatrixXd tmp;
    tmp.resize(N, 3);
    for (int i = 0; i < num_iterations; i++) {
        for (int idx=0; idx < N; idx++) { 
            compute_lambda(position_star, grid_result, cube_s, bot_left, up_right,
            grid_indices, lambdas, pho0, h_kernel, epsilon, mass, idx);
        }
        tmp.setZero();
        for (int idx=0; idx<N; idx++) {
            update_position(position_star, tmp, grid_result, cube_s, 
            bot_left, up_right, grid_indices, lambdas, pho0, h_kernel, epsilon,
            k, delta_q, n_coor, idx);
        }
        position_star = tmp;
    }
    update_velocity(positions, position_star, dt, velocity);
    positions = position_star;
}