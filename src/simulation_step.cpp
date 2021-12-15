#include <simulation_step.h>
#include <advect.h>
#include <compute_lambda.h>
#include <compute_delta_position.h>
#include <update_position.h>
#include <update_velocity.h>
#include <viscosity_confinement.h>

void simulation_step(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &gravity,
    Eigen::Vector3d &bottom_left,
    Eigen::Vector3d &top_right,
    double dt,
    double h,
    double mass,
    double pho0, 
    double epsilon,
    double num_iteration
) {
    Eigen::MatrixXd positions_star;
    positions_star.resize(positions.rows(), 3);
    
    advect(positions, positions_star, velocity, gravity, dt);

    Eigen::VectorXd lambdas;
    lambdas.resize(positions.rows());
    lambdas.setZero();
    Eigen::MatrixXd delta_positions(positions.rows(), 3);

    Eigen::VectorXd phos;
    phos.resize(positions.rows());
    phos.setZero();

    for (int iter = 0; iter < num_iteration; iter++) {
        for (int i = 0; i < positions.rows(); i++) {
            compute_lambda(positions_star, pho0, mass, epsilon, h, lambdas, phos, i);
        }

        for (int i=0;i <positions_star.rows(); i++) {
            compute_delta_position(positions_star, pho0, h, lambdas, delta_positions, i);
        }

        update_positions(positions_star, delta_positions);

        update_velocity(positions, positions_star, velocity, dt);

        for (int i=0;i <positions_star.rows(); i++) {
            viscosity_confinement(positions_star, velocity, phos, h, i);
        }
        
    }
    positions = positions_star;
}