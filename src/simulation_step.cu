#include <simulation_step.cuh>
#include <advect.cuh>
#include <mem_util.cuh>
#include <build_grid.cuh>
#include <compute_lambda.cuh>
// #include <compute_delta_position.h>
// #include <update_position.h>
// #include <update_velocity.h>
// #include <viscosity_confinement.h>
// #include <build_grid.h>
#include <iostream>

void simulation_step(
    Eigen::MatrixXd &positions,
    float* cpu_device_buf,
    float3* device_positions,
    float3* device_positions_star,
    float3* velocity,
    float3* gravity,
    float3* sim_space_bot_left,
    float3* sim_space_top_right,
    int* result,
    int* grid_index,
    int* particle_index,
    int* cell_start,
    int* cell_end,
    float* lambdas,
    float cube_s,
    float dt,
    float h,
    float mass,
    float pho0, 
    float epsilon,
    float num_iteration
) {
    int N = positions.rows();

    dim3 grid_dim(64, 1, 1);
    dim3 thread_block(128, 1, 1);
    advect<<<grid_dim, thread_block>>>(device_positions, device_positions_star,
    velocity, gravity, sim_space_bot_left, sim_space_top_right, dt, N);
    cudaDeviceSynchronize();

    build_grid(device_positions_star, result, cube_s, sim_space_bot_left,
    sim_space_top_right, grid_index, particle_index, cell_start, cell_end, N);
    cudaDeviceSynchronize();

    compute_lambda<<<grid_dim, thread_block>>>(device_positions_star, pho0, mass, epsilon, h,
    lambdas, cell_start, cell_end, grid_index, particle_index,
    sim_space_bot_left, sim_space_top_right, cube_s, N);
    cudaDeviceSynchronize();

    cudaError_t status;
    if ((status = cudaMemcpy(device_positions, device_positions_star, N*sizeof(float)*3,cudaMemcpyDeviceToDevice))!= cudaSuccess) {
        std::cout << "ERROR memcpy: " << cudaGetErrorName(status) << std::endl;
    }
    to_cpu(device_positions, cpu_device_buf, positions);


}