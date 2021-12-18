#include <simulation_step.cuh>
#include <advect.cuh>
#include <mem_util.cuh>
// #include <compute_lambda.h>
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

    cudaError_t status;
    if ((status = cudaMemcpy(device_positions, device_positions_star, N*sizeof(float)*3,cudaMemcpyDeviceToDevice))!= cudaSuccess) {
        std::cout << "ERROR: " << cudaGetErrorName(status) << std::endl;
    }
    to_cpu(device_positions, cpu_device_buf, positions);


}