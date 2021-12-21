#include <EigenTypes.h>
#include <cuda_runtime.h>


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
    float3* delta_positions,
    float cube_s,
    float dt,
    float h,
    float mass,
    float pho0, 
    float epsilon,
    float num_iteration,
    bool simulation
);