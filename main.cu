#include <iostream>
#include <thread>
#include <igl/opengl/glfw/Viewer.h>
#include <cuda_runtime.h>

#include <init_particles.cuh>
#include <mem_util.cuh>
#include <simulation_step.cuh>


Eigen::MatrixXd positions;
float3* velocity;
float* cpu_device_buf;
float3* positions_device;
float3* positions_star_device;

int* result;
int* grid_index;
int* particle_index;
int* cell_start;
int* cell_end;
float* lambdas;

double particle_init_step = 0.1;
igl::opengl::glfw::Viewer viewer;

float3* sim_space_bot_left;
float3* sim_space_top_right;


//constants
float3* gravity_m;

//simulation time and time step
float t = 0; //simulation time 
float dt = 0.01; //time step
float cube_s = 0.4;
float h = cube_s;
float mass = 1.0;
float pho0 = 10000.0;
float epsilon = 1000;
float num_iteration = 3;

//simulation loop
bool simulating = true;

bool simulation_callback() {

    // while (simulating) {
    simulation_step(positions, cpu_device_buf, positions_device, positions_star_device,
    velocity, gravity_m, sim_space_bot_left, sim_space_top_right, result,
    grid_index, particle_index, cell_start, cell_end, lambdas,
    cube_s, dt, h, mass, pho0, epsilon, num_iteration);

    // const Eigen::RowVector3d particle_color(0.333, 0.647, 0.905);
    // viewer.data().set_points(positions, particle_color);
    std::cout << "Complete a step" << std::endl;
    // // }
    return true;
}

int main(int argc, char **argv) {

    std::cout<<"Start Project\n";

    //setup libigl viewer and activate 

    viewer.core().background_color.setConstant(1.0);

    cudaError_t status;
    if ((status = cudaMalloc(&sim_space_bot_left, sizeof(float3))) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    if ((status = cudaMalloc(&sim_space_top_right, sizeof(float3))) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    if ((status = cudaMalloc(&gravity_m, sizeof(float3))) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    cpu_device_buf = (float*)malloc(sizeof(float)*3);
    Eigen::MatrixXd tmp(1, 3);
    tmp << 0.0, 0.0, 0.0;
    to_gpu(tmp, cpu_device_buf, sim_space_bot_left);
    tmp << 8.0, 8.0, 8.0;
    to_gpu(tmp, cpu_device_buf, sim_space_top_right);
    tmp << 0.0, -9.8, 0.0;
    to_gpu(tmp, cpu_device_buf, gravity_m);
    free(cpu_device_buf);

    Eigen::Vector3d particle_init_bot_left;
    particle_init_bot_left << 0.1, 0.1, 0.1;


    const Eigen::RowVector3d particle_color(0.333, 0.647, 0.905);
    init_particles(positions, particle_init_bot_left, particle_init_step, 
    20, 20, 20);
    cpu_device_buf = (float*)malloc(sizeof(float)*3*positions.rows());
    
    if ((status = cudaMalloc(&positions_device, sizeof(float3)*positions.rows())) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    if ((status = cudaMalloc(&positions_star_device, sizeof(float3)*positions.rows())) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    if ((status = cudaMalloc(&velocity, sizeof(float3)*positions.rows())) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    if ((status = cudaMalloc(&result, sizeof(int)*positions.rows())) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    if ((status = cudaMalloc(&grid_index, sizeof(int)*positions.rows())) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    if ((status = cudaMalloc(&particle_index, sizeof(int)*positions.rows())) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    if ((status = cudaMalloc(&cell_start, sizeof(int)*positions.rows())) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    if ((status = cudaMalloc(&cell_end, sizeof(int)*positions.rows())) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }
    if ((status = cudaMalloc(&lambdas, sizeof(float)*positions.rows())) != cudaSuccess) {
        std::cout << "ERROR cudaMalloc" << cudaGetErrorName(status) << std::endl;
    }

    to_gpu(positions, cpu_device_buf, positions_device);
    // positions.setZero();
    // to_cpu(positions_device, cpu_device_buf, positions);

    viewer.data().set_points(positions, particle_color);
    viewer.data().point_size = 5.0;

    // Eigen::Vector3d g_v;
    // g_v << 0.0, -9.8, 0.0;

    // std::thread simulation_thread(simulation_callback);
    // simulation_thread.detach();

    viewer.callback_key_pressed =
			[&](igl::opengl::glfw::Viewer&, unsigned char key, int)->bool
		{
			switch (key)
			{
			case 'A':
			case 'a':
				//with ghost pressure
				simulation_callback();
                viewer.data().set_points(positions, particle_color);
                // std::cout << positions << std::endl;
				break;
			default:
				return false;
			}
			return true;
		};

    viewer.launch();

    return 0;
}
