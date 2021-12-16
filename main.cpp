#include <iostream>
#include <thread>
#include <igl/opengl/glfw/Viewer.h>

#include <init_particles.h>
#include <gravity_matrix.h>
#include <simulation_step.h>

Eigen::MatrixXd positions;
Eigen::MatrixXd velocity;
double particle_init_step = 0.1;
igl::opengl::glfw::Viewer viewer;

//constants
Eigen::MatrixXd gravity_m;

//simulation time and time step
double t = 0; //simulation time 
double dt = 0.01; //time step
double h = 0.2;
double mass = 1.0;
double pho0 = 8000.0;
double epsilon = 1000;
double num_iteration = 3;

//simulation loop
bool simulating = true;

bool simulation_callback() {

    // while (simulating) {
    simulation_step(positions, velocity, gravity_m, dt, h, mass, pho0,epsilon, num_iteration);
    const Eigen::RowVector3d particle_color(0.333, 0.647, 0.905);
    viewer.data().set_points(positions, particle_color);
    std::cout << "Complete a step" << std::endl;
    // }
    return true;
}

int main(int argc, char **argv) {

    std::cout<<"Start Project\n";

    //setup libigl viewer and activate 
    
    viewer.core().background_color.setConstant(1.0);


    Eigen::Vector3d particle_init_bot_left;
    particle_init_bot_left << 0.1, 0.1, 0.1;
    const Eigen::RowVector3d particle_color(0.333, 0.647, 0.905);
    init_particles(positions, particle_init_bot_left, particle_init_step, 
    10, 10, 10);
    velocity.resize(positions.rows(), 3);
    velocity.setZero();
    viewer.data().set_points(positions, particle_color);
    viewer.data().point_size = 10.0;

    Eigen::Vector3d g_v;
    g_v << 0.0, -9.8, 0.0;
    gravity_matrix(gravity_m, g_v, positions.rows());

    std::thread simulation_thread(simulation_callback);
    simulation_thread.detach();

    viewer.callback_key_pressed =
			[&](igl::opengl::glfw::Viewer&, unsigned char key, int)->bool
		{
			switch (key)
			{
			case 'A':
			case 'a':
				//with ghost pressure
				simulation_callback();
				break;
			default:
				return false;
			}
			return true;
		};

    viewer.launch();

    return 0;
}
