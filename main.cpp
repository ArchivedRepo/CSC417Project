#include <iostream>
#include <thread>
#include <igl/opengl/glfw/Viewer.h>

#include <init_particles.h>

Eigen::MatrixXd positions;
double particle_init_step = 0.2;

//simulation time and time step
double t = 0; //simulation time 
double dt = 0.00001; //time step

//simulation loop
bool simulating = true;

bool simulation_callback() {

    while(simulating) {

    }
    return false;
}

int main(int argc, char **argv) {

    std::cout<<"Start Project\n";

    //setup libigl viewer and activate 
    igl::opengl::glfw::Viewer viewer;
    viewer.core().background_color.setConstant(1.0);


    Eigen::Vector3d particle_init_bot_left;
    particle_init_bot_left << 0.1, 0.1, 0.1;
    const Eigen::RowVector3d particle_color(0.333, 0.647, 0.905);
    init_particles(positions, particle_init_bot_left, particle_init_step, 
    20, 30, 10);
    viewer.data().set_points(positions, particle_color);
    viewer.data().point_size = 10.0;

    viewer.launch();

    return 0;
}
