#include <iostream>
#include <thread>
#include <igl/opengl/glfw/Viewer.h>

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
    viewer.launch();

    return 0;
}
