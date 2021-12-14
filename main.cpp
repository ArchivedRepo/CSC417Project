#include <iostream>
#include <thread>

#include <assignment_setup.h>
#include <visualization.h>

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

    std::cout<<"Start A6\n";

    //setup libigl viewer and activate 
    Visualize::setup(true);
    Visualize::viewer().launch();

    return 1; 

}
