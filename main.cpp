#include <iostream>
#include <thread>

#include <assignment_setup.h>
#include <visualization.h>
#include <init_particles.h>

//Simulation State
Eigen::VectorXd q;
Eigen::VectorXd qdot;

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

bool draw_callback(igl::opengl::glfw::Viewer &viewer) {
    
    return false;
}

void f(int &a, int b, int c) {
    a = b + c;


}

void g(Eigen::Vector3d &a, Eigen::Vector3d b, Eigen::Vector3d c) {
    a = b + c;
}

template<typename Ret, typename B, typename C>
void h(Ret &&a, B b, C c, void (*func)(Ret, B, C)) {
    func(a,b,c);
}

int main(int argc, char **argv) {

    std::cout<<"Start A6\n";

    //assignment specific setup
    assignment_setup(argc, argv, q, qdot);

    //run simulation in seperate thread to avoid slowing down the UI
    std::thread simulation_thread(simulation_callback);
    simulation_thread.detach();

    Eigen::MatrixXd positions;
    Eigen::Vector3d bot_left;
    Eigen::Vector3d up_right;

    bot_left.setZero();
    up_right << 2.0, 2.0, 2.0;
    init_particles(positions, bot_left, up_right, 0.05);

    const Eigen::RowVector3d particle_color(0.0, 0.6, 1.0);

    Visualize::viewer().data().set_points(positions, particle_color);
    Visualize::viewer().data().point_size = 5.0;

    //setup libigl viewer and activate 
    Visualize::setup(q, qdot, true);
    Visualize::viewer().core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    Visualize::viewer().callback_post_draw = &draw_callback;
    Visualize::viewer().launch();

    return 1; 

}
