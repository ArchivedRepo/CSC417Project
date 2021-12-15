#include <iostream>
#include <thread>

#include <assignment_setup.h>
#include <visualization.h>
#include <init_particles.h>
#include <simulation_step.h>

Eigen::MatrixXd positions;
Eigen::MatrixXd velocity;
Eigen::Vector3d gravity(0.0, 0.0, -9.8);
Eigen::Vector3d bot_left;
Eigen::Vector3d up_right;
double cube_s = 1;
int num_iterations = 4;
double pho0 = 8000.0;
double epsilon = 1000.0;
double mass = 1.0;
double h_kernel = cube_s;
double k = 0.001;
double delta_q = 0.7*h_kernel;
double n_coor = 4;

//simulation time and time step
double t = 0; //simulation time 
double dt = 0.1; //time step

//simulation loop
bool simulating = true;

static void print_position(Eigen::MatrixXd positions) {
    std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
    for (int i = 0; i < positions.rows(); i++) {
        std::cout << positions.row(i) << std::endl;
    }
}

bool simulation_callback() {
    simulation_step(positions, velocity, gravity, bot_left,
    up_right, cube_s, num_iterations, dt, pho0, epsilon, 
    mass, h_kernel, k, delta_q, n_coor);
    return true;
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
    
    //run simulation in seperate thread to avoid slowing down the UI
    // std::thread simulation_thread(simulation_callback);
    // simulation_thread.detach();

    bot_left.setZero();
    up_right << 2.0, 2.0, 2.0;
    
    init_particles(positions, bot_left, up_right, 0.5);
    velocity.resize(positions.rows(), 3);
    velocity.setZero();

    const Eigen::RowVector3d particle_color(0.0, 0.6, 1.0);

    Visualize::viewer().data().set_points(positions, particle_color);
    Visualize::viewer().data().point_size = 20.0;

    Visualize::viewer().callback_key_pressed =
			[&](igl::opengl::glfw::Viewer&, unsigned char key, int)->bool
		{
			switch (key)
			{
			case 'A':
			case 'a':
				//with ghost pressure
				simulation_callback();
                // print_position(positions);
                Visualize::viewer().data().set_points(positions, particle_color);
				break;
			default:
				return false;
			}
			return true;
		};

    //setup libigl viewer and activate 
    Visualize::setup(true);
    Visualize::viewer().core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    Visualize::viewer().callback_post_draw = &draw_callback;
    Visualize::viewer().launch();

    return 1; 

}
