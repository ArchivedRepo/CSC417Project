#ifndef  VISUALIZATION_H
#define  VISUALIZATION_H

#define IMGUI_DEFINE_MATH_OPERATORS

#include <igl/unproject.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

//stl
#include <vector>
#include <array>
#include <deque>

//Eigen
#include <Eigen/Dense>

namespace Visualize {

    void setup(const Eigen::VectorXd &q, const Eigen::VectorXd &qdot, bool ps_plot = false);
    igl::opengl::glfw::Viewer & viewer();
    igl::opengl::glfw::imgui::ImGuiMenu & viewer_menu();

}


#endif