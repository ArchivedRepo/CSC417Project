#include <kernels.h>
#include <math.h>

double poly6(double r, double h) {
    if (r >= 0 && r <= h) {
        return (315.0 * pow(h*h-r*r, 3.0)) / (64.0 * M_PI * pow(h, 9.0));
    }
    return 0.0;
}

void spiky_grad(Eigen::Vector3d &r, double h, Eigen::Vector3d &grad) {
    double norm = r.norm();
    if (norm > 0 && norm <= h) {
        grad = r.normalized() * (-(45.0 * pow(norm - h, 2.0)) / (M_PI * pow(h, 6.0))); 
    } else {
        grad.setZero();
    }
}