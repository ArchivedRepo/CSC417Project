#include <kernels.h>

float poly6(Eigen::Vector3d &R, double h) {
    float r = R.norm();
    float h2 = h * h;
    float r2 = r * r;
    float nominator = (h2 - r2) * (h2 - r2) *(h2 - r2);
    float h4 = h * h * h * h;
    float h9 = h4 * h4 * h;
    if (r >= 0 && r < h) {
        return (315.0 * nominator) / (64.0 * M_PI * h9);
    } else {
        return 0.0;
    }
}

void splky_grad(Eigen::Vector3d &R, double h, Eigen::Vector3d &grad) {
    double h3 = h * h * h;
    double h6 = h3 * h3;
    double r = R.norm();
    if (r >= 0 && r <= h) {
        double multiplier = - (45.0 / (M_PI * h6));
        grad = multiplier * (h - r) * (h-r) * R.normalized();
    }
    grad.setZero();
}