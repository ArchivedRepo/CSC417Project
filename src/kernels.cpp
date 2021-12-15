#include <kernels.h>

float poly6(Eigen::Vector3d &R, double h) {
    double h2 = h * h;
    double h4 = h2 * h2;
    double h8 = h4 * h4;
    double h9 = h8 * h;

    double coeff = 315.0/ (64.0 * M_PI * h9);

    double norm2 = R.squaredNorm();

    if (norm2 <= h2){
        return coeff * pow(h2 - norm2, 3);
    }

    return 0.0;
}

void spiky_grad(Eigen::Vector3d &R, double h, Eigen::Vector3d &grad) {

    double coef;
    double h6 = h * h;
	h6 = h6 * h6 * h6;
	coef = -45.f / (M_PI * h6);
    double norm = R.norm();

    if (norm > 1e-3 && norm < h){
        grad = coef * (h - norm) * (h - norm) * R.normalized();
    }else{
        grad.setZero();
    }
}