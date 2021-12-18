#include <cuda_runtime.h>
#include <EigenTypes.h>

void to_gpu(
    Eigen::MatrixXd &src,
    float* buf,
    float3* dest
);

void to_cpu(
    float3* src,
    float* buf,
    Eigen::MatrixXd &dest
);