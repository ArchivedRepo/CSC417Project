#include <mem_util.cuh>
#include <iostream>

void to_gpu(
    Eigen::MatrixXd &src,
    float* buf,
    float3* dest
) {
    for (int i =0; i< src.rows(); i++) {
        buf[i*3+0] = src(i, 0);
        buf[i*3+1] = src(i, 1);
        buf[i*3+2] = src(i, 2);
    }
    cudaError_t status;
    if ((status = cudaMemcpy(dest, buf, src.rows()*3*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        std::cout << "to_gpu: " << cudaGetErrorName(status) <<std::endl;
    }
}

void to_cpu(
    float3* src,
    float* buf,
    Eigen::MatrixXd &dest
) {
    cudaError_t status;
    if ((status = cudaMemcpy(buf, src, dest.rows()*3*sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        std::cout << "to_cpu: " << cudaGetErrorName(status) <<std::endl;
    }
    int pos = 0;
    for (int i =0; i < dest.rows(); i++) {
        dest.row(i) << buf[pos], buf[pos+1], buf[pos+2];
        pos += 3;
    }
}