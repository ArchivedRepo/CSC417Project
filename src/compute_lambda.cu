#include <compute_lambda.cuh>
#include <build_grid.cuh>

static __device__ float poly6(float r, float h) {
    if (r >= 0 && r <= h) {
        return (315.0 * powf(h*h-r*r, 3.0)) / (64.0 * M_PI * powf(h, 9.0));
    }
    return 0.0;
}

static __device__ float3 spiky_grad(float3 r, float h) {
    float norm = norm3df(r.x, r.y, r.z);
    float3 normalized = make_float3(r.x / norm, r.y / norm, r.z / norm);
    if (norm > 0 && norm <= h) {
        float coeff = -(45.0 * powf(norm - h, 2.0)) / (M_PI * powf(h, 6.0));
        return make_float3(normalized.x*coeff, normalized.y*coeff, normalized.z*coeff); 
    } else {
        return make_float3(0.0, 0.0, 0.0); 
    }
}

static __device__ int compute_index(
    float3 position,
    float3 *bot_left,
    int num_L, int num_W, int num_H, float cube_s
) {
    float3 relative_pos = make_float3(
        position.x-bot_left->x,
        position.y-bot_left->y,
        position.z-bot_left->z
    );
    
    int3 int_pos;
    int_pos.x = (int)(relative_pos.x / cube_s);
    int_pos.y = (int)(relative_pos.y / cube_s);
    int_pos.z = (int)(relative_pos.z / cube_s);
    int h = int_pos.z * (num_W * num_L);
    int w = int_pos.y * num_L;
    return h + int_pos.x + w;
}

static __device__ float3 index_to_xyh(int index, int L, int W, int H) {
    int h = index / (L * W);
    int plane_coor = index % (L * W);
    int x = plane_coor % L;
    int y = plane_coor / L;
    return make_float3(x, y, h);
}

static __device__ int xyh_to_index(int x, int y, int h, int L, int W, int H) {
    return h * (W * L) + y * L + x;
}

__global__ void compute_lambda(
    float3* positions,
    float pho0,
    float mass,
    float epsilon,
    float h,
    float* lambdas,
    int* cell_start,
    int* cell_end,
    int* grid_index,
    int* particle_index,
    float3* bot_left,
    float3* up_right,
    float cube_s,
    int N
) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    int num_L = ceilf((up_right->x - bot_left->x) / cube_s);
    int num_W = ceilf((up_right->y - bot_left->y) / cube_s);
    int num_H = ceilf((up_right->z - bot_left->z) / cube_s);

    float3 p_i = positions[i];

    int this_index = compute_index(p_i, bot_left, num_L, num_W, num_H, cube_s);

    int x_index, y_index, h_index;

    float3 grid_vec = index_to_xyh(this_index, num_L, num_W, num_H);
    x_index = grid_vec.x;
    y_index = grid_vec.y;
    h_index = grid_vec.z;

    float pho = 0.0;

    float3 grad_i = make_float3(0.0, 0.0, 0.0);

    float grad_sum = 0.0;

    for (int x_offset = -1 ; x_offset < 2; x_offset++ ) {
        for (int y_offset = -1; y_offset < 2; y_offset++) {
            for (int h_offset = -1; h_offset < 2; h_offset++) {

                int cur_x, cur_y, cur_h, cur_index;

                cur_x = x_index + x_offset;
                cur_y = y_index + y_offset;
                cur_h = h_index + h_offset;

                if (cur_x < 0 || cur_y < 0 || cur_h < 0 || cur_x >= num_L || cur_y >= num_W || cur_h >= num_H) {
                    continue;
                }

                cur_index = xyh_to_index(cur_x, cur_y, cur_h, num_L, num_W, num_H);

                int start, end;
                start = cell_start[cur_index];
                end = cell_end[cur_index];

                for (int j = start; j < end; j++) {
                    int target_id = particle_index[j];

                    float3 p_j = positions[target_id];
                    float3 diff = make_float3(p_i.x-p_j.x, p_i.y-p_j.y, p_i.z-p_j.z);

                    float norm = norm3df(diff.x, diff.y, diff.z);
                    pho += mass * poly6(norm, h);

                    float3 local_grad = spiky_grad(diff, h);
                    
                    grad_i.x = grad_i.x + local_grad.x;
                    grad_i.y = grad_i.y + local_grad.y;
                    grad_i.z = grad_i.z + local_grad.z;

                    float squared_norm = norm3df(local_grad.x, local_grad.y, local_grad.z);
                    squared_norm = squared_norm * squared_norm;

                    if (target_id != i) {
                        grad_sum += (1.0/pho0)*(1.0/pho0) * squared_norm;
                    }
                }
            }
        }
    }
    float grad_i_squared_norm = norm3df(grad_i.x, grad_i.y, grad_i.z);
    grad_i_squared_norm = grad_i_squared_norm * grad_i_squared_norm;
    float C_i = (pho/pho0) - 1.0;
    float denominator = grad_sum + (1.0/pho0)*(1.0/pho0)*grad_i_squared_norm;
    lambdas[i] = -C_i / (denominator+epsilon);
}