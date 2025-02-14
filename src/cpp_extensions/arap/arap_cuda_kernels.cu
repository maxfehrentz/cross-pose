#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cusolverDn.h>

// Add error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        throw std::runtime_error("CUDA error"); \
    } \
} while(0)

// Analytical 3x3 SVD for device code
__device__ void svd3x3(const float* A_in, float* U, float* S, float* V) {
    // Copy input
    float A[9];
    for(int i = 0; i < 9; i++) {
        A[i] = A_in[i];
    }

    // Compute A^T A
    float ATA[9] = {0};
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            for(int k = 0; k < 3; k++) {
                ATA[i*3 + j] += A[k*3 + i] * A[k*3 + j];
            }
        }
    }

    // Compute eigenvalues (using characteristic equation for 3x3)
    float p1 = ATA[1]*ATA[1] + ATA[2]*ATA[2] + ATA[5]*ATA[5];
    float q = (ATA[0] + ATA[4] + ATA[8])/3.0f;
    float p2 = (ATA[0] - q)*(ATA[0] - q) + (ATA[4] - q)*(ATA[4] - q) + 
               (ATA[8] - q)*(ATA[8] - q) + 2*p1;
    float p = sqrtf(p2/6.0f);
    
    // Compute singular values
    float det = (A[0]*A[4]*A[8] + A[1]*A[5]*A[6] + A[2]*A[3]*A[7] - 
                A[2]*A[4]*A[6] - A[0]*A[5]*A[7] - A[1]*A[3]*A[8]);
    S[0] = sqrtf(fmaxf(ATA[0], 0.0f));
    S[1] = sqrtf(fmaxf(ATA[4], 0.0f));
    S[2] = det > 0 ? fabs(det)/(S[0]*S[1]) : sqrtf(fmaxf(ATA[8], 0.0f));

    // Compute singular vectors
    for(int i = 0; i < 3; i++) {
        float s = S[i];
        if(s > 1e-10f) {
            float inv_s = 1.0f/s;
            U[i*3 + 0] = A[0]*inv_s;
            U[i*3 + 1] = A[1]*inv_s;
            U[i*3 + 2] = A[2]*inv_s;
        } else {
            U[i*3 + 0] = (i == 0) ? 1.0f : 0.0f;
            U[i*3 + 1] = (i == 1) ? 1.0f : 0.0f;
            U[i*3 + 2] = (i == 2) ? 1.0f : 0.0f;
        }
    }

    // V = U for symmetric matrix
    for(int i = 0; i < 9; i++) {
        V[i] = U[i];
    }
}

// Kernel to compute covariance matrices
__global__ void compute_covariance_kernel(
    const float* vertices,          // [N, 3]
    const float* original_vertices, // [N, 3]
    const int64_t* neighbors,       // [N, K]
    const bool* mask,              // [N, K]
    const float* weights,          // [N, K]
    float* covariance_matrices,    // [N, 3, 3]
    int num_vertices,
    int max_neighbors
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print dimensions for the first thread
    if (vid == 0) {
        printf("CUDA Kernel Dimensions:\n");
        printf("num_vertices: %d\n", num_vertices);
        printf("max_neighbors: %d\n", max_neighbors);
        printf("total mask size should be: %d\n", num_vertices * max_neighbors);
    }
    
    // TODO: understand why this is problematic
    // // Print problematic accesses
    // if (vid >= num_vertices - 5 && vid < num_vertices + 5) {
    //     printf("Thread %d accessing mask at indices:\n", vid);
    //     for (int k = 0; k < max_neighbors; k++) {
    //         int mask_idx = vid * max_neighbors + k;
    //         printf("  mask[%d] (vid=%d, k=%d)\n", mask_idx, vid, k);
    //     }
    // }
    
    if (vid >= num_vertices) return;

    // Initialize covariance matrix for this vertex
    float cov[3][3] = {0};
    
    // Compute covariance matrix
    for (int k = 0; k < max_neighbors; k++) {
        if (!mask[vid * max_neighbors + k]) continue;
        
        int nid = neighbors[vid * max_neighbors + k];
        float weight = weights[vid * max_neighbors + k];
        
        // Compute edge vectors
        float orig_edge[3], curr_edge[3];
        for (int d = 0; d < 3; d++) {
            orig_edge[d] = original_vertices[nid * 3 + d] - original_vertices[vid * 3 + d];
            curr_edge[d] = vertices[nid * 3 + d] - vertices[vid * 3 + d];
        }
        
        // Accumulate weighted outer product
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                cov[i][j] += weight * orig_edge[i] * curr_edge[j];
            }
        }
    }
    
    // Store result
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            covariance_matrices[vid * 9 + i * 3 + j] = cov[i][j];
        }
    }
}

// Kernel to compute rotations from covariance matrices
__global__ void compute_rotations_kernel(
    const float* covariance_matrices, // [N, 3, 3]
    float* rotations,                // [N, 3, 3]
    int num_vertices
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= num_vertices) return;
    
    // Get pointers to this vertex's matrices
    const float* cov = &covariance_matrices[vid * 9];
    float* rot = &rotations[vid * 9];
    
    // Compute SVD
    float U[9], S[3], V[9];
    svd3x3(cov, U, S, V);
    
    // Compute R = V * U^T
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += V[i * 3 + k] * U[j * 3 + k];  // Note: U is transposed
            }
            rot[i * 3 + j] = sum;
        }
    }
    
    // Handle reflection case; check if det(R) < 0
    float det = rot[0] * (rot[4] * rot[8] - rot[5] * rot[7]) -
                rot[1] * (rot[3] * rot[8] - rot[5] * rot[6]) +
                rot[2] * (rot[3] * rot[7] - rot[4] * rot[6]);
    if (det < 0) {
        rot[0] = -rot[0];
        rot[1] = -rot[1];
        rot[2] = -rot[2];
    }
}

// Kernel to compute energy
__global__ void compute_energy_kernel(
    const float* vertices,          // [N, 3]
    const float* original_vertices, // [N, 3]
    const float* rotations,        // [N, 3, 3]
    const int64_t* neighbors,       // [N, K]
    const bool* mask,              // [N, K]
    const float* weights,          // [N, K]
    float* energy_per_vertex,      // [N]
    int num_vertices,
    int max_neighbors
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= num_vertices) return;
    
    float energy = 0.0f;
    
    // Get rotation matrix for this vertex
    const float* rot = &rotations[vid * 9];
    
    for (int k = 0; k < max_neighbors; k++) {
        if (!mask[vid * max_neighbors + k]) continue;
        
        int nid = neighbors[vid * max_neighbors + k];
        float weight = weights[vid * max_neighbors + k];
        
        // Compute edge vectors
        float orig_edge[3], curr_edge[3];
        for (int d = 0; d < 3; d++) {
            orig_edge[d] = original_vertices[nid * 3 + d] - original_vertices[vid * 3 + d];
            curr_edge[d] = vertices[nid * 3 + d] - vertices[vid * 3 + d];
        }
        
        // Rotate original edge
        float rotated_edge[3] = {0};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rotated_edge[i] += rot[i * 3 + j] * orig_edge[j];
            }
        }
        
        // Compute squared difference
        float diff = 0.0f;
        for (int d = 0; d < 3; d++) {
            float d_i = curr_edge[d] - rotated_edge[d];
            diff += d_i * d_i;
        }
        
        energy += weight * diff;
    }
    
    energy_per_vertex[vid] = energy;
}

// Add gradient computation kernel
__global__ void compute_arap_gradient_kernel(
    const float* vertices,          // [N, 3]
    const float* original_vertices, // [N, 3]
    const float* rotations,        // [N, 3, 3]
    const int64_t* neighbors,      // [N, K]
    const bool* mask,              // [N, K]
    const float* weights,          // [N, K]
    const float* grad_output,      // [1] or [N]
    float* grad_vertices,          // [N, 3]
    int num_vertices,
    int max_neighbors
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= num_vertices) return;
    
    float grad[3] = {0.0f, 0.0f, 0.0f};
    const float* rot = &rotations[vid * 9];
    float grad_scale = grad_output[0];  // or grad_output[vid] if per-vertex
    
    for (int k = 0; k < max_neighbors; k++) {
        if (!mask[vid * max_neighbors + k]) continue;
        
        int nid = neighbors[vid * max_neighbors + k];
        float weight = weights[vid * max_neighbors + k];
        
        // Compute edge vectors
        float orig_edge[3], curr_edge[3];
        for (int d = 0; d < 3; d++) {
            orig_edge[d] = original_vertices[nid * 3 + d] - original_vertices[vid * 3 + d];
            curr_edge[d] = vertices[nid * 3 + d] - vertices[vid * 3 + d];
        }
        
        // Compute rotated edge
        float rotated_edge[3] = {0};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rotated_edge[i] += rot[i * 3 + j] * orig_edge[j];
            }
        }
        
        // Compute gradient contribution
        for (int d = 0; d < 3; d++) {
            float diff = curr_edge[d] - rotated_edge[d];
            grad[d] += 2.0f * weight * diff * grad_scale;
        }
        
        // Add contribution to neighbor's gradient (atomic as neighbor might be shared)
        for (int d = 0; d < 3; d++) {
            atomicAdd(&grad_vertices[nid * 3 + d], -grad[d]);
        }
    }
    
    // Add accumulated gradient to vertex
    for (int d = 0; d < 3; d++) {
        grad_vertices[vid * 3 + d] += grad[d];
    }
}

// Launcher functions
void launch_compute_covariance(const float* vertices, const float* original_vertices,
                             const int64_t* neighbors, const bool* mask, const float* weights,
                             float* covariance_matrices, int num_vertices, int max_neighbors) {
    const int threads = 256;
    // A block is a group of threads that can be executed in parallel; compute the number of blocks needed
    const int blocks = (num_vertices + threads - 1) / threads;
    compute_covariance_kernel<<<blocks, threads>>>(vertices, original_vertices, neighbors,
                                                 mask, weights, covariance_matrices,
                                                 num_vertices, max_neighbors);
    CUDA_CHECK(cudaGetLastError());
}

void launch_compute_rotations(const float* covariance_matrices, float* rotations,
                            int num_vertices) {
    const int threads = 256;
    const int blocks = (num_vertices + threads - 1) / threads;
    compute_rotations_kernel<<<blocks, threads>>>(covariance_matrices, rotations, num_vertices);
    CUDA_CHECK(cudaGetLastError());
}

void launch_compute_energy(const float* vertices, const float* original_vertices,
                         const float* rotations, const int64_t* neighbors,
                         const bool* mask, const float* weights, float* energy,
                         int num_vertices, int max_neighbors) {
    const int threads = 256;
    const int blocks = (num_vertices + threads - 1) / threads;
    compute_energy_kernel<<<blocks, threads>>>(vertices, original_vertices, rotations,
                                             neighbors, mask, weights, energy,
                                             num_vertices, max_neighbors);
    CUDA_CHECK(cudaGetLastError());
}

void launch_compute_gradient(const float* vertices, const float* original_vertices,
                           const float* rotations, const int64_t* neighbors,
                           const bool* mask, const float* weights,
                           const float* grad_output, float* grad_vertices,
                           int num_vertices, int max_neighbors) {
    const int threads = 256;
    const int blocks = (num_vertices + threads - 1) / threads;
    compute_arap_gradient_kernel<<<blocks, threads>>>(vertices, original_vertices,
                                                    rotations, neighbors, mask, weights,
                                                    grad_output, grad_vertices,
                                                    num_vertices, max_neighbors);
    CUDA_CHECK(cudaGetLastError());
}
