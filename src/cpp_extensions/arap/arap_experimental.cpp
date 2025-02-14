#include "arap.h"
#include <torch/extension.h>
#include <omp.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>

std::tuple<torch::Tensor, torch::Tensor> compute_arap_energy(
    const torch::Tensor& vertices,
    const torch::Tensor& original_vertices,
    const torch::Tensor& neighbors,
    const torch::Tensor& neighbor_mask,
    const torch::Tensor& weights) {
    
    TORCH_CHECK(vertices.device().is_cpu(), "vertices must be a CPU tensor");
    TORCH_CHECK(vertices.is_contiguous(), "vertices must be contiguous");
    
    const int num_vertices = vertices.size(0);
    const int max_neighbors = neighbors.size(1);
    
    // Create tensors for storing edges
    auto curr_edges = torch::zeros({num_vertices, max_neighbors, 3}, vertices.options());
    auto orig_edges = torch::zeros({num_vertices, max_neighbors, 3}, vertices.options());
    
    // Get all accessors
    auto vertices_accessor = vertices.accessor<float,2>();
    auto orig_vertices_accessor = original_vertices.accessor<float,2>();
    auto neighbors_accessor = neighbors.accessor<int64_t,2>();
    auto mask_accessor = neighbor_mask.accessor<bool,2>();
    auto curr_edges_accessor = curr_edges.accessor<float,3>();
    auto orig_edges_accessor = orig_edges.accessor<float,3>();
    auto weights_accessor = weights.accessor<float,2>();
    
    // Prepare storage for rotation matrices [N, 3, 3]
    auto rotations = torch::zeros({num_vertices, 3, 3}, vertices.options());
    auto rotations_accessor = rotations.accessor<float,3>();
    
    float total_energy = 0.0f;
    
    {
        // Thread-local matrices for Eigen computations
        Eigen::Matrix3f cov, U, V, R;
        Eigen::Vector3f curr_edge, orig_edge, rotated_edge;

        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < num_vertices; i++) {
            // Reset covariance matrix
            cov.setZero();
            
            // First pass: compute covariance matrix
            for(int j = 0; j < max_neighbors; j++) {
                if(!mask_accessor[i][j]) continue;
                
                int neighbor_idx = neighbors_accessor[i][j];
                
                // Get edges and store them for reuse
                for(int k = 0; k < 3; k++) {
                    curr_edge(k) = vertices_accessor[i][k] - vertices_accessor[neighbor_idx][k];
                    orig_edge(k) = orig_vertices_accessor[i][k] - orig_vertices_accessor[neighbor_idx][k];
                    curr_edges_accessor[i][j][k] = curr_edge(k);
                    orig_edges_accessor[i][j][k] = orig_edge(k);
                }
                
                // Accumulate covariance matrix
                cov += weights_accessor[i][neighbor_idx] * orig_edge * curr_edge.transpose();
            }
            
            // Compute optimal rotation using SVD
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
            U = svd.matrixU();
            V = svd.matrixV();
            
            // Handle reflection case
            float det = (V * U.transpose()).determinant();
            if(det < 0) {
                V.col(2) = -V.col(2);
            }

            // Compute and store rotation matri
            R = V * U.transpose();
            for(int m = 0; m < 3; m++) {
                for(int n = 0; n < 3; n++) {
                    rotations_accessor[i][m][n] = R(m,n);
                }
            }
            
            // Second pass: reuse stored edges
            #pragma omp parallel for reduction(+:total_energy)
            for(int j = 0; j < max_neighbors; j++) {
                if(!mask_accessor[i][j]) continue;

                int neighbor_idx = neighbors_accessor[i][j];
                
                // Reuse stored edges
                for(int k = 0; k < 3; k++) {
                    curr_edge(k) = curr_edges_accessor[i][j][k];
                    orig_edge(k) = orig_edges_accessor[i][j][k];
                }
                
                // Rotate original edge
                rotated_edge = R * orig_edge;
                
                // Compute difference and energy
                Eigen::Vector3f diff = curr_edge - rotated_edge;
                total_energy += weights_accessor[i][neighbor_idx] * diff.squaredNorm();
            }
        }
    }
    
    return std::make_tuple(
        torch::tensor(total_energy),
        rotations
    );
}

torch::Tensor compute_arap_gradient(
    const torch::Tensor& vertices,
    const torch::Tensor& original_vertices,
    const torch::Tensor& neighbors,
    const torch::Tensor& neighbor_mask,
    const torch::Tensor& rotations,
    const torch::Tensor& weights,
    const torch::Tensor& grad_output) {
    
    auto grad_vertices = torch::zeros_like(vertices);
    
    auto vertices_accessor = vertices.accessor<float,2>();
    auto orig_vertices_accessor = original_vertices.accessor<float,2>();
    auto neighbors_accessor = neighbors.accessor<int64_t,2>();
    auto mask_accessor = neighbor_mask.accessor<bool,2>();
    auto rotations_accessor = rotations.accessor<float,3>();
    auto grad_accessor = grad_vertices.accessor<float,2>();
    auto weights_accessor = weights.accessor<float,2>();
    const int num_vertices = vertices.size(0);
    const int max_neighbors = neighbors.size(1);
    // Apparently incoming gradient from next layer/loss
    const float grad_scale = grad_output.item<float>();
    
    #pragma omp parallel
    {
        Eigen::Matrix3f R;
        Eigen::Vector3f curr_edge, orig_edge, rotated_edge, edge_grad;
        // Create thread-local gradient array
        float local_grad[3] = {0.0f, 0.0f, 0.0f};
        
        #pragma omp for
        for(int i = 0; i < num_vertices; i++) {
            // Get rotation matrix for this vertex
            for(int m = 0; m < 3; m++) {
                for(int n = 0; n < 3; n++) {
                    R(m,n) = rotations_accessor[i][m][n];
                }
            }
            
            Eigen::Matrix3f R_neighbor;

            for(int j = 0; j < max_neighbors; j++) {
                if(!mask_accessor[i][j]) continue;
                
                int neighbor_idx = neighbors_accessor[i][j];

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {
                        R_neighbor(m,n) = rotations_accessor[neighbor_idx][m][n];
                    }
                }
                
                // Get edges
                for(int k = 0; k < 3; k++) {
                    curr_edge(k) = vertices_accessor[i][k] - vertices_accessor[neighbor_idx][k];
                    orig_edge(k) = orig_vertices_accessor[i][k] - orig_vertices_accessor[neighbor_idx][k];
                }
                
                // Rotate original edge
                rotated_edge = 0.5f * (R + R_neighbor) * orig_edge;
                
                // Compute gradient
                edge_grad = 4.0f * weights_accessor[i][neighbor_idx] * (curr_edge - rotated_edge) * grad_scale;
                
                // Accumulate to local gradient array instead of directly to grad_accessor
                for(int k = 0; k < 3; k++) {
                    local_grad[k] += edge_grad(k);
                }
            }
            
            // Single critical section after the loop to update global gradient
            #pragma omp critical
            {
                for(int k = 0; k < 3; k++) {
                    grad_accessor[i][k] += local_grad[k];
                }
            }
        }
    }
    
    return grad_vertices;
}