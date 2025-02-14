#include "arap_cuda.h"
#include <cuda_runtime.h>

// Declare the launcher functions
void launch_compute_covariance(const float*, const float*, const int64_t*, 
                             const bool*, const float*, float*, int, int);
void launch_compute_rotations(const float*, float*, int);
void launch_compute_energy(const float*, const float*, const float*, const int64_t*,
                         const bool*, const float*, float*, int, int);
void launch_compute_gradient(const float*, const float*, const float*, const int64_t*,
                           const bool*, const float*, const float*, float*, int, int);

// Wrapper function for forward pass
std::tuple<torch::Tensor, torch::Tensor> compute_arap_energy_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& original_vertices,
    const torch::Tensor& neighbors,
    const torch::Tensor& neighbor_mask,
    const torch::Tensor& weights)
{
    // Check if tensors are on CUDA
    if (!vertices.is_cuda()) throw std::runtime_error("vertices must be a CUDA tensor");
    if (!original_vertices.is_cuda()) throw std::runtime_error("original_vertices must be a CUDA tensor");
    if (!neighbors.is_cuda()) throw std::runtime_error("neighbors must be a CUDA tensor");
    if (!neighbor_mask.is_cuda()) throw std::runtime_error("neighbor_mask must be a CUDA tensor");
    if (!weights.is_cuda()) throw std::runtime_error("weights must be a CUDA tensor");

    // Get tensor dimensions
    auto num_vertices = vertices.size(0);
    auto max_neighbors = neighbors.size(1);
        
    // Setting options to match vertices tensor in terms of dtype and device
    auto options = torch::TensorOptions().dtype(vertices.dtype()).device(vertices.device());

    // TODO: ARAP energy and rotaions are defined per vertex
    auto energy = torch::zeros({num_vertices}, options);
    auto covariances = torch::zeros({num_vertices, 3, 3}, options);
    auto rotations = torch::zeros({num_vertices, 3, 3}, options);
    
    // Launch kernel, passing pointers to the tensors received from pytorch
    // Will return the covariances in the covariances tensor
    launch_compute_covariance(
        vertices.data_ptr<float>(),
        original_vertices.data_ptr<float>(),
        neighbors.data_ptr<int64_t>(),
        neighbor_mask.data_ptr<bool>(),
        weights.data_ptr<float>(),
        covariances.data_ptr<float>(),
        num_vertices,
        max_neighbors
    );
    
    // Then compute rotations from covariances, results in rotations tensor
    launch_compute_rotations(
        covariances.data_ptr<float>(),
        rotations.data_ptr<float>(),
        num_vertices
    );
    
    // Finally compute energy using the rotations, results in energy tensor
    launch_compute_energy(
        vertices.data_ptr<float>(),
        original_vertices.data_ptr<float>(),
        rotations.data_ptr<float>(),
        neighbors.data_ptr<int64_t>(),
        neighbor_mask.data_ptr<bool>(),
        weights.data_ptr<float>(),
        energy.data_ptr<float>(),
        num_vertices,
        max_neighbors
    );
    
    return std::make_tuple(energy.sum(), rotations);
}

// Wrapper function for backward pass
torch::Tensor compute_arap_gradient_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& original_vertices,
    const torch::Tensor& neighbors,
    const torch::Tensor& neighbor_mask,
    const torch::Tensor& rotations,
    const torch::Tensor& weights,
    const torch::Tensor& grad_output)
{
    const int num_vertices = vertices.size(0);
    const int max_neighbors = neighbors.size(1);
    
    auto grad_vertices = torch::zeros_like(vertices);
    
    launch_compute_gradient(
        vertices.data_ptr<float>(),
        original_vertices.data_ptr<float>(),
        rotations.data_ptr<float>(),
        neighbors.data_ptr<int64_t>(),
        neighbor_mask.data_ptr<bool>(),
        weights.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_vertices.data_ptr<float>(),
        num_vertices,
        max_neighbors
    );
    
    return grad_vertices;
}
