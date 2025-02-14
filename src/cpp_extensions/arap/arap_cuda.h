#pragma once
#include <torch/extension.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

std::tuple<torch::Tensor, torch::Tensor> compute_arap_energy_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& original_vertices,
    const torch::Tensor& neighbors,
    const torch::Tensor& neighbor_mask,
    const torch::Tensor& weights);

torch::Tensor compute_arap_gradient_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& original_vertices,
    const torch::Tensor& neighbors,
    const torch::Tensor& neighbor_mask,
    const torch::Tensor& rotations,
    const torch::Tensor& weights,
    const torch::Tensor& grad_output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_arap_energy_cuda", &compute_arap_energy_cuda, "ARAP Energy (CUDA)");
    m.def("compute_arap_gradient_cuda", &compute_arap_gradient_cuda, "ARAP Gradient (CUDA)");
}