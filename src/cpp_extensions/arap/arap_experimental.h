#pragma once
#include <torch/extension.h>

// Return (energy, rotations) for caching
std::tuple<torch::Tensor, torch::Tensor> compute_arap_energy(
    const torch::Tensor& vertices,          // [#vertices, 3]
    const torch::Tensor& original_vertices, // [#vertices, 3]
    const torch::Tensor& neighbors,         // [#vertices, max_neighbors]
    const torch::Tensor& neighbor_mask,    // [#vertices, max_neighbors]
    const torch::Tensor& weights);          // [#vertices, #vertices]

torch::Tensor compute_arap_gradient(
    const torch::Tensor& vertices,
    const torch::Tensor& original_vertices,
    const torch::Tensor& neighbors,
    const torch::Tensor& neighbor_mask,
    const torch::Tensor& rotations,         // [#vertices, 3, 3]
    const torch::Tensor& weights,          // [#vertices, #vertices]
    const torch::Tensor& grad_output);

// TORCH_EXTENSION_NAME is defined in setup.py, m is th module we are building
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Add python functions to the module
    m.def("compute_arap_energy", &compute_arap_energy, "ARAP Energy (C++)");
    m.def("compute_arap_gradient", &compute_arap_gradient, "ARAP Gradient (C++)");
}