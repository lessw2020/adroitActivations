// cuda_tensor_loader.cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

torch::Tensor loadPthTensorToCuda(const std::string& filename, int deviceId) {
    // Set the CUDA device
    cudaSetDevice(deviceId);

    // Load the tensor from the .pth file
    torch::Tensor tensor;
    try {
        // Attempt to load as a single tensor
        tensor = torch::load(filename);
    } catch (const c10::Error& e) {
        // If loading as a single tensor fails, try loading as a state dict
        auto state_dict = torch::load(filename);
        if (state_dict.is_dict()) {
            // Assume the first item in the dict is our tensor
            // You might need to adjust this if you know the specific key
            tensor = state_dict.begin()->value().toTensor();
        } else {
            throw std::runtime_error("File does not contain a tensor or a state dict");
        }
    }

    // Move the tensor to the specified CUDA device
    tensor = tensor.to(torch::Device(torch::kCUDA, deviceId));

    return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_pth_tensor_to_cuda", &loadPthTensorToCuda, "Load a .pth tensor file to CUDA",
          py::arg("filename"), py::arg("device_id") = 0);
}
