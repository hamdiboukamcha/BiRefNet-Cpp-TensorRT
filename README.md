# BiRefNet C++ TENSORRT
A high-performance C++ implementation of the Bilateral Reference Network (**BiRefNet**) leveraging **TensorRT** and **CUDA**, optimized for real-time, high-resolution dichotomous image segmentation.

<img src="asset/BiRefNet-Cpp-TensorRT.JPG" alt="BiRefNet Banner" width="800"/>

<a href="https://github.com/hamdiboukamcha/BiRefNet-Cpp-TensorRT" style="margin: 0 2px;">
    <img src="https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub" alt="GitHub">
</a>
<a href="https://github.com/hamdiboukamcha/BiRefNet-Cpp-TensorRT?tab=License" style="margin: 0 2px;">
    <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat&logo=license" alt="License">
</a>

---

## 🌐 Overview

**BiRefNet C++ TENSORRT** is designed to efficiently run bilateral reference segmentation tasks on the GPU. By harnessing TensorRT’s optimizations and CUDA kernels, it aims to deliver state-of-the-art performance with minimal latency.

### Key Features

- **TensorRT Acceleration**: Speed up inference for segmentation tasks using serialized TRT engines.  
- **CUDA Integration**: Comprehensive GPU-based preprocessing, postprocessing, and memory handling.  
- **High-Resolution Support**: Out-of-the-box ability to process high-resolution images (e.g., 1024x1024).  
- **Easy Integration**: C++17 codebase for easy deployment into existing pipelines.  

---

## 📢 What's New

- **Enhanced Bilateral Reference**: Improves dichotomous segmentation outputs by leveraging dual reference guidance.  
- **Improved Memory Footprint**: Optimized GPU allocation for large-batch or high-resolution workloads.  
- **Configurable Precision**: Support for **FP16** or **FP32** modes (requires GPU with half-precision support for FP16).  
- **Flexible I/O**: Easily integrate your own data loaders or pipeline steps thanks to modular design.

---

## 📂 Project Structure

		BiRefNet/ ├── include 
		          │ └── birefnet.h # Main BiRefNet class definition 
              ├── src 
              │ └── birefnet.cpp # Implementation of the BiRefNet class 
              ├── CMakeLists.txt # CMake configuration 
              └── main.cpp # Demo application


- **include/birefnet.h**  
  Header file defining the `BiRefNet` class, which manages TensorRT engine creation, execution, and memory buffers.

- **src/birefnet.cpp**  
  Source implementation for loading serialized engines, running inference, and handling output postprocessing.

- **CMakeLists.txt**  
  Configuration for building the project using CMake. Adjust paths to TensorRT, CUDA, and OpenCV as needed.

- **main.cpp**  
  A minimal example demonstrating how to load the model, run inference on images or videos, and save the results.

---

## 🚀 Installation

1. **Clone the Repository**

   git clone https://github.com/hamdiboukamcha/BiRefNet-Cpp-TensorRT.git
   cd BiRefNet-Cpp-TensorRT
	 mkdir build && cd build
	 cmake ..
	 make -j$(nproc)

	
## 📦 Dependencies
CUDA
Required for GPU acceleration and kernel launches (e.g., CUDA 11.x or later).

TensorRT
High-performance deep learning inference library (v8.x or later recommended).

OpenCV
Needed for image loading, preprocessing, and basic visualization.

C++17
This project uses modern C++ features. Ensure your compiler supports C++17 or above.				  

## 🔍 Code Overview
Main Components
BiRefNet Class

Initializes a TensorRT engine from a given engine/model path.
Handles preprocessing (image resizing, mean/std normalization, etc.).
Runs inference and postprocesses outputs into segmentation maps.
Manages CUDA resources and streams.
Logger Class (in main.cpp)

Implements TensorRT’s ILogger interface for custom logging.
Notable Functions
BiRefNet::BiRefNet(...)

Constructor that loads a .trt (serialized TensorRT) engine into memory.
BiRefNet::predict(cv::Mat& image)

Main function to run inference: takes an OpenCV cv::Mat as input, returns the segmented result as cv::Mat.
BiRefNet::preprocess(...)

Converts an image into normalized floats (mean subtraction, division by std, etc.).
BiRefNet::postprocess(...)

Reshapes the raw output into meaningful image data, typically an 8-bit or 32-bit matrix for segmentation.

## 🎬 Usage
Prepare Your Engine

Convert your model to ONNX, then build a TensorRT engine (e.g., using trtexec or a custom builder).
	trtexec --onnx=birefnet_model.onnx --saveEngine=BiRefNet-tiny.engine --fp16
	
## 📞 Contact
For further inquiries or advanced usage discussions:

Email: your.email@example.com
LinkedIn: Your Name

## 📜 Citation
If you use BiRefNet C++ TENSORRT in your academic work or research, please cite:

@misc{Boukamcha2025BiRefNet,
    author       = {Hamdi Boukamcha},
    title        = {BiRefNet C++ TENSORRT},
    year         = {2025},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/hamdiboukamcha/BiRefNet-Cpp-TensorRT}}
}
