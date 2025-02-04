# Feed-Forward-Neural-Network

A simple feed forward neural network trained to a certain number of epochs and gives a prediction for a certain test input, all done using **LibTorch C++**.

## Feedforward Neural Network using LibTorch

This project demonstrates a simple **Feedforward Neural Network (FFNN)** implemented using **LibTorch**, the C++ API of **PyTorch**. The neural network is trained on a random dataset, and the goal is to classify data based on the sum of two features.

### Overview

- The neural network consists of:
  - **Input layer**: 2 features (x, y).
  - **Two hidden layers**: With ReLU activations.
  - **Output layer**: Produces a value that is compared to a target (either 0 or 1).
- **Optimizer**: Adam optimizer.
- **Loss Function**: Mean Squared Error (MSE) loss.

### Requirements

- **LibTorch**: The PyTorch C++ library.
- **CMake**: A tool for managing the build process.
- **A C++17 compatible compiler** (e.g., `g++` or `clang++`).

### Installing LibTorch

1. Go to the [LibTorch website](https://pytorch.org/get-started/locally/) and download the appropriate version for your system.
2. Extract the `libtorch` folder to a convenient location.

### CMakeLists.txt File

Let this be the content of your **CMakeLists.txt** file:

```cmake
cmake_minimum_required(VERSION 3.1)
project(main)

# Set the path to the LibTorch folder
set(Torch_DIR "/mnt/c/libtorch/share/cmake/Torch")

# Set C++ standard to 17 and disable C++ extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the Torch package
find_package(Torch REQUIRED)

# Add the executable and link the libraries
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Set C++ standard for the target
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
