#include <torch/torch.h>
#include <iostream>

// Define a simple FeedForward neural network
struct Net : torch::nn::Module {
    torch::nn::Linear fc1, fc2, fc3;

    Net(int64_t input_size, int64_t hidden_size, int64_t output_size)
        : fc1(input_size, hidden_size), fc2(hidden_size, hidden_size), fc3(hidden_size, output_size) {
        // Register the layers as submodules
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    // Define the forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = fc3(x);
        return x;
    }
};

int main() {
    // Setting up the device
    torch::Device device(torch::kCPU);

    // Hyperparameters
    const int64_t input_size = 2;    // Two features (x, y)
    const int64_t hidden_size = 64;  // Number of neurons in hidden layers
    const int64_t output_size = 1;   // Output (0 or 1)
    const int64_t batch_size = 10;
    const int64_t num_epochs = 100;
    const double learning_rate = 0.001;

    // Creating the model
    Net model(input_size, hidden_size, output_size);
    model.to(device);

    // Optimizer
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(learning_rate));

    // Create a random dataset (2D points)
    torch::Tensor inputs = torch::randn({100, input_size});
    torch::Tensor targets = torch::sum(inputs, 1).unsqueeze(1).gt(0).to(torch::kFloat32);  // Target: 1 if sum(x, y) > 0 else 0

    // Training loop
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        model.train();

        // Forward pass
        optimizer.zero_grad();
        torch::Tensor outputs = model.forward(inputs);
        torch::Tensor loss = torch::mse_loss(outputs, targets);

        // Backpropagation
        loss.backward();
        optimizer.step();

        // Printing the loss every 10 epochs
        if (epoch % 10 == 0) {
            std::cout << "Epoch [" << epoch << "/" << num_epochs << "], Loss: " << loss.item<float>() << std::endl;
        }
    }

    // Testing
    model.eval();
    torch::Tensor test_input = torch::randn({1, input_size});
    torch::Tensor prediction = model.forward(test_input);

    std::cout << "Test input: " << test_input << std::endl;
    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
}
