// Neural Network.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "MultiLayerPerceptron.h"
#include <format>
#include <chrono>

using namespace std::literals::chrono_literals;
using namespace std;


// Python-style print function
constexpr void print(const std::string_view str_fmt, auto&&... args) {
    fputs(std::vformat(str_fmt, std::make_format_args(args...)).c_str(), stdout);
}

// Trains MLP on set of inputs & Outputs
void trainMLP(MultiLayerPerceptron& mlp, const vector<vector<double>> inputs, const vector<vector<double>> labels, const int epochs) {

    if (inputs.size() != labels.size()) {
        print("ERROR: inputs and labels have different number of samples\n");
        return;
    }

    auto start_time = std::chrono::steady_clock::now();

    double MSE;

    for (int i = 0; i < epochs; i++) {

        MSE = 0.0;

        for (int e = 0; e < inputs.size(); e++) {
            MSE += mlp.bp(inputs[e], labels[e]);
        }

        MSE /= inputs.size();
    }

    print("Finished Training\nMSE: {}\nElapsed Time: {}ms\n\n", MSE, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count());
}


int main()
{

    // Training set of Logic Gates as example us of NN
    MultiLayerPerceptron orGate = MultiLayerPerceptron({ 2, 2, 1 });
    MultiLayerPerceptron andGate = MultiLayerPerceptron({ 2, 2, 1 });
    MultiLayerPerceptron xorGate = MultiLayerPerceptron({ 2, 2, 1 });
    MultiLayerPerceptron xnorGate = MultiLayerPerceptron({ 2, 2, 1 });
    MultiLayerPerceptron nandGate = MultiLayerPerceptron({ 2, 2, 1 });
    MultiLayerPerceptron norGate = MultiLayerPerceptron({ 2, 2, 1 });

    trainMLP(orGate, { {0,0}, {0,1}, {1,0}, {1,1} }, { {0}, {1}, {1}, {1} }, 3000);
    trainMLP(andGate, { {0,0}, {0,1}, {1,0}, {1,1} }, { {0}, {0}, {0}, {1} }, 3000);               
    trainMLP(xorGate, { {0,0}, {0,1}, {1,0}, {1,1} }, { {0}, {1}, {1}, {0} }, 3000);       
    trainMLP(xnorGate, { {0,0}, {0,1}, {1,0}, {1,1} }, { {1}, {0}, {0}, {1} }, 3000);
    trainMLP(norGate, { {0,0}, {0,1}, {1,0}, {1,1} }, { {1}, {0}, {0}, {0} }, 3000);
    trainMLP(nandGate, { {0,0}, {0,1}, {1,0}, {1,1} }, { {1}, {1}, {1}, {0} }, 3000);

    print("OR Behavior:\n{}\n{}\n{}\n{}\n\n", orGate.run({ 0,0 })[0], orGate.run({ 0,1 })[0], orGate.run({ 1,0 })[0], orGate.run({ 1,1 })[0]);
    print("AND Behavior:\n{}\n{}\n{}\n{}\n\n", andGate.run({ 0,0 })[0], andGate.run({ 0,1 })[0], andGate.run({ 1,0 })[0], andGate.run({ 1,1 })[0]);
    print("XOR Behavior:\n{}\n{}\n{}\n{}\n\n", xorGate.run({ 0,0 })[0], xorGate.run({ 0,1 })[0], xorGate.run({ 1,0 })[0], xorGate.run({ 1,1 })[0]);
    print("XNOR Behavior:\n{}\n{}\n{}\n{}\n\n", xnorGate.run({ 0,0 })[0], xnorGate.run({ 0,1 })[0], xnorGate.run({ 1,0 })[0], xnorGate.run({ 1,1 })[0]);
    print("NOR Behavior:\n{}\n{}\n{}\n{}\n\n", norGate.run({ 0,0 })[0], norGate.run({ 0,1 })[0], norGate.run({ 1,0 })[0], norGate.run({ 1,1 })[0]);
    print("NAND Behavior:\n{}\n{}\n{}\n{}\n\n", nandGate.run({ 0,0 })[0], nandGate.run({ 0,1 })[0], nandGate.run({ 1,0 })[0], nandGate.run({ 1,1 })[0]);
}
