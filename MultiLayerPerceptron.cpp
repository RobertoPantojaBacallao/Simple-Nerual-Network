#include "MultiLayerPerceptron.h"

MultiLayerPerceptron::MultiLayerPerceptron(vector<int> layers, double bias, double learning_rate) {
	this->layers = layers;
	this->bias = bias;
	this->learning_rate = learning_rate;

	for (int i = 0; i < layers.size(); i++) {
		values.push_back(vector<double>(layers[i], 0.0));
		errors.push_back(vector<double>(layers[i], 0.0));
		network.push_back(vector<Perceptron>());

		 // Skip input layer, then populate network with perceptrons
		if (i > 0){
			for (int j = 0; j < layers[i]; j++) {
				network[i].push_back(Perceptron(layers[i - 1], bias));
			}
		}
	}
}

void MultiLayerPerceptron::set_weights(vector<vector<vector<double>>> init_weights) {
	for (int i = 0; i < init_weights.size(); i++) {
		for (int j = 0; j < init_weights[i].size(); j++) {
			network[i+1][j].set_weights(init_weights[i][j]);
		}
	}
}

void MultiLayerPerceptron::print_weights() {
	cout << endl;

	for (int i = 1; i < network.size(); i++) {
		for (int j = 0; j < layers[i]; j++) {
			cout << "Layer " << i + 1 << " Neuron " << j << ": ";
			for (auto& it : network[i][j].weights) {
				cout << it << "   ";
			}
			cout << endl;
		}
	}
}

vector<double> MultiLayerPerceptron::run(vector<double> x) {
	values[0] = x;
	for (int i = 1; i < network.size(); i++) {
		for (int j = 0; j < layers[i]; j++) {
			values[i][j] = network[i][j].run(values[i - 1]);
		}
	}

	return values.back();
}

double MultiLayerPerceptron::bp(vector<double> x, vector<double> y) {
	vector<double> output = run(x);

	vector<double> error;
	double MSE = 0.0;
	for (int i = 0; i < y.size(); i++) {
		error.push_back(y[i] - output[i]);
		MSE += pow(error[i], 2);
	}
	MSE /= layers.back();

	// Calculate output error terms
	for (int i = 0; i < output.size(); i++) {
		errors.back()[i] = output[i] * (1 - output[i]) * error[i];
	}

	// Calculate error term of each unit in network
	for (int i = network.size() - 2; i > 0; i--) {
		for (int h = 0; h < network[i].size(); h++) {
			double fwd_error = 0.0;
			for (int k = 0; k < layers[i + 1]; k++) {
				fwd_error += network[i + 1][k].weights[h] * errors[i + 1][k];
			}
			errors[i][h] = values[i][h] * (1 - values[i][h]) * fwd_error;
		}
	}


	for (int i = 1; i < network.size(); i++) {
		for (int j = 0; j < layers[i]; j++) {
			for (int k = 0; k < layers[i - 1] + 1; k++) {
				// Calculate Delta
				double delta;
				if (k == layers[i - 1]) {
					delta = learning_rate * errors[i][j] * bias;
				}
				else {
					delta = learning_rate * errors[i][j] * values[i - 1][k];
				}

				// Update weights
				network[i][j].weights[k] += delta;
			}
		}
	}

	return MSE;

}