#include "Perceptron.h"

double frand() {
	return (2.0 * static_cast<double>(rand()) / RAND_MAX) - 1.0;
}

Perceptron::Perceptron(int inputs, double bias) {
	this->bias = bias;
	weights.resize(inputs + 1);
	generate(weights.begin(), weights.end(), frand);
}

double Perceptron::run(vector<double> init_weights) {
	init_weights.push_back(bias);
	return sigmoid(inner_product(init_weights.begin(), init_weights.end(), weights.begin(), static_cast<double>(0.0)));
}

void Perceptron::set_weights(vector<double> new_weights) {
	this->weights = new_weights;
}

double Perceptron::sigmoid(double x) {
	return ( 1.0 / ( 1.0 + exp(-x) ) );
}

