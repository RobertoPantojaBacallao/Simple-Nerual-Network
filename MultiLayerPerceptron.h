#pragma once

#include "Perceptron.h"

using namespace std;

class MultiLayerPerceptron {
	vector<int> layers;
	double bias;
	double learning_rate;
	vector<vector<Perceptron>> network;
	vector<vector<double>> values;
	vector<vector<double>> errors;

public:
	MultiLayerPerceptron(vector<int> layers, double bias = 1.0, double learning_rate = 0.5);
	void set_weights(vector<vector<vector<double>>> init_weights);
	void print_weights();
	vector<double> run(vector<double> x);
	double bp(vector<double>x, vector<double> y);
	void calculateDelta(int row, int col, int output);
};