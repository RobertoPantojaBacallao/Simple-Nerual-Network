#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>

using namespace std;

class Perceptron {
public:
	vector<double> weights;
	double bias;
	Perceptron(int inputs, double bias = 1.0);
	double run(vector<double> init_weights);
	double sigmoid(double x);
	void set_weights(vector<double> new_weights);
};