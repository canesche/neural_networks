#ifndef MPL__H
#define MLP__H

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

class Perceptron {
public:
  Perceptron(int inputs, double bias = 1.0);
  double run(std::vector<double> x);
  void set_weights(std::vector<double> w_init);
  double sigmoid(double x);

private:
  std::vector<double> weights;
  double bias;
};

#endif