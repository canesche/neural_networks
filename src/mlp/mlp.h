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
  std::vector<double> get_weights();

private:
  std::vector<double> weights;
  double bias;
};

class MultiLayerPerceptron {
public:
  MultiLayerPerceptron(std::vector<int> layers, double bias = 1.0,
                       double eta = 0.5);
  void set_weights(std::vector<std::vector<std::vector<double>>> w_init);
  void print_weights();
  std::vector<double> run(std::vector<double> x);
  double back_propagation(std::vector<double> x, std::vector<double> y);

private:
  std::vector<int> layers;
  double bias;
  double eta;
  std::vector<std::vector<Perceptron>> network;
  std::vector<std::vector<double>> values;
  std::vector<std::vector<double>> d;
};

#endif