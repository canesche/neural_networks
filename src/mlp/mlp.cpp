#include "mlp.h"

double frand() { return (2.0 * (double)rand() / RAND_MAX) - 1.0; }

// Return a new Perceptron object
Perceptron::Perceptron(int inputs, double bias) {
  this->bias = bias;
  weights.resize(inputs + 1);
  std::generate(weights.begin(), weights.end(), frand);
}

// Run the perceptron
double Perceptron::run(std::vector<double> x) {
  x.push_back(bias);
  double sum =
      std::inner_product(x.begin(), x.end(), weights.begin(), (double)0.0);
  return sigmoid(sum);
}

// Set the weights
void Perceptron::set_weights(std::vector<double> w_init) {
  this->weights = w_init;
}

// Evaluate the sigmoid function
double Perceptron::sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
