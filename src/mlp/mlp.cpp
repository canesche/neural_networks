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

// Get the weights
std::vector<double> Perceptron::get_weights() {
  return this->weights;
}

// Evaluate the sigmoid function
double Perceptron::sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Return a new MultiLayerPerceptron object
MultiLayerPerceptron::MultiLayerPerceptron(std::vector<int> layers, double bias,
                                           double eta) {
  this->layers = layers;
  this->bias = bias;
  this->eta = eta;

  for (int i = 0; i < layers.size(); ++i) {
    this->values.push_back(std::vector<double>(layers[i], 0.0));
    this->network.push_back(std::vector<Perceptron>());
    if (i > 0) { // Network in pos 0 is the input layer, It has no neurons
      for (int j = 0; j < layers[i]; ++j) {
        this->network[i].push_back(Perceptron(layers[i-1], bias));
      }
    }
  }
}

void MultiLayerPerceptron::set_weights(std::vector<std::vector<std::vector<double>>> w_init) {
  for (int i = 0; i < w_init.size(); ++i) {
    for (int j = 0; j < w_init[i].size(); ++j) {
      this->network[i+1][j].set_weights(w_init[i][j]);
    }
  }
}

void MultiLayerPerceptron::print_weights() {
  for (int i = 1; i < this->network.size(); ++i) {
    for (int j = 0; j < layers[i]; ++j) {
      std::cout << "Layer " << i+1 << " Neuron " << j << ": ";
      for (auto &it : this->network[i][j].get_weights())
        std::cout << it << " ";
      std::cout << "\n";
    }
  }
}

// Run the Perceptron
std::vector<double> MultiLayerPerceptron::run(std::vector<double> x) {
  this->values[0] = x;
  for (int i = 1; i < this->network.size(); ++i) {
    for (int j = 0; j < layers[i]; ++j) {
      this->values[i][j] = this->network[i][j].run(this->values[i-1]);
    }
  }
  return this->values.back();
}