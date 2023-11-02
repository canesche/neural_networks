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
std::vector<double> Perceptron::get_weights() { return this->weights; }

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
        this->network[i].push_back(Perceptron(layers[i - 1], bias));
      }
    }
  }
}

void MultiLayerPerceptron::set_weights(
    std::vector<std::vector<std::vector<double>>> w_init) {
  for (int i = 0; i < w_init.size(); ++i) {
    for (int j = 0; j < w_init[i].size(); ++j) {
      this->network[i + 1][j].set_weights(w_init[i][j]);
    }
  }
}

void MultiLayerPerceptron::print_weights() {
  for (int i = 1; i < this->network.size(); ++i) {
    for (int j = 0; j < layers[i]; ++j) {
      std::cout << "Layer " << i + 1 << " Neuron " << j << ": ";
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
      this->values[i][j] = this->network[i][j].run(this->values[i - 1]);
    }
  }
  return this->values.back();
}

// Run Back propagation
double MultiLayerPerceptron::back_propagation(std::vector<double> x, std::vector<double> y) {
  // Feed a sample to the network
  std::vector<double> outputs = run(x);
  
  std::cout << "ola\n";

  // Calculate the MSE
  std::vector<double> error;
  double MSE = 0.0;
  for (int i = 0; i < y.size(); ++i) {
    error.push_back(y[i] - outputs[i]);
    MSE += error[i] * error[i];
  }
  MSE /= this->layers.back();

  std::cout << "ola\n";

  // Calculate the output error terms
  for (int i = 0; i < outputs.size(); ++i) {
    this->d.back()[i] = outputs[i] * (1 - outputs[i]) * error[i];
  }

  std::cout << "ola\n";

  // Calculate the error term of each unit on each layer
  for (int i = this->network.size()-2; i > 0; --i) {
    for (int j = 0; j < this->network[i].size(); ++j) {
      double fwd_error = 0.0;
      for (int k = 0; k < this->layers[i+1]; ++k) {
        fwd_error += this->network[i+1][k].get_weights()[j] * this->d[i+1][k];
      }
      this->d[i][j] = this->values[i][j] * (1-this->values[i][j]) * fwd_error;
    }
  }

  // Calculate the deltas and update the weights
  for (int i = 1; i < this->network.size(); ++i) {
    for (int j = 0; j < this->layers[i]; ++j) {
      for (int k = 0; k < this->layers[i-1]+1; ++k) {
        double delta;
        if (k == this->layers[i-1]) {
          delta = this->eta * this->d[i][j] * bias;
        } else {
          delta = this->eta * this->d[i][j] * this->values[i-1][k];
        }
        this->network[i][j].get_weights()[k] += delta;
      }
    }
  }
  return MSE;
}