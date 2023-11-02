#include "mlp.h"
#include <iostream>

int main(int argc, char *argv[]) {
  srand(time(NULL));
  rand();

  std::cout << "----- Logic Gate Example -----\n\n";

  Perceptron *p = new Perceptron(2);
  p->set_weights({10, 10, -15}); // AND Example

  std::cout << "AND Gate: \n";
  std::cout << "0 AND 0 = " << p->run({0, 0}) << "\n";
  std::cout << "0 AND 1 = " << p->run({0, 1}) << "\n";
  std::cout << "1 AND 0 = " << p->run({1, 0}) << "\n";
  std::cout << "1 AND 1 = " << p->run({1, 1}) << "\n";

  p->set_weights({15, 15, -10}); // OR Example
  std::cout << "\nOR Gate: \n";
  std::cout << "0 OR 0 = " << p->run({0, 0}) << "\n";
  std::cout << "0 OR 1 = " << p->run({0, 1}) << "\n";
  std::cout << "1 OR 0 = " << p->run({1, 0}) << "\n";
  std::cout << "1 OR 1 = " << p->run({1, 1}) << "\n";

  MultiLayerPerceptron *mlp = new MultiLayerPerceptron({2, 2, 1});
  mlp->set_weights({{{-10, -10, 15}, {15, 15, -10}}, {{10, 10, -15}}});
  std::cout << "Weights for XOR: \n";
  mlp->print_weights();

  std::cout << "\nXOR Gate: \n";
  std::cout << "0 XOR 0 = " << mlp->run({0, 0})[0] << "\n";
  std::cout << "0 XOR 1 = " << mlp->run({0, 1})[0] << "\n";
  std::cout << "1 XOR 0 = " << mlp->run({1, 0})[0] << "\n";
  std::cout << "1 XOR 1 = " << mlp->run({1, 1})[0] << "\n";

  // Tained XOR
  std::cout << "\nTrained XOR using neural network\n";
  MultiLayerPerceptron *mlp_trained = new MultiLayerPerceptron({2, 2, 1}); 
  double MSE;
  for (int i = 0; i < 3000; ++i) {
    MSE = 0.0;
    MSE += mlp_trained->back_propagation({0,0},{0});
    MSE += mlp_trained->back_propagation({0,1},{1});
    MSE += mlp_trained->back_propagation({1,0},{1});
    MSE += mlp_trained->back_propagation({1,1},{0});
    MSE /= 4.0;
    if (i % 100 == 0) {
      std::cout << "MSE = " << MSE << std::endl;
    }
  }

  mlp_trained->print_weights();
  std::cout << "\nXOR Gate: \n";
  std::cout << "0 XOR 0 = " << mlp_trained->run({0, 0})[0] << "\n";
  std::cout << "0 XOR 1 = " << mlp_trained->run({0, 1})[0] << "\n";
  std::cout << "1 XOR 0 = " << mlp_trained->run({1, 0})[0] << "\n";
  std::cout << "1 XOR 1 = " << mlp_trained->run({1, 1})[0] << "\n";

  return 0;
}