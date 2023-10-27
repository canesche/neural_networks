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

  p->set_weights({15, 15, -10}); // AND Example
  std::cout << "\nOR Gate: \n";
  std::cout << "0 OR 0 = " << p->run({0, 0}) << "\n";
  std::cout << "0 OR 1 = " << p->run({0, 1}) << "\n";
  std::cout << "1 OR 0 = " << p->run({1, 0}) << "\n";
  std::cout << "1 OR 1 = " << p->run({1, 1}) << "\n";

  return 0;
}