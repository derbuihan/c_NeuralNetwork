#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct Network Network;
struct Network {
  Matrix *W1;
  Matrix *(*forward)(Network *net, Matrix *X);
};

Matrix *forward(Network *net, Matrix *X) {
  Matrix *z1 = new_matrix(X->rows, net->W1->cols);
  matrix_multiply_matrix(z1, X, net->W1);
  return z1;
}

Network *new_network(int num_inputs, int num_outputs) {
  Network *net = malloc(sizeof(Network));
  net->W1 = new_matrix(num_inputs, num_outputs);
  init_matrix_random(net->W1);
  net->forward = forward;
  return net;
}

int main(void) {
  printf("Hello, World!\n");

  Network *net = new_network(10, 3);

  Matrix *X = new_matrix(5, 10);
  init_matrix_random(X);

  Matrix *y = new_matrix(5, 1);
  init_matrix_from_array(y, (double[]){2, 0, 1, 2, 1}, 5, 1);

  Matrix *z = net->forward(net, X);

  double loss = cross_entropy_loss(y, z);
  printf("Loss: %f\n", loss);

  return 0;
}