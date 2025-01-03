#include "simple_network.h"

#include "mnist.h"

#include <math.h>
#include <stdlib.h>

void init_network(Network *net) {
  int COLS = 28 * 28;

  net->W1 = new_matrix(COLS, 50);
  net->W2 = new_matrix(50, 100);
  net->W3 = new_matrix(100, 10);
  init_matrix_uniform_random(net->W1, -1.0 / sqrt(COLS), 1.0 / sqrt(COLS));
  init_matrix_uniform_random(net->W2, -1.0 / sqrt(50), 1.0 / sqrt(50));
  init_matrix_uniform_random(net->W3, -1.0 / sqrt(100), 1.0 / sqrt(100));

  net->b1 = new_matrix(1, 50);
  net->b2 = new_matrix(1, 100);
  net->b3 = new_matrix(1, 10);
  init_matrix_uniform_random(net->b1, -1.0 / sqrt(COLS), 1.0 / sqrt(COLS));
  init_matrix_uniform_random(net->b2, -1.0 / sqrt(50), 1.0 / sqrt(50));
  init_matrix_uniform_random(net->b3, -1.0 / sqrt(100), 1.0 / sqrt(100));

  net->grad_W1 = new_matrix(COLS, 50);
  net->grad_W2 = new_matrix(50, 100);
  net->grad_W3 = new_matrix(100, 10);

  net->grad_b1 = new_matrix(1, 50);
  net->grad_b2 = new_matrix(1, 100);
  net->grad_b3 = new_matrix(1, 10);
}

void free_network(Network *net) {
  free_matrix(net->W1);
  free_matrix(net->W2);
  free_matrix(net->W3);
  free_matrix(net->b1);
  free_matrix(net->b2);
  free_matrix(net->b3);
  free(net);
}

Matrix *forward(Network *net, Matrix *X) {
  Matrix *t1 = new_matrix(X->rows, 50);
  matrix_mul_matrix(t1, X, net->W1);

  Matrix *a1 = new_matrix(X->rows, 50);
  matrix_add_vector(a1, t1, net->b1);

  Matrix *z1 = new_matrix(X->rows, 50);
  sigmoid_matrix(z1, a1);

  Matrix *t2 = new_matrix(X->rows, 100);
  matrix_mul_matrix(t2, z1, net->W2);

  Matrix *a2 = new_matrix(X->rows, 100);
  matrix_add_vector(a2, t2, net->b2);

  Matrix *z2 = new_matrix(X->rows, 100);
  sigmoid_matrix(z2, a2);

  Matrix *t3 = new_matrix(X->rows, 10);
  matrix_mul_matrix(t3, z2, net->W3);

  Matrix *a3 = new_matrix(X->rows, 10);
  matrix_add_vector(a3, t3, net->b3);

  // Matrix *y = new_matrix(X->rows, 10);
  // softmax_matrix(y, a3);

  free_matrix(t1);
  free_matrix(a1);
  free_matrix(z1);
  free_matrix(t2);
  free_matrix(a2);
  free_matrix(z2);
  free_matrix(t3);
  // free_matrix(a3);

  return a3;
}

void backward(Network *net, Matrix *X, Matrix *y_true) {
  // Forward pass
  Matrix *t1 = new_matrix(X->rows, 50);
  matrix_mul_matrix(t1, X, net->W1);
  Matrix *a1 = new_matrix(X->rows, 50);
  matrix_add_vector(a1, t1, net->b1);
  Matrix *z1 = new_matrix(X->rows, 50);
  sigmoid_matrix(z1, a1);

  Matrix *t2 = new_matrix(X->rows, 100);
  matrix_mul_matrix(t2, z1, net->W2);
  Matrix *a2 = new_matrix(X->rows, 100);
  matrix_add_vector(a2, t2, net->b2);
  Matrix *z2 = new_matrix(X->rows, 100);
  sigmoid_matrix(z2, a2);

  Matrix *t3 = new_matrix(X->rows, 10);
  matrix_mul_matrix(t3, z2, net->W3);
  Matrix *a3 = new_matrix(X->rows, 10);
  matrix_add_vector(a3, t3, net->b3);

  // Backward pass
  Matrix *dL_da3 = new_matrix(X->rows, 10);
  matrix_sub_matrix(dL_da3, a3, y_true);

  matrix_transpose_mul_matrix(net->grad_W3, z2, dL_da3);
  matrix_sum_rows(net->grad_b3, dL_da3);

  Matrix *dL_dz2 = new_matrix(X->rows, 100);
  matrix_mul_matrix_transpose(dL_dz2, dL_da3, net->W3);
  Matrix *dL_da2 = new_matrix(X->rows, 100);
  sigmoid_derivative_matrix(dL_da2, a2);
  matrix_elementwise_mul(dL_dz2, dL_dz2, dL_da2);

  matrix_transpose_mul_matrix(net->grad_W2, z1, dL_dz2);
  matrix_sum_rows(net->grad_b2, dL_dz2);

  Matrix *dL_dz1 = new_matrix(X->rows, 50);
  matrix_mul_matrix_transpose(dL_dz1, dL_dz2, net->W2);
  Matrix *dL_da1 = new_matrix(X->rows, 50);
  sigmoid_derivative_matrix(dL_da1, a1);
  matrix_elementwise_mul(dL_dz1, dL_dz1, dL_da1);

  matrix_transpose_mul_matrix(net->grad_W1, X, dL_dz1);
  matrix_sum_rows(net->grad_b1, dL_dz1);

  // Free temporary matrices
  free_matrix(t1);
  free_matrix(a1);
  free_matrix(z1);
  free_matrix(t2);
  free_matrix(a2);
  free_matrix(z2);
  free_matrix(t3);
  free_matrix(a3);
  free_matrix(dL_da3);
  free_matrix(dL_dz2);
  free_matrix(dL_da2);
  free_matrix(dL_dz1);
  free_matrix(dL_da1);
}

Network *new_network() {
  Network *net = malloc(sizeof(Network));
  net->init = init_network;
  net->free = free_network;
  net->forward = forward;
  net->backward = backward;
  net->init(net);
  return net;
}
