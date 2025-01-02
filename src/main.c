#include "matrix.h"
#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct Network Network;
struct Network {
  Matrix *W1;
  Matrix *W2;
  Matrix *W3;
  Matrix *b1;
  Matrix *b2;
  Matrix *b3;
  void (*init)(Network *net);
  void (*free)(Network *net);
  Matrix *(*forward)(Network *net, Matrix *X);
};

void init_network(Network *net) {
  net->W1 = new_matrix(COLS, 50);
  net->W2 = new_matrix(50, 100);
  net->W3 = new_matrix(100, 10);

  net->b1 = new_matrix(1, 50);
  net->b2 = new_matrix(1, 100);
  net->b3 = new_matrix(1, 10);
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
  matrix_multiply_matrix(t1, X, net->W1);

  Matrix *a1 = new_matrix(X->rows, 50);
  matrix_add_vector(a1, t1, net->b1);

  Matrix *z1 = new_matrix(X->rows, 50);
  sigmoid_matrix(z1, a1);

  Matrix *t2 = new_matrix(X->rows, 100);
  matrix_multiply_matrix(t2, z1, net->W2);

  Matrix *a2 = new_matrix(X->rows, 100);
  matrix_add_vector(a2, t2, net->b2);

  Matrix *z2 = new_matrix(X->rows, 100);
  sigmoid_matrix(z2, a2);

  Matrix *t3 = new_matrix(X->rows, 10);
  matrix_multiply_matrix(t3, z2, net->W3);

  Matrix *a3 = new_matrix(X->rows, 10);
  matrix_add_vector(a3, t3, net->b3);

  Matrix *y = new_matrix(X->rows, 10);
  softmax_matrix(y, a3);

  free_matrix(t1);
  free_matrix(a1);
  free_matrix(z1);
  free_matrix(t2);
  free_matrix(a2);
  free_matrix(z2);
  free_matrix(t3);
  free_matrix(a3);

  return y;
}

Network *new_network() {
  Network *net = malloc(sizeof(Network));
  net->init = init_network;
  net->free = free_network;
  net->forward = forward;
  net->init(net);
  return net;
}

void init_matrix_from_file(Matrix *m, const char *filename, int rows,
                           int cols) {
  double *data = malloc(rows * cols * sizeof(double));
  load_csv(filename, data, rows, cols);
  init_matrix_from_array(m, data, rows, cols);
  free(data);
}

int main(void) {
  printf("Hello, World!\n");

  // Load datasets
  double *data = malloc(ROWS * COLS * sizeof(double));
  double *labels = malloc(ROWS * sizeof(double));
  load_mnist_datasets("../datasets/mnist_test.csv", data, labels);

  Matrix *X = new_matrix(ROWS, COLS);
  init_matrix_from_array(X, data, ROWS, COLS);
  free(data);

  // Load weights
  Network *net = new_network();
  init_matrix_from_file(net->W1, "../datasets/W1.csv", COLS, 50);
  init_matrix_from_file(net->W2, "../datasets/W2.csv", 50, 100);
  init_matrix_from_file(net->W3, "../datasets/W3.csv", 100, 10);

  Matrix *b1_ = new_matrix(50, 1);
  init_matrix_from_file(b1_, "../datasets/b1.csv", 50, 1);
  transpose_matrix(net->b1, b1_);
  free_matrix(b1_);

  Matrix *b2_ = new_matrix(100, 1);
  init_matrix_from_file(b2_, "../datasets/b2.csv", 100, 1);
  transpose_matrix(net->b2, b2_);
  free_matrix(b2_);

  Matrix *b3_ = new_matrix(10, 1);
  init_matrix_from_file(b3_, "../datasets/b3.csv", 10, 1);
  transpose_matrix(net->b3, b3_);
  free_matrix(b3_);

  // calculate accuracy
  int correct = 0;
  Matrix *y = net->forward(net, X);
  for (int i = 0; i < ROWS; i++) {
    int label = (int)labels[i];
    int pred = 0;
    double max = y->elements[i * y->cols];
    for (int j = 1; j < y->cols; j++) {
      if (y->elements[i * y->cols + j] > max) {
        max = y->elements[i * y->cols + j];
        pred = j;
      }
    }
    if (abs(pred - label) < 0.001) {
      correct++;
    }
  }
  printf("Accuracy: %.2f%%\n", (double)correct / ROWS * 100);

  return 0;
}