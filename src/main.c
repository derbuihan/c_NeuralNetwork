#include <stdio.h>

#include "matrix.h"
#include "mnist.h"

#include <stdlib.h>

int main(void) {
  printf("Hello, World!\n");

  double *data = malloc(ROWS * COLS * sizeof(double));
  double *labels = malloc(ROWS * sizeof(double));
  load_mnist_datasets("../datasets/mnist_test.csv", data, labels);

  Matrix *X = new_matrix(ROWS, COLS);
  init_matrix_from_array(X, data, ROWS, COLS);
  free(data);

  // W1
  Matrix *W1 = new_matrix_from_file("../datasets/W1.csv", COLS, 50);

  // W2
  Matrix *W2 = new_matrix_from_file("../datasets/W2.csv", 50, 100);

  // W3
  Matrix *W3 = new_matrix_from_file("../datasets/W3.csv", 100, 10);

  // b1
  Matrix *b1_ = new_matrix_from_file("../datasets/b1.csv", 50, 1);
  Matrix *b1 = new_matrix(1, 50);
  transpose_matrix(b1, b1_);

  // b2
  Matrix *b2_ = new_matrix_from_file("../datasets/b2.csv", 100, 1);
  Matrix *b2 = new_matrix(1, 100);
  transpose_matrix(b2, b2_);

  // b3
  Matrix *b3_ = new_matrix_from_file("../datasets/b3.csv", 10, 1);
  Matrix *b3 = new_matrix(1, 10);
  transpose_matrix(b3, b3_);

  // Forward pass
  Matrix *t1 = new_matrix(ROWS, 50);
  matrix_multiply_matrix(t1, X, W1);

  Matrix *a1 = new_matrix(ROWS, 50);
  matrix_add_vector(a1, t1, b1);

  Matrix *z1 = new_matrix(ROWS, 50);
  sigmoid_matrix(z1, a1);

  Matrix *t2 = new_matrix(ROWS, 100);
  matrix_multiply_matrix(t2, z1, W2);

  Matrix *a2 = new_matrix(ROWS, 100);
  matrix_add_vector(a2, t2, b2);

  Matrix *z2 = new_matrix(ROWS, 100);
  sigmoid_matrix(z2, a2);

  Matrix *t3 = new_matrix(ROWS, 10);
  matrix_multiply_matrix(t3, z2, W3);

  Matrix *a3 = new_matrix(ROWS, 10);
  matrix_add_vector(a3, t3, b3);

  Matrix *y = new_matrix(ROWS, 10);
  softmax_matrix(y, a3);

  int acc_coount = 0;
  for (int i = 0; i < ROWS; i++) {
    int max_idx = 0;
    double max_value = 0;
    for (int j = 0; j < 10; j++) {
      if (y->elements[i * 10 + j] > max_value) {
        max_value = y->elements[i * 10 + j];
        max_idx = j;
      }
    }

    if (abs(labels[i] - max_idx) < 0.001) {
      acc_coount++;
    }
  }

  printf("Accuracy: %f\n", (double)acc_coount / ROWS);

  return 0;
}