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
  // free(data);
  printf("%lf\n", sum_elements_matrix(X));
  // print_matrix(X);

  Vector *y = new_vector(ROWS);
  init_vector_from_array(y, labels, ROWS);
  // free(labels);
  printf("%lf\n", sum_elements_vector(y));
  // print_vector(y);

  // W1
  double data1[COLS][50];
  load_csv("../datasets/W1.csv", data1, COLS, 50);
  Matrix *W1 = new_matrix(COLS, 50);
  init_matrix_from_array(W1, data1, COLS, 50);
  printf("%lf\n", sum_elements_matrix(W1));
  // print_matrix(W1);

  return 0;
}