#include <stdio.h>

#include "matrix.h"
#include "mnist.h"

int main(void) {

  double data[ROWS][COLS];
  int labels[ROWS];
  load_csv("../datasets/mnist_test.csv", data, labels);

  Matrix *X = matrix_create(ROWS, COLS);
  init_matrix_from_array(X, data);

  Matrix *y = matrix_create(ROWS, 1);
  init_matrix_from_array(y, labels);

  printf("Hello, World!\n");
  return 0;
}