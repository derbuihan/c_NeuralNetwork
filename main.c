#include <stdio.h>

#include "matrix.h"
#include "mnist.h"

int main(void) {

  double data[ROWS][COLS];
  double labels[ROWS];
  load_csv("../datasets/mnist_test.csv", data, labels);

  Matrix *X = new_matrix(ROWS, COLS);
  init_matrix_from_array(X, data, ROWS, COLS);

  Vector *y = new_vector(ROWS);
  init_vector_from_array(y, labels, ROWS);

  printf("Hello, World!\n");

  // print_matrix(X);
  // print_vector(y);

  return 0;
}