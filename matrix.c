#include "matrix.h"

#include <stdlib.h>

Vector *vector_create(int size) {
  Vector *v = malloc(sizeof(Vector));

  v->elements = malloc(size * sizeof(double));
  v->size = size;

  return v;
}
Matrix *matrix_create(int rows, int cols) {
  Matrix *m = malloc(sizeof(Matrix));

  m->elements = malloc(rows * cols * sizeof(double));
  m->rows = rows;
  m->cols = cols;

  return m;
}

void free_vector(Vector *v) {
  free(v->elements);
  free(v);
}

void free_matrix(Matrix *m) {
  free(m->elements);
  free(m);
}

void init_vector_from_array(Vector *v, const double *data) {
  for (int i = 0; i < v->size; i++) {
    v->elements[i] = data[i];
  }
}

void init_matrix_from_array(Matrix *m, const double **data) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->elements[i * m->cols + j] = data[i][j];
    }
  }
}
