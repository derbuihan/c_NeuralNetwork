#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

Vector *new_vector(int size) {
  Vector *v = malloc(sizeof(Vector));

  v->elements = malloc(size * sizeof(double));
  v->size = size;

  return v;
}
Matrix *new_matrix(int rows, int cols) {
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

void init_vector_from_array(Vector *v, const double *data, const int size) {
  if (v->size != size) {
    fprintf(stderr, "Error: size mismatch\n");
    exit(1);
  }

  for (int i = 0; i < size; i++) {
    v->elements[i] = data[i];
  }
}

void init_matrix_from_array(Matrix *m, double *data, const int rows,
                            const int cols) {
  if (m->rows != rows || m->cols != cols) {
    fprintf(stderr, "Error: size mismatch\n");
    exit(1);
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      m->elements[i * cols + j] = data[i * cols + j];
    }
  }
}

void print_vector(Vector *v) {
  for (int i = 0; i < v->size; i++) {
    printf("%f ", v->elements[i]);
  }
  printf("\n");
}

void print_matrix(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      printf("%f ", m->elements[i * m->cols + j]);
    }
    printf("\n");
  }
}
