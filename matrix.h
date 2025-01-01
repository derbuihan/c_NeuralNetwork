#ifndef MATRIX_H
#define MATRIX_H

typedef struct Vector {
  double *elements;
  int size;
} Vector;

typedef struct Matrix {
  double *elements;
  int rows;
  int cols;
} Matrix;

Vector *vector_create(int size);
Matrix *matrix_create(int rows, int cols);
void free_vector(Vector *v);
void free_matrix(Matrix *m);

void init_vector_from_array(Vector *v, const double *data);
void init_matrix_from_array(Matrix *m, const double **data);

#endif // MATRIX_H
