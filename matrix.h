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

Vector *new_vector(int size);
Matrix *new_matrix(int rows, int cols);

void free_vector(Vector *v);
void free_matrix(Matrix *m);

void init_vector_from_array(Vector *v, const double *data, const int size);
void init_matrix_from_array(Matrix *m, double *data, const int rows,
                            const int cols);

void print_vector(Vector *v);
void print_matrix(Matrix *m);

#endif // MATRIX_H
