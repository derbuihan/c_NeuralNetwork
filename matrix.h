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

Vector *new_vector_from_file(const char *filename, int size);
Matrix *new_matrix_from_file(const char *filename, int rows, int cols);

void free_vector(Vector *v);
void free_matrix(Matrix *m);

void init_vector_random(Vector *v);
void init_matrix_random(Matrix *m);
void init_vector_from_array(Vector *v, const double *data, const int size);
void init_matrix_from_array(Matrix *m, double *data, const int rows,
                            const int cols);

void matrix_add_matrixt(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_multiply_matrix(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_add_vector(Matrix *result, const Matrix *m, const Vector *v);
void sigmoid_matrix(Matrix *result, const Matrix *m);
void softmax_matrix(Matrix *result, const Matrix *m);

void print_vector(Vector *v);
void print_matrix(Matrix *m);

double sum_elements_vector(const Vector *v);
double sum_elements_matrix(const Matrix *m);

#endif // MATRIX_H
