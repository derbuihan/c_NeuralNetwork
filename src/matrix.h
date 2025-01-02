#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
  double *elements;
  int rows;
  int cols;
} Matrix;

Matrix *new_matrix(int rows, int cols);
Matrix *new_matrix_from_file(const char *filename, int rows, int cols);

void free_matrix(Matrix *m);

void init_matrix_uniform_random(Matrix *m, double low, double high);
void init_matrix_normal_random(Matrix *m, double mean, double std);
void init_matrix_from_array(Matrix *m, double *data, const int rows,
                            const int cols);
void init_matrix_from_file(Matrix *m, const char *filename, int rows, int cols);

void matrix_add_matrixt(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_mul_matrix(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_sub_matrix(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_add_vector(Matrix *result, const Matrix *m, const Matrix *v);
void matrix_mul_scalar(Matrix *result, const Matrix *m, const double scalar);
void sigmoid_matrix(Matrix *result, const Matrix *m);
void softmax_matrix(Matrix *result, const Matrix *m);
void transpose_matrix(Matrix *result, const Matrix *m);
double cross_entropy_loss(const Matrix *y, const Matrix *m);
void matrix_sum_rows(Matrix *result, Matrix *m);
void matrix_transpose_mul_matrix(Matrix *result, const Matrix *a,
                                 const Matrix *b);
void matrix_mul_matrix_transpose(Matrix *result, const Matrix *a,
                                 const Matrix *b);
void sigmoid_derivative_matrix(Matrix *result, const Matrix *m);
void matrix_elementwise_mul(Matrix *result, const Matrix *a, const Matrix *b);

void print_matrix(Matrix *m);

#endif // MATRIX_H
