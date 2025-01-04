#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix Matrix;
struct Matrix {
  double *elements;
  double *gradients;
  int rows;
  int cols;

  Matrix **inputs;
  int num_inputs;

  void (*backward)(Matrix *self);
  void (*zero_grad)(Matrix *self);
};

void free_matrix(Matrix *m);
Matrix *new_matrix(int rows, int cols);
Matrix *new_matrix_from_file(char *filename, int rows, int cols);

void init_matrix_uniform_random(Matrix *m, double low, double high);
void init_matrix_normal_random(Matrix *m, double mean, double std);
void init_matrix_from_array(Matrix *m, double *data, int rows, int cols);
void init_matrix_from_file(Matrix *m, char *filename, int rows, int cols);

void matrix_add_matrix(Matrix *result, Matrix *a, Matrix *b);
void matrix_mul_matrix(Matrix *result, Matrix *a, Matrix *b);
void matrix_add_vector(Matrix *result, Matrix *m, Matrix *v);
void sigmoid_matrix(Matrix *result, Matrix *m);
void softmax_matrix(Matrix *result, Matrix *m);
double cross_entropy_loss(Matrix *y_true, Matrix *y_pred);

void print_matrix(Matrix *m);

#endif // MATRIX_H
