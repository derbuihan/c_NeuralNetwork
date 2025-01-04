#include "matrix.h"

#include "mnist.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void backward_none(Matrix *self) {
  if (self->num_inputs != 0) {
    printf("backward_none\n");
  }
  // Do nothing
}

static void zero_grad_matrix(Matrix *self) {
  for (int i = 0; i < self->rows * self->cols; i++) {
    self->gradients[i] = 0;
  }
}

Matrix *new_matrix(int rows, int cols) {
  Matrix *m = malloc(sizeof(Matrix));
  if (m == NULL) {
    fprintf(stderr, "Error: malloc failed\n");
    exit(1);
  }

  m->elements = malloc(rows * cols * sizeof(double));
  m->gradients = malloc(rows * cols * sizeof(double));
  m->rows = rows;
  m->cols = cols;
  m->inputs = NULL;
  m->num_inputs = 0;
  m->backward = backward_none;
  m->zero_grad = zero_grad_matrix;
  return m;
}

Matrix *new_matrix_from_file(char *filename, int rows, int cols) {
  double *data = malloc(rows * cols * sizeof(double));
  load_csv(filename, data, rows, cols);
  Matrix *m = new_matrix(rows, cols);
  init_matrix_from_array(m, data, rows, cols);
  free(data);
  return m;
}

void free_matrix(Matrix *m) {
  free(m->elements);
  free(m->gradients);
  for (int i = 0; i < m->num_inputs; i++) {
    free_matrix(m->inputs[i]);
  }
  free(m->inputs);
  free(m);
}

static double rand_uniform(double low, double high) {
  return low + (high - low) * rand() / RAND_MAX;
}

void init_matrix_uniform_random(Matrix *m, double low, double high) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->elements[i * m->cols + j] = rand_uniform(low, high);
    }
  }
}

static double rand_normal(double mean, double std) {
  double u = rand() / (RAND_MAX + 1.0);
  double v = rand() / (RAND_MAX + 1.0);
  double z = sqrt(-2 * log(u)) * cos(2 * M_PI * v);
  return mean + std * z;
}

void init_matrix_normal_random(Matrix *m, double mean, double std) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->elements[i * m->cols + j] = rand_normal(mean, std);
    }
  }
}

void init_matrix_from_array(Matrix *m, double *data, int rows, int cols) {
  if (m->rows != rows || m->cols != cols) {
    fprintf(stderr, "Error: size mismatch\n");
    fprintf(stderr, "init_matrix_from_array: %d x %d, %d x %d\n", m->rows,
            m->cols, rows, cols);
    exit(1);
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      m->elements[i * cols + j] = data[i * cols + j];
    }
  }
}

void init_matrix_from_file(Matrix *m, char *filename, int rows, int cols) {
  double *data = malloc(rows * cols * sizeof(double));
  load_csv(filename, data, rows, cols);
  init_matrix_from_array(m, data, rows, cols);
  free(data);
}

static void backward_matrix_add_matrix(Matrix *self) {
  Matrix *a = self->inputs[0];
  Matrix *b = self->inputs[1];

  for (int i = 0; i < self->rows; i++) {
    for (int j = 0; j < self->cols; j++) {
      a->gradients[i * a->cols + j] += self->gradients[i * self->cols + j];
      b->gradients[i * b->cols + j] += self->gradients[i * self->cols + j];
    }
  }
}

void matrix_add_matrix(Matrix *result, Matrix *a, Matrix *b) {
  if (result->rows != a->rows || result->rows != b->rows ||
      result->cols != a->cols || result->cols != b->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    fprintf(stderr, "matrix_add_matrix: %d x %d, %d x %d, %d x %d\n",
            result->rows, result->cols, a->rows, a->cols, b->rows, b->cols);
    exit(1);
  }

  for (int i = 0; i < result->rows; i++) {
    for (int j = 0; j < result->cols; j++) {
      result->elements[i * result->cols + j] =
          a->elements[i * a->cols + j] + b->elements[i * b->cols + j];
    }
  }

  result->inputs = malloc(2 * sizeof(Matrix *));
  result->inputs[0] = a;
  result->inputs[1] = b;
  result->num_inputs = 2;
  result->backward = backward_matrix_add_matrix;
}

static void backward_matrix_mul_matrix(Matrix *self) {
  Matrix *a = self->inputs[0];
  Matrix *b = self->inputs[1];

  for (int i = 0; i < self->rows; i++) {
    for (int j = 0; j < self->cols; j++) {
      for (int k = 0; k < a->cols; k++) {
        a->gradients[i * a->cols + k] +=
            self->gradients[i * self->cols + j] * b->elements[k * b->cols + j];
        b->gradients[k * b->cols + j] +=
            self->gradients[i * self->cols + j] * a->elements[i * a->cols + k];
      }
    }
  }
}

void matrix_mul_matrix(Matrix *result, Matrix *a, Matrix *b) {
  if (result->rows != a->rows || result->cols != b->cols ||
      a->cols != b->rows) {
    fprintf(stderr, "Error: size mismatch\n");
    fprintf(stderr, "matrix_mul_matrix: %d x %d, %d x %d, %d x %d\n",
            result->rows, result->cols, a->rows, a->cols, b->rows, b->cols);
    exit(1);
  }

  for (int i = 0; i < result->rows; i++) {
    for (int j = 0; j < result->cols; j++) {
      double sum = 0;
      for (int k = 0; k < a->cols; k++) {
        sum += a->elements[i * a->cols + k] * b->elements[k * b->cols + j];
      }
      result->elements[i * result->cols + j] = sum;
    }
  }

  result->inputs = malloc(2 * sizeof(Matrix *));
  result->inputs[0] = a;
  result->inputs[1] = b;
  result->num_inputs = 2;
  result->backward = backward_matrix_mul_matrix;
}

static void backward_matrix_add_vector(Matrix *self) {
  Matrix *m = self->inputs[0];
  Matrix *v = self->inputs[1];

  for (int i = 0; i < self->rows; i++) {
    for (int j = 0; j < self->cols; j++) {
      m->gradients[i * m->cols + j] += self->gradients[i * self->cols + j];
      v->gradients[j] += self->gradients[i * self->cols + j];
    }
  }
}

void matrix_add_vector(Matrix *result, Matrix *m, Matrix *v) {
  /* result = m + v
   * m: (batch_size, input_size)
   * v: (1, input_size)
   */

  if (result->rows != m->rows || result->cols != m->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    fprintf(stderr, "matrix_add_vector: %d x %d, %d x %d, %d x %d\n",
            result->rows, result->cols, m->rows, m->cols, v->rows, v->cols);
    exit(1);
  }

  if (v->rows != 1 || v->cols != m->cols) {
    fprintf(stderr, "Error: unsupported size\n");
    fprintf(stderr, "matrix_add_vector: %d x %d, %d x %d, %d x %d\n",
            result->rows, result->cols, m->rows, m->cols, v->rows, v->cols);
    exit(1);
  }

  for (int i = 0; i < result->rows; i++) {
    for (int j = 0; j < result->cols; j++) {
      result->elements[i * result->cols + j] =
          m->elements[i * m->cols + j] + v->elements[j];
    }
  }

  result->inputs = malloc(2 * sizeof(Matrix *));
  result->inputs[0] = m;
  result->inputs[1] = v;
  result->num_inputs = 2;
  result->backward = backward_matrix_add_vector;
}

static void backward_sigmoid_matrix(Matrix *self) {
  Matrix *m = self->inputs[0];

  for (int i = 0; i < self->rows; i++) {
    for (int j = 0; j < self->cols; j++) {
      int idx = i * self->cols + j;
      m->gradients[idx] += self->gradients[idx] * self->elements[idx] *
                           (1 - self->elements[idx]);
    }
  }
}

void sigmoid_matrix(Matrix *result, Matrix *m) {
  if (result->rows != m->rows || result->cols != m->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    fprintf(stderr, "sigmoid_matrix: %d x %d, %d x %d\n", result->rows,
            result->cols, m->rows, m->cols);
    exit(1);
  }

  for (int i = 0; i < result->rows; i++) {
    for (int j = 0; j < result->cols; j++) {
      result->elements[i * result->cols + j] =
          1 / (1 + exp(-m->elements[i * m->cols + j]));
    }
  }

  result->inputs = malloc(sizeof(Matrix *));
  result->inputs[0] = m;
  result->num_inputs = 1;
  result->backward = backward_sigmoid_matrix;
}

void softmax_matrix(Matrix *result, Matrix *m) {
  if (result->rows != m->rows || result->cols != m->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    fprintf(stderr, "softmax_matrix: %d x %d, %d x %d\n", result->rows,
            result->cols, m->rows, m->cols);
    exit(1);
  }

  for (int i = 0; i < result->rows; i++) {
    double max = m->elements[i * m->cols];
    for (int j = 0; j < result->cols; j++) {
      if (m->elements[i * m->cols + j] > max) {
        max = m->elements[i * m->cols + j];
      }
    }

    double sum = 0;
    for (int j = 0; j < result->cols; j++) {
      result->elements[i * result->cols + j] =
          exp(m->elements[i * m->cols + j] - max);
      sum += result->elements[i * result->cols + j];
    }

    for (int j = 0; j < result->cols; j++) {
      result->elements[i * result->cols + j] /= sum;
    }
  }
}

double cross_entropy_loss(Matrix *y_true, Matrix *y_pred) {
  /* y_true is a one-hot encoded matrix
   * y_pred is a logits matrix (before softmax)
   */
  if (y_true->rows != y_pred->rows || y_true->cols != y_pred->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    fprintf(stderr, "cross_entropy_loss: %d x %d, %d x %d\n", y_true->rows,
            y_true->cols, y_pred->rows, y_pred->cols);
    exit(1);
  }

  Matrix *y_pred_softmax = new_matrix(y_pred->rows, y_pred->cols);
  softmax_matrix(y_pred_softmax, y_pred);

  double loss = 0;
  double epsilon = 1e-15;
  for (int i = 0; i < y_true->rows; i++) {
    for (int j = 0; j < y_true->cols; j++) {
      if (y_true->elements[i * y_true->cols + j] > 0) {
        double prob =
            y_pred_softmax->elements[i * y_pred_softmax->cols + j] + epsilon;
        loss += y_true->elements[i * y_true->cols + j] * log(prob);
      }
    }
  }

  free_matrix(y_pred_softmax);
  return -loss / y_true->rows;
}

void print_matrix(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      printf("%f ", m->elements[i * m->cols + j]);
    }
    printf("\n");
  }
}
