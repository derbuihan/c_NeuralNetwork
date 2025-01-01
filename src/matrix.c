#include "matrix.h"

#include "mnist.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Matrix *new_matrix(int rows, int cols) {
  Matrix *m = malloc(sizeof(Matrix));

  m->elements = malloc(rows * cols * sizeof(double));
  m->rows = rows;
  m->cols = cols;

  return m;
}

Matrix *new_matrix_from_file(const char *filename, int rows, int cols) {
  double *data = malloc(rows * cols * sizeof(double));
  load_csv(filename, data, rows, cols);
  Matrix *m = new_matrix(rows, cols);
  init_matrix_from_array(m, data, rows, cols);
  free(data);
  return m;
}

void free_matrix(Matrix *m) {
  free(m->elements);
  free(m);
}

double rand_normal(double mean, double stddev) {
  double u = rand() / (RAND_MAX + 1.0);
  double v = rand() / (RAND_MAX + 1.0);
  double z = sqrt(-2 * log(u)) * cos(2 * M_PI * v);
  return mean + stddev * z;
}

void init_matrix_random(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->elements[i * m->cols + j] = rand_normal(0, 1);
    }
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

void matrix_add_matrixt(Matrix *result, const Matrix *a, const Matrix *b) {
  if (result->rows != a->rows || result->rows != b->rows ||
      result->cols != a->cols || result->cols != b->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    exit(1);
  }

  for (int i = 0; i < result->rows; i++) {
    for (int j = 0; j < result->cols; j++) {
      result->elements[i * result->cols + j] =
          a->elements[i * a->cols + j] + b->elements[i * b->cols + j];
    }
  }
}

void matrix_multiply_matrix(Matrix *result, const Matrix *a, const Matrix *b) {
  if (result->rows != a->rows || result->cols != b->cols ||
      a->cols != b->rows) {
    fprintf(stderr, "Error: size mismatch\n");
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
}

void matrix_add_vector(Matrix *result, const Matrix *m, const Matrix *v) {
  if (result->rows != m->rows || result->cols != m->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    exit(1);
  }

  if (v->cols == 1) {
    if (v->rows != m->rows) {
      fprintf(stderr, "Error: size mismatch\n");
      exit(1);
    }

    for (int i = 0; i < result->rows; i++) {
      for (int j = 0; j < result->cols; j++) {
        result->elements[i * result->cols + j] =
            m->elements[i * m->cols + j] + v->elements[i];
      }
    }
  } else if (v->rows == 1) {
    if (v->cols != m->cols) {
      fprintf(stderr, "Error: size mismatch\n");
      exit(1);
    }

    for (int i = 0; i < result->rows; i++) {
      for (int j = 0; j < result->cols; j++) {
        result->elements[i * result->cols + j] =
            m->elements[i * m->cols + j] + v->elements[j];
      }
    }
  }
}

void sigmoid_matrix(Matrix *result, const Matrix *m) {
  if (result->rows != m->rows || result->cols != m->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    exit(1);
  }

  for (int i = 0; i < result->rows; i++) {
    for (int j = 0; j < result->cols; j++) {
      result->elements[i * result->cols + j] =
          1 / (1 + exp(-m->elements[i * m->cols + j]));
    }
  }
}

void softmax_matrix(Matrix *result, const Matrix *m) {
  if (result->rows != m->rows || result->cols != m->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    exit(1);
  }

  for (int i = 0; i < result->rows; i++) {
    double max = m->elements[i * m->cols];
    for (int j = 1; j < result->cols; j++) {
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

void transpose_matrix(Matrix *result, const Matrix *m) {
  if (result->rows != m->cols || result->cols != m->rows) {
    fprintf(stderr, "Error: size mismatch\n");
    exit(1);
  }

  for (int i = 0; i < result->rows; i++) {
    for (int j = 0; j < result->cols; j++) {
      result->elements[i * result->cols + j] = m->elements[j * m->cols + i];
    }
  }
}

void print_matrix(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      printf("%f ", m->elements[i * m->cols + j]);
    }
    printf("\n");
  }
}

double sum_elements_matrix(const Matrix *m) {
  double sum = 0;
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      sum += m->elements[i * m->cols + j];
    }
  }
  return sum;
}
