#include "optimizer.h"

#include <math.h>
#include <stdlib.h>

static void update_sdg(SGD_Optimizer *optimizer) {
  for (int i = 0; i < optimizer->m->rows; i++) {
    for (int j = 0; j < optimizer->m->cols; j++) {
      optimizer->m->elements[i * optimizer->m->cols + j] -=
          optimizer->learning_rate *
          optimizer->grad_m->elements[i * optimizer->m->cols + j];
    }
  }
}

SGD_Optimizer *new_sgd_optimizer(Matrix *m, Matrix *grad_m,
                                 double learning_rate) {
  SGD_Optimizer *optimizer = malloc(sizeof(SGD_Optimizer));
  optimizer->m = m;
  optimizer->grad_m = grad_m;
  optimizer->learning_rate = learning_rate;
  optimizer->update = update_sdg;
  return optimizer;
}

static void update_adam(Adam_Optimizer *optimizer) {
  optimizer->t++;
  Matrix *m = optimizer->m;
  Matrix *grad_m = optimizer->grad_m;
  Matrix *moment1 = optimizer->moment1;
  Matrix *moment2 = optimizer->moment2;
  double beta1_t = optimizer->beta1;
  double beta2_t = optimizer->beta2;
  double epsilon = optimizer->epsilon;
  double learning_rate = optimizer->learning_rate;

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      int idx = i * m->cols + j;
      moment1->elements[idx] = beta1_t * moment1->elements[idx] +
                               (1 - beta1_t) * grad_m->elements[idx];
      moment2->elements[idx] =
          beta2_t * moment2->elements[idx] +
          (1 - beta2_t) * grad_m->elements[idx] * grad_m->elements[idx];
      double m_hat = moment1->elements[idx] / (1 - beta1_t);
      double v_hat = moment2->elements[idx] / (1 - beta2_t);
      m->elements[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
  }

  optimizer->beta1 *= optimizer->beta1;
  optimizer->beta2 *= optimizer->beta2;
}

Adam_Optimizer *new_adam_optimizer(Matrix *m, Matrix *grad_m,
                                   double learning_rate) {
  Adam_Optimizer *optimizer = malloc(sizeof(Adam_Optimizer));
  optimizer->m = m;
  optimizer->grad_m = grad_m;
  optimizer->moment1 = new_matrix(m->rows, m->cols);
  optimizer->moment2 = new_matrix(m->rows, m->cols);
  optimizer->beta1 = 0.9;
  optimizer->beta2 = 0.999;
  optimizer->epsilon = 1e-8;
  optimizer->learning_rate = learning_rate;
  optimizer->t = 0;
  optimizer->update = update_adam;
  return optimizer;
}
