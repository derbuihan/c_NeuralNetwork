#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"

typedef struct SGD_Optimizer SGD_Optimizer;
struct SGD_Optimizer {
  Matrix *m;
  Matrix *grad_m;
  double learning_rate;
  void (*update)(SGD_Optimizer *optimizer);
};

SGD_Optimizer *new_sgd_optimizer(Matrix *m, Matrix *grad_m,
                                 double learning_rate);

typedef struct Adam_Optimizer Adam_Optimizer;
struct Adam_Optimizer {
  Matrix *m;
  Matrix *grad_m;
  Matrix *moment1;
  Matrix *moment2;
  double beta1;
  double beta2;
  double epsilon;
  double learning_rate;
  int t;
  void (*update)(Adam_Optimizer *optimizer);
};

Adam_Optimizer *new_adam_optimizer(Matrix *m, Matrix *grad_m,
                                   double learning_rate);

#endif // OPTIMIZER_H
