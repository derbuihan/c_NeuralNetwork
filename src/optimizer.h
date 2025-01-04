#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "simple_network.h"

typedef struct SGD_Optimizer SGD_Optimizer;
struct SGD_Optimizer {
  Network *net;
  double learning_rate;
  void (*zero_grad)(SGD_Optimizer *optimizer);
  void (*step)(SGD_Optimizer *optimizer);
};

void free_sgd_optimizer(SGD_Optimizer *optim);
SGD_Optimizer *new_sgd_optimizer(Network *net, double learning_rate);

typedef struct Adam_Optimizer Adam_Optimizer;
struct Adam_Optimizer {
  Network *net;
  double learning_rate;

  // Adam parameters
  double beta1;
  double beta2;
  double epsilon;
  int t;
  double **moment1; // (net->num_layers, m->rows * m->cols)
  double **moment2; // (net->num_layers, m->rows * m->cols)

  void (*zero_grad)(Adam_Optimizer *optimizer);
  void (*step)(Adam_Optimizer *optimizer);
};

void free_adam_optimizer(Adam_Optimizer *optim);
Adam_Optimizer *new_adam_optimizer(Network *net, double learning_rate);

#endif // OPTIMIZER_H
