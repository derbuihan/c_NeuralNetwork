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

void free_optimizer(SGD_Optimizer *optim);
SGD_Optimizer *new_sgd_optimizer(Network *net, double learning_rate);

// typedef struct Adam_Optimizer Adam_Optimizer;
// struct Adam_Optimizer {
//   Matrix *m;
//   Matrix *grad_m;
//   Matrix *moment1;
//   Matrix *moment2;
//   double beta1;
//   double beta2;
//   double epsilon;
//   double learning_rate;
//   int t;
//   void (*update)(Adam_Optimizer *optimizer);
// };
//
// Adam_Optimizer *new_adam_optimizer(Matrix *m, Matrix *grad_m,
//                                    double learning_rate);
//

#endif // OPTIMIZER_H
