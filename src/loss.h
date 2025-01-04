#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

typedef struct Loss Loss;

struct Loss {
  Matrix **params;
  int num_params;

  int *options;
  int num_options;

  double (*forward)(Loss *self, Matrix *y_true, Matrix *y_pred);
  void (*backward)(Loss *self);
};

void free_loss(Loss *loss_fn);
Loss *new_cross_entropy_loss(int batch_size, int class_num);

#endif // LOSS_H
