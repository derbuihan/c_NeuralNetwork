#include "loss.h"

#include <stdio.h>
#include <stdlib.h>

void free_loss(Loss *loss_fn) {
  for (int i = 0; i < loss_fn->num_params; i++) {
    free_matrix(loss_fn->params[i]);
  }
  free(loss_fn->params);
  free(loss_fn->options);
  free(loss_fn);
}

/* Cross Entropy Loss */

static double forward_cross_entropy_loss(Loss *loss_fn, Matrix *y_true,
                                         Matrix *y_pred) {
  /* y_true is a one-hot encoded matrix
   * y_pred is a logits matrix (before softmax)
   */

  loss_fn->options->y_true = y_true;
  loss_fn->options->y_pred = y_pred;

  Matrix *y_pred_softmax = loss_fn->params[0];
  softmax_matrix(y_pred_softmax, y_pred);

  return cross_entropy_loss(y_true, y_pred_softmax);
}

static void backward_cross_entropy_loss(Loss *loss_fn) {
  /* dL/dy_pred = y_pred_softmax - y_true
   */
  Matrix *y_pred = loss_fn->options->y_pred;
  Matrix *y_pred_softmax = loss_fn->params[0];
  Matrix *y_true = loss_fn->options->y_true;

  for (int i = 0; i < y_true->rows; i++) {
    for (int j = 0; j < y_true->cols; j++) {
      int idx = i * y_true->cols + j;
      y_pred->gradients[idx] =
          y_pred_softmax->elements[idx] - y_true->elements[idx];
    }
  }
}

Loss *new_cross_entropy_loss(int batch_size, int class_num) {
  Loss *loss = malloc(sizeof(Loss));

  // params
  loss->params = malloc(sizeof(Matrix *));
  loss->params[0] = new_matrix(batch_size, class_num); // y_pred_softmax
  loss->num_params = 1;

  // options
  loss->options = malloc(sizeof(LossOptions));

  // functions
  loss->forward = forward_cross_entropy_loss;
  loss->backward = backward_cross_entropy_loss;
  return loss;
}
