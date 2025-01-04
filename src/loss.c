#include "loss.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void free_loss(Loss *loss_fn) {
  for (int i = 0; i < loss_fn->num_params; i++) {
    // free_matrix(loss_fn->params[i]);
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

  loss_fn->params[0] = y_pred;
  softmax_matrix(loss_fn->params[1], y_pred);
  loss_fn->params[2] = y_true;

  return cross_entropy_loss(y_true, y_pred);
}

static void backward_cross_entropy_loss(Loss *loss_fn) {
  /* dL/dy_pred = y_pred_softmax - y_true
   */
  Matrix *y_pred = loss_fn->params[0];
  Matrix *y_pred_softmax = loss_fn->params[1];
  Matrix *y_true = loss_fn->params[2];

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
  loss->params =
      malloc(3 * sizeof(Matrix *)); // 0: y_pred, 1: y_pred_softmax 2: y_true
  loss->params[1] = new_matrix(batch_size, class_num); // y_pred_softmax
  loss->num_params = 3;

  // options
  loss->options = malloc(1 * sizeof(int));
  loss->options[0] = class_num;

  // functions
  loss->forward = forward_cross_entropy_loss;
  loss->backward = backward_cross_entropy_loss;
  return loss;
}

// Loss *loss_fn = new_cross_entropy_loss();
// double loss = loss_fn->forward(loss_fn, y_true_batch, y_pred);
// loss_fn->backward(loss_fn);
//
// double cross_entropy_loss(const Matrix *y_true, const Matrix *y_pred) {
//
//   Matrix *y_pred_softmax = new_matrix(y_pred->rows, y_pred->cols);
//   softmax_matrix(y_pred_softmax, y_pred);
//
//   double loss = 0;
//   double epsilon = 1e-15;
//   for (int i = 0; i < y_true->rows; i++) {
//     for (int j = 0; j < y_true->cols; j++) {
//       if (y_true->elements[i * y_true->cols + j] > 0) {
//         double prob =
//             y_pred_softmax->elements[i * y_pred_softmax->cols + j] + epsilon;
//         loss += y_true->elements[i * y_true->cols + j] * log(prob);
//       }
//     }
//   }
//
//   free_matrix(y_pred_softmax);
//   return -loss / y_true->rows;
// }