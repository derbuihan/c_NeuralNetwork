#include "loss.h"

#include <math.h>
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

double forward_cross_entropy_loss(Loss *loss_fn, Matrix *y_true,
                                  Matrix *y_pred) {
  /* y_true is a one-hot encoded matrix
   * y_pred is a logits matrix (before softmax)
   */
  if (y_true->rows != y_pred->rows || y_true->cols != y_pred->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    fprintf(stderr, "cross_entropy_loss: %d x %d, %d x %d\n", y_true->rows,
            y_true->cols, y_pred->rows, y_pred->cols);
    exit(1);
  }

  Matrix *y_pred_softmax = loss_fn->params[0];
  softmax_matrix(y_pred_softmax, y_pred);

  double loss = 0;
  double epsilon = 1e-15;
  for (int i = 0; i < y_true->rows; i++) {
    for (int j = 0; j < y_true->cols; j++) {
      int idx = i * y_true->cols + j;
      if (y_true->elements[idx] > 0) {
        double prob = y_pred_softmax->elements[idx] + epsilon;
        loss += y_true->elements[idx] * log(prob);
      }
    }
  }

  return -loss / y_true->rows;
}

static void backward_none(Loss *loss_fn) {
  // Do nothing
  Matrix *y_pred_softmax = loss_fn->params[0];

  loss_fn->params[0]->backward(loss_fn->params[0]);
}

Loss *new_cross_entropy_loss(int class_num) {
  Loss *loss = malloc(sizeof(Loss));

  // params
  loss->params = malloc(1 * sizeof(Matrix *));
  loss->params[0] = new_matrix(BATCH_SIZE, class_num);
  loss->num_params = 1;

  // options
  loss->options = malloc(1 * sizeof(int));
  loss->options[0] = class_num;

  // functions
  loss->forward = forward_cross_entropy_loss;
  loss->backward = backward_none;
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