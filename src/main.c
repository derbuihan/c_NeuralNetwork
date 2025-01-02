#include "matrix.h"
#include "mnist.h"
#include "simple_network.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double calculate_accuracy(Matrix *y_true, Matrix *y_pred) {
  /* y_true is one-hot encoded labels
   * y_pred is logits matrix
   */
  if (y_true->rows != y_pred->rows || y_true->cols != y_pred->cols) {
    fprintf(stderr, "Error: size mismatch\n");
    exit(1);
  }

  int correct = 0;
  for (int i = 0; i < y_pred->rows; i++) {
    int pred = 0;
    double max = y_pred->elements[i * y_pred->cols];
    for (int j = 0; j < y_pred->cols; j++) {
      if (y_pred->elements[i * y_pred->cols + j] > max) {
        max = y_pred->elements[i * y_pred->cols + j];
        pred = j;
      }
    }

    int label = 0;
    max = y_true->elements[i * y_true->cols];
    for (int j = 0; j < y_true->cols; j++) {
      if (y_true->elements[i * y_true->cols + j] > max) {
        max = y_true->elements[i * y_true->cols + j];
        label = j;
      }
    }

    if (pred == label)
      correct++;
  }
  return (double)correct / y_true->rows;
}

void update_weights(Matrix *m, Matrix *grad_m, double learning_rate) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->elements[i * m->cols + j] -=
          learning_rate * grad_m->elements[i * m->cols + j];
    }
  }
}

void load_mini_batch(Matrix *X, Matrix *y_true, Matrix *X_batch,
                     Matrix *y_true_batch, int batch_size) {
  int indices[batch_size];

  int count = 0;
  while (count < batch_size) {
    int index = rand() % (X->rows);
    int found = 0;
    for (int i = 0; i < count; i++) {
      if (indices[i] == index) {
        found = 1;
        break;
      }
    }
    if (!found) {
      indices[count] = index;
      count++;
    }
  }

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < X->cols; j++) {
      X_batch->elements[i * X->cols + j] =
          X->elements[indices[i] * X->cols + j];
    }
    for (int j = 0; j < y_true->cols; j++) {
      y_true_batch->elements[i * y_true->cols + j] =
          y_true->elements[indices[i] * y_true->cols + j];
    }
  }
}

int main(void) {
  printf("Hello, World!\n");

  // Load datasets
  double *data = malloc(ROWS * COLS * sizeof(double));
  double *labels = malloc(ROWS * sizeof(double) * 10);
  load_mnist_datasets("../datasets/mnist_test.csv", data, labels);

  Matrix *X = new_matrix(ROWS, COLS);
  init_matrix_from_array(X, data, ROWS, COLS);
  free(data);

  Matrix *y_true = new_matrix(ROWS, 10); // Assuming y_true is one-hot encoded
  init_matrix_from_array(y_true, labels, ROWS, 10);
  free(labels);

  // Mini batch
  int batch_size = 512;
  Matrix *X_batch = new_matrix(batch_size, COLS);
  Matrix *y_true_batch = new_matrix(batch_size, 10);

  // Initialize network
  Network *net = new_network();

  // Train network
  int epochs = 1000;
  for (int i = 1; i <= epochs; i++) {
    // Load mini batch
    load_mini_batch(X, y_true, X_batch, y_true_batch, batch_size);

    // Forward pass
    Matrix *y_pred_batch = net->forward(net, X_batch);
    double accrary = calculate_accuracy(y_true_batch, y_pred_batch);
    double loss = cross_entropy_loss(y_true_batch, y_pred_batch);
    printf("Epoch %d: Loss: %f, Accuracy: %.2f%%\n", i, loss,
           (double)accrary * 100);
    free_matrix(y_pred_batch);

    // Backward pass
    net->backward(net, X_batch, y_true_batch);
    double learning_rate = 0.0001;

    update_weights(net->W1, net->grad_W1, learning_rate);
    update_weights(net->W2, net->grad_W2, learning_rate);
    update_weights(net->W3, net->grad_W3, learning_rate);
    update_weights(net->b1, net->grad_b1, learning_rate);
    update_weights(net->b2, net->grad_b2, learning_rate);
    update_weights(net->b3, net->grad_b3, learning_rate);
  }

  return 0;
}
