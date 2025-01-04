#include "loss.h"
#include "matrix.h"
#include "mnist.h"
#include "optimizer.h"
#include "simple_network.h"
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

void validate(Network *net, Matrix *X_test, Matrix *y_test) {
  Matrix *y_pred = net->forward(net, X_test);
  double accuracy = calculate_accuracy(y_test, y_pred);
  double loss = cross_entropy_loss(y_test, y_pred);
  printf("Validation: Loss: %f, Accuracy: %.2f%%\n", loss,
         (double)accuracy * 100);
  free_matrix(y_pred);
}

int main(void) {
  printf("Hello, World!\n");

  // Load train dataset
  int train_rows = 60000;
  int train_cols = 28 * 28;
  double *train_data = malloc(train_rows * train_cols * sizeof(double));
  double *train_labels = malloc(train_rows * sizeof(double) * 10);
  load_mnist_datasets("../datasets/mnist_train.csv", train_data, train_labels,
                      train_rows, train_cols);

  Matrix *X_train = new_matrix(train_rows, train_cols);
  init_matrix_from_array(X_train, train_data, train_rows, train_cols);
  free(train_data);

  Matrix *y_train = new_matrix(train_rows, 10); // y_train is one-hot encoded
  init_matrix_from_array(y_train, train_labels, train_rows, 10);
  free(train_labels);

  // Load test dataset
  int test_rows = 10000;
  int test_cols = 28 * 28;
  double *test_data = malloc(test_rows * test_cols * sizeof(double));
  double *test_labels = malloc(test_rows * sizeof(double) * 10);
  load_mnist_datasets("../datasets/mnist_test.csv", test_data, test_labels,
                      test_rows, test_cols);

  Matrix *X_test = new_matrix(test_rows, test_cols);
  init_matrix_from_array(X_test, test_data, test_rows, test_cols);
  free(test_data);

  Matrix *y_test = new_matrix(test_rows, 10); // y_test is one-hot encoded
  init_matrix_from_array(y_test, test_labels, test_rows, 10);
  free(test_labels);

  // Initialize network
  Network *net = new_network();

  SGD_Optimizer *optim = new_sgd_optimizer(net, 0.001);
  Loss *loss_fn = new_cross_entropy_loss(10);

  // Train loop
  for (int i = 1; i <= 10; i++) {
    // Load mini-batch
    Matrix *X_batch = new_matrix(BATCH_SIZE, X_train->cols);
    Matrix *y_true_batch = new_matrix(BATCH_SIZE, 10);
    load_mini_batch(X_train, y_train, X_batch, y_true_batch, BATCH_SIZE);

    // Forward pass
    optim->zero_grad(optim);
    Matrix *y_pred = net->forward(net, X_batch);
    double loss = loss_fn->forward(loss_fn, y_true_batch, y_pred);

    // Backward pass
    loss_fn->backward(loss_fn);
    net->backward(net);
    optim->step(optim);

    // Free memory
    free_matrix(X_batch);
    free_matrix(y_true_batch);

    // Print loss
    printf("Epoch %d: Loss: %f\n", i, loss);
  }

  free_network(net);
  free_optimizer(optim);
  free_loss(loss_fn);

  return 0;
}
