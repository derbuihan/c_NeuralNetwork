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

void train(Network *net, Matrix *X_train, Matrix *y_train, Matrix *X_test,
           Matrix *y_test, int epochs, int batch_size, double learning_rate) {
  Matrix *X_batch = new_matrix(batch_size, X_train->cols);
  Matrix *y_true_batch = new_matrix(batch_size, 10);

  Adam_Optimizer *optim_W1 =
      new_adam_optimizer(net->W1, net->grad_W1, learning_rate);
  Adam_Optimizer *optim_b1 =
      new_adam_optimizer(net->b1, net->grad_b1, learning_rate);
  Adam_Optimizer *optim_W2 =
      new_adam_optimizer(net->W2, net->grad_W2, learning_rate);
  Adam_Optimizer *optim_b2 =
      new_adam_optimizer(net->b2, net->grad_b2, learning_rate);
  Adam_Optimizer *optim_W3 =
      new_adam_optimizer(net->W3, net->grad_W3, learning_rate);
  Adam_Optimizer *optim_b3 =
      new_adam_optimizer(net->b3, net->grad_b3, learning_rate);

  for (int i = 1; i <= epochs; i++) {
    load_mini_batch(X_train, y_train, X_batch, y_true_batch, batch_size);

    if (i % 100 == 0) {
      printf("Epoch %d: ", i);
      validate(net, X_test, y_test);
    }

    net->backward(net, X_batch, y_true_batch);

    optim_W1->update(optim_W1);
    optim_b1->update(optim_b1);
    optim_W2->update(optim_W2);
    optim_b2->update(optim_b2);
    optim_W3->update(optim_W3);
    optim_b3->update(optim_b3);
  }

  free_matrix(X_batch);
  free_matrix(y_true_batch);
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

  // init_matrix_from_file(net->W1, "../datasets/W1.csv", 784, 50);
  // init_matrix_from_file(net->W2, "../datasets/W2.csv", 50, 100);
  // init_matrix_from_file(net->W3, "../datasets/W3.csv", 100, 10);
  //
  // Matrix *b1_ = new_matrix(50, 1);
  // init_matrix_from_file(b1_, "../datasets/b1.csv", 50, 1);
  // transpose_matrix(net->b1, b1_);
  //
  // Matrix *b2_ = new_matrix(100, 1);
  // init_matrix_from_file(b2_, "../datasets/b2.csv", 100, 1);
  // transpose_matrix(net->b2, b2_);
  //
  // Matrix *b3_ = new_matrix(10, 1);
  // init_matrix_from_file(b3_, "../datasets/b3.csv", 10, 1);
  // transpose_matrix(net->b3, b3_);
  //
  // validate(net, X_test, y_test);

  train(net, X_train, y_train, X_test, y_test, 10000, 32, 0.001);

  return 0;
}
