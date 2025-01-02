#include "matrix.h"
#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct Network Network;
struct Network {
  Matrix *W1;
  Matrix *W2;
  Matrix *W3;
  Matrix *b1;
  Matrix *b2;
  Matrix *b3;

  Matrix *grad_W1;
  Matrix *grad_W2;
  Matrix *grad_W3;
  Matrix *grad_b1;
  Matrix *grad_b2;
  Matrix *grad_b3;

  void (*init)(Network *net);
  void (*free)(Network *net);
  Matrix *(*forward)(Network *net, Matrix *X);
  void (*backward)(Network *net, Matrix *X, Matrix *y_true);
};

void init_network(Network *net) {
  net->W1 = new_matrix(COLS, 50);
  net->W2 = new_matrix(50, 100);
  net->W3 = new_matrix(100, 10);

  net->b1 = new_matrix(1, 50);
  net->b2 = new_matrix(1, 100);
  net->b3 = new_matrix(1, 10);

  net->grad_W1 = new_matrix(COLS, 50);
  net->grad_W2 = new_matrix(50, 100);
  net->grad_W3 = new_matrix(100, 10);

  net->grad_b1 = new_matrix(1, 50);
  net->grad_b2 = new_matrix(1, 100);
  net->grad_b3 = new_matrix(1, 10);
}

void free_network(Network *net) {
  free_matrix(net->W1);
  free_matrix(net->W2);
  free_matrix(net->W3);
  free_matrix(net->b1);
  free_matrix(net->b2);
  free_matrix(net->b3);
  free(net);
}

Matrix *forward(Network *net, Matrix *X) {
  Matrix *t1 = new_matrix(X->rows, 50);
  matrix_mul_matrix(t1, X, net->W1);

  Matrix *a1 = new_matrix(X->rows, 50);
  matrix_add_vector(a1, t1, net->b1);

  Matrix *z1 = new_matrix(X->rows, 50);
  sigmoid_matrix(z1, a1);

  Matrix *t2 = new_matrix(X->rows, 100);
  matrix_mul_matrix(t2, z1, net->W2);

  Matrix *a2 = new_matrix(X->rows, 100);
  matrix_add_vector(a2, t2, net->b2);

  Matrix *z2 = new_matrix(X->rows, 100);
  sigmoid_matrix(z2, a2);

  Matrix *t3 = new_matrix(X->rows, 10);
  matrix_mul_matrix(t3, z2, net->W3);

  Matrix *a3 = new_matrix(X->rows, 10);
  matrix_add_vector(a3, t3, net->b3);

  // Matrix *y = new_matrix(X->rows, 10);
  // softmax_matrix(y, a3);

  free_matrix(t1);
  free_matrix(a1);
  free_matrix(z1);
  free_matrix(t2);
  free_matrix(a2);
  free_matrix(z2);
  free_matrix(t3);
  // free_matrix(a3);

  return a3;
}

void backward(Network *net, Matrix *X, Matrix *y_true) {
  // Forward pass
  Matrix *t1 = new_matrix(X->rows, 50);
  matrix_mul_matrix(t1, X, net->W1);
  Matrix *a1 = new_matrix(X->rows, 50);
  matrix_add_vector(a1, t1, net->b1);
  Matrix *z1 = new_matrix(X->rows, 50);
  sigmoid_matrix(z1, a1);

  Matrix *t2 = new_matrix(X->rows, 100);
  matrix_mul_matrix(t2, z1, net->W2);
  Matrix *a2 = new_matrix(X->rows, 100);
  matrix_add_vector(a2, t2, net->b2);
  Matrix *z2 = new_matrix(X->rows, 100);
  sigmoid_matrix(z2, a2);

  Matrix *t3 = new_matrix(X->rows, 10);
  matrix_mul_matrix(t3, z2, net->W3);
  Matrix *a3 = new_matrix(X->rows, 10);
  matrix_add_vector(a3, t3, net->b3);

  // Backward pass
  Matrix *dL_da3 = new_matrix(X->rows, 10);
  matrix_sub_matrix(dL_da3, a3, y_true);

  matrix_transpose_mul_matrix(net->grad_W3, z2, dL_da3);
  matrix_sum_rows(net->grad_b3, dL_da3);

  Matrix *dL_dz2 = new_matrix(X->rows, 100);
  matrix_mul_matrix_transpose(dL_dz2, dL_da3, net->W3);
  Matrix *dL_da2 = new_matrix(X->rows, 100);
  sigmoid_derivative_matrix(dL_da2, a2);
  matrix_elementwise_mul(dL_dz2, dL_dz2, dL_da2);

  matrix_transpose_mul_matrix(net->grad_W2, z1, dL_dz2);
  matrix_sum_rows(net->grad_b2, dL_dz2);

  Matrix *dL_dz1 = new_matrix(X->rows, 50);
  matrix_mul_matrix_transpose(dL_dz1, dL_dz2, net->W2);
  Matrix *dL_da1 = new_matrix(X->rows, 50);
  sigmoid_derivative_matrix(dL_da1, a1);
  matrix_elementwise_mul(dL_dz1, dL_dz1, dL_da1);

  matrix_transpose_mul_matrix(net->grad_W1, X, dL_dz1);
  matrix_sum_rows(net->grad_b1, dL_dz1);

  // Free temporary matrices
  free_matrix(t1);
  free_matrix(a1);
  free_matrix(z1);
  free_matrix(t2);
  free_matrix(a2);
  free_matrix(z2);
  free_matrix(t3);
  free_matrix(a3);
  free_matrix(dL_da3);
  free_matrix(dL_dz2);
  free_matrix(dL_da2);
  free_matrix(dL_dz1);
  free_matrix(dL_da1);
}

Network *new_network() {
  Network *net = malloc(sizeof(Network));
  net->init = init_network;
  net->free = free_network;
  net->forward = forward;
  net->backward = backward;
  net->init(net);
  return net;
}

double calculate_accuracy(Network *net, Matrix *X, Matrix *y_true) {
  /* X is input data
   * y_true is one-hot encoded labels
   */
  Matrix *y_pred = net->forward(net, X);

  int correct = 0;
  for (int i = 0; i < X->rows; i++) {
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

  free_matrix(y_pred);

  return (double)correct / X->rows;
}

void update_weights(Matrix *m, Matrix *grad_m, double learning_rate) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->elements[i * m->cols + j] -=
          learning_rate * grad_m->elements[i * m->cols + j];
    }
  }
}

void create_mini_batch(Matrix *X, Matrix *y_true, Matrix *X_batch,
                       Matrix *y_true_batch, int batch_size) {
  // Randomly select indices
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

  // initialize network
  Network *net = new_network();
  init_matrix_random(net->W1);
  init_matrix_random(net->W2);
  init_matrix_random(net->W3);
  init_matrix_random(net->b1);
  init_matrix_random(net->b2);
  init_matrix_random(net->b3);

  // Train network
  int epochs = 1000;
  int batch_size = 512;
  Matrix *X_batch = new_matrix(batch_size, COLS);
  Matrix *y_true_batch = new_matrix(batch_size, 10);
  for (int i = 1; i <= epochs; i++) {

    // mini-batch
    create_mini_batch(X, y_true, X_batch, y_true_batch, batch_size);

    // Forward pass
    Matrix *y_pred_batch = net->forward(net, X_batch);

    // Calculate loss
    double loss = cross_entropy_loss(y_true_batch, y_pred_batch);
    printf("Epoch %d: Loss: %f\n", i, loss);
    free_matrix(y_pred_batch);

    // calculate accuracy
    double accrary = calculate_accuracy(net, X_batch, y_true_batch);
    printf("Accuracy: %.2f%%\n", (double)accrary * 100);

    // Backward pass
    net->backward(net, X_batch, y_true_batch);

    // Update weights
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
