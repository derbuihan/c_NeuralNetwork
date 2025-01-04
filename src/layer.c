#include "layer.h"

#include <math.h>
#include <stdlib.h>

void free_layer(Layer *layer) {
  for (int i = 0; i < layer->num_params; i++) {
    free_matrix(layer->params[i]);
  }
  free(layer->params);
  free(layer);
}

void backward_layer(Layer *layer) {
  int num_params = layer->num_params;
  Matrix **params = layer->params;
  for (int i = num_params - 1; i >= 0; i--) {
    params[i]->backward(params[i]);
  }
}

void zero_grad_layer(Layer *layer) {
  int num_params = layer->num_params;
  Matrix **params = layer->params;
  for (int i = 0; i < num_params; i++) {
    Matrix *m = params[i];
    m->zero_grad(m);
  }
}

/* Linear Layer */

static Matrix *forward_linear_layer(Layer *layer, Matrix *X) {
  /* t1 = X * W (batch_size, output_size)
   * a1 = t1 + b (batch_size, output_size)
   */
  Matrix *W = layer->params[0];
  Matrix *t1 = layer->params[1];
  Matrix *b = layer->params[2];
  Matrix *a1 = layer->params[3];

  matrix_mul_matrix(t1, X, W);
  matrix_add_vector(a1, t1, b);

  return a1;
}

Layer *new_linear_layer(int batch_size, int input_size, int output_size) {
  /* X: inputs (batch_size, input_size)
   * W: weights (input_size, output_size)
   * t1 = X * W (batch_size, output_size)
   * b: bias (1, output_size)
   * a1 = t1 + b (batch_size, output_size)
   * a1: outputs (batch_size, output_size)
   * params = [W, t1, b, a1]
   */

  Layer *layer = malloc(sizeof(Layer));

  // params
  layer->params = malloc(4 * sizeof(Matrix *));
  layer->params[0] = new_matrix(input_size, output_size); // W
  layer->params[1] = new_matrix(batch_size, output_size); // t1
  layer->params[2] = new_matrix(1, output_size);          // b
  layer->params[3] = new_matrix(batch_size, output_size); // a1
  layer->num_params = 4;
  init_matrix_uniform_random(layer->params[0], -1.0 / sqrt(input_size),
                             1.0 / sqrt(input_size));
  init_matrix_uniform_random(layer->params[2], -1.0 / sqrt(input_size),
                             1.0 / sqrt(input_size));

  // functions
  layer->forward = forward_linear_layer;
  layer->backward = backward_layer;
  layer->zero_grad = zero_grad_layer;
  return layer;
}

/* Sigmoid Layer */

static Matrix *forward_sigmoid_layer(Layer *layer, Matrix *X) {
  /* Z = sigmoid(X) (batch_size, input_size)
   */
  Matrix *Z = layer->params[0];
  sigmoid_matrix(Z, X);
  return Z;
}

Layer *new_sigmoid_layer(int batch_size, int input_size) {
  /* X: inputs (batch_size, input_size)
   * Z = sigmoid(X) (batch_size, input_size)
   * Z: outputs (batch_size, input_size)
   * params = [Z]
   */
  Layer *layer = malloc(sizeof(Layer));

  // params
  layer->params = malloc(1 * sizeof(Matrix *));
  layer->params[0] = new_matrix(batch_size, input_size); // Z
  layer->num_params = 1;

  // functions
  layer->forward = forward_sigmoid_layer;
  layer->backward = backward_layer;
  layer->zero_grad = zero_grad_layer;
  return layer;
}
