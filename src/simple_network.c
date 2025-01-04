#include "simple_network.h"

#include <stdlib.h>

void free_network(Network *net) {
  for (int i = 0; i < net->num_layers; i++) {
    free_layer(net->layers[i]);
  }
  free(net->layers);
  free(net);
}

static Matrix *forward(Network *net, Matrix *X) {
  for (int i = 0; i < net->num_layers; i++) {
    X = net->layers[i]->forward(net->layers[i], X);
  }
  return X;
}

static void backward(Network *net) {
  for (int i = net->num_layers - 1; i >= 0; i--) {
    net->layers[i]->backward(net->layers[i]);
  }
}

static void zero_grad_network(Network *net) {
  for (int i = 0; i < net->num_layers; i++) {
    net->layers[i]->zero_grad(net->layers[i]);
  }
}

Network *new_network(int batch_size) {
  Network *net = malloc(sizeof(Network));

  // layers
  net->layers = malloc(5 * sizeof(Layer *));
  net->layers[0] = new_linear_layer(batch_size, 28 * 28, 50);
  net->layers[1] = new_sigmoid_layer(batch_size, 50);
  net->layers[2] = new_linear_layer(batch_size, 50, 100);
  net->layers[3] = new_sigmoid_layer(batch_size, 100);
  net->layers[4] = new_linear_layer(batch_size, 100, 10);
  net->num_layers = 5;

  // functions
  net->forward = forward;
  net->backward = backward;
  net->zero_grad = zero_grad_network;
  return net;
}
