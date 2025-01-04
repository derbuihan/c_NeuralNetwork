#include "simple_network.h"

#include <stdlib.h>

void free_network(Network *net) {
  for (int i = 0; i < net->num_layers; i++) {
    free_layer(net->layers[i]);
  }
  free(net);
}

Matrix *forward(Network *net, Matrix *X) {
  for (int i = 0; i < net->num_layers; i++) {
    X = net->layers[i]->forward(net->layers[i], X);
  }
  return X;
  /*
    Matrix *a1 = net->layers[0]->forward(net->layers[0], X);
    Matrix *z1 = net->layers[1]->forward(net->layers[1], a1);
    Matrix *a2 = net->layers[2]->forward(net->layers[2], z1);
    Matrix *z2 = net->layers[3]->forward(net->layers[3], a2);
    Matrix *a3 = net->layers[4]->forward(net->layers[4], z2);
    return a3;
  */
}

void backward(Network *net) {
  for (int i = net->num_layers - 1; i >= 0; i--) {
    net->layers[i]->backward(net->layers[i]);
  }
}

Network *new_network() {
  Network *net = malloc(sizeof(Network));

  // layers
  net->layers = malloc(5 * sizeof(Layer *));
  net->layers[0] = new_linear_layer(28 * 28, 50);
  net->layers[1] = new_sigmoid_layer(50);
  net->layers[2] = new_linear_layer(50, 100);
  net->layers[3] = new_sigmoid_layer(100);
  net->layers[4] = new_linear_layer(100, 10);
  net->num_layers = 5;

  // functions
  net->forward = forward;
  net->backward = backward;
  return net;
}
