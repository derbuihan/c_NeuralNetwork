#ifndef SIMPLE_NETWORK_H
#define SIMPLE_NETWORK_H

#include "layer.h"

typedef struct Network Network;
struct Network {
  Layer **layers;
  int num_layers;

  Matrix *(*forward)(Network *net, Matrix *X);
  void (*backward)(Network *net);
};

void free_network(Network *net);
Network *new_network();

#endif // SIMPLE_NETWORK_H
