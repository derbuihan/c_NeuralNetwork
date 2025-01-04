#ifndef SIMPLE_NETWORK_H
#define SIMPLE_NETWORK_H

#include "layer.h"

typedef struct Network Network;
struct Network {
  Layer **layers;
  int num_layers;

  Matrix *(*forward)(Network *net, Matrix *X);
  void (*backward)(Network *net);
  void (*zero_grad)(Network *net);
};

void free_network(Network *net);
Network *new_network(int BATCH_SIZE);

#endif // SIMPLE_NETWORK_H
