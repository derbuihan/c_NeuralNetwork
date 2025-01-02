#ifndef SIMPLE_NETWORK_H
#define SIMPLE_NETWORK_H

#include "matrix.h"

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

void init_network(Network *net);
void free_network(Network *net);
Matrix *forward(Network *net, Matrix *X);
void backward(Network *net, Matrix *X, Matrix *y_true);
Network *new_network();

#endif // SIMPLE_NETWORK_H
