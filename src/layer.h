#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

#define BATCH_SIZE 64

typedef struct Layer Layer;
struct Layer {
  Matrix **params;
  int num_params;

  int *options;
  int num_options;

  Matrix *(*forward)(Layer *layer, Matrix *X);
  void (*backward)(Layer *layer);
};

void free_layer(Layer *layer);
Layer *new_linear_layer(int input_size, int output_size);
Layer *new_sigmoid_layer(int input_size);

#endif // LAYER_H
