#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

typedef struct Layer Layer;
struct Layer {
  Matrix **params;
  int num_params;

  int *options;
  int num_options;

  Matrix *(*forward)(Layer *layer, Matrix *X);
  void (*backward)(Layer *layer);
  void (*zero_grad)(Layer *layer);
};

void free_layer(Layer *layer);
Layer *new_linear_layer(int batch_size, int input_size, int output_size);
Layer *new_sigmoid_layer(int batch_size, int input_size);

#endif // LAYER_H
