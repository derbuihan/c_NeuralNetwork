#ifndef TENSOR_H
#define TENSOR_H

typedef struct Tensor Tensor;
struct Tensor {
  double *elements;
  double *gradients;

  int *shape;
  int rank;
  int size;

  Tensor **inputs;
  int num_inputs;
  void (*backward)(Tensor *self);
};

#endif // TENSOR_H
