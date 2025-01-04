#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>

void backward_none(Tensor *self) {
  // Do nothing
}

Tensor *new_tensor(int *shape, int rank) {
  Tensor *t = malloc(sizeof(Tensor));
  t->elements = malloc(sizeof(double));
  t->gradients = malloc(sizeof(double));

  t->shape = malloc(rank * sizeof(int));
  for (int i = 0; i < rank; i++) {
    t->shape[i] = shape[i];
  }
  t->rank = rank;

  t->size = 1;
  for (int i = 0; i < rank; i++) {
    t->shape[i] = shape[i];
    t->size *= shape[i];
  }

  t->inputs = NULL;
  t->num_inputs = 0;

  t->backward = backward_none;
  return t;
}

void free_tensor(Tensor *t) {
  free(t->elements);
  free(t->gradients);
  free(t->shape);
  free(t->inputs);
  free(t);
}

void print_tensor(Tensor *t) {
  if (t->rank != 2) {
    fprintf(stderr, "Error: rank must be 2\n");
  }

  printf("Tensor: rank=%d, shape=[", t->rank);
  for (int i = 0; i < t->rank; i++) {
    printf("%d", t->shape[i]);
    if (i < t->rank - 1) {
      printf(", ");
    }
  }
  printf("]\n");

  printf("Elements:\n");
  for (int i = 0; i < t->shape[0]; i++) {
    for (int j = 0; j < t->shape[1]; j++) {
      printf("%f ", t->elements[i * t->shape[1] + j]);
    }
    printf("\n");
  }
}

static void backward_tensor_add_tensor(Tensor *self) {
  Tensor *a = self->inputs[0];
  Tensor *b = self->inputs[1];

  for (int i = 0; i < self->size; i++) {
    a->gradients[i] += self->gradients[i];
    b->gradients[i] += self->gradients[i];
  }
}

void tensor_add_tensor(Tensor *result, const Tensor *a, const Tensor *b) {
  if (result->rank != a->rank || result->rank != b->rank ||
      result->shape[0] != a->shape[0] || result->shape[0] != b->shape[0] ||
      result->shape[1] != a->shape[1] || result->shape[1] != b->shape[1]) {
    fprintf(stderr, "Error: size mismatch\n");
    exit(1);
  }

  for (int i = 0; i < result->shape[0]; i++) {
    for (int j = 0; j < result->shape[1]; j++) {
      result->elements[i * result->shape[1] + j] =
          a->elements[i * a->shape[1] + j] + b->elements[i * b->shape[1] + j];
    }
  }

  result->backward = backward_tensor_add_tensor;
}

int main() {
  printf("Hello, World!\n");

  Tensor *t = new_tensor((int[]){2, 3}, 2);
  print_tensor(t);

  return 0;
}