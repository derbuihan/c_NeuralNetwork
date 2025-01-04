#include "optimizer.h"

#include <math.h>
#include <stdlib.h>

/* SGD Optimizer */

void free_sgd_optimizer(SGD_Optimizer *optim) { free(optim); }

static void zero_grad_sdg(SGD_Optimizer *optim) {
  Network *net = optim->net;
  net->zero_grad(net);
}

static void update_matrix_sgd(Matrix *m, double learning_rate) {
  for (int i = 0; i < m->rows * m->cols; i++) {
    m->elements[i] -= learning_rate * m->gradients[i];
  }
}

static void step_sgd(SGD_Optimizer *optim) {
  Network *net = optim->net;
  double learning_rate = optim->learning_rate;

  for (int i = 0; i < net->num_layers; i++) {
    Layer *layer = net->layers[i];
    for (int j = 0; j < layer->num_params; j++) {
      Matrix *m = layer->params[j];
      update_matrix_sgd(m, learning_rate);
    }
  }
}

SGD_Optimizer *new_sgd_optimizer(Network *net, double learning_rate) {
  SGD_Optimizer *optim = malloc(sizeof(SGD_Optimizer));
  optim->net = net;
  optim->learning_rate = learning_rate;
  optim->zero_grad = zero_grad_sdg;
  optim->step = step_sgd;
  return optim;
}

/* Adam Optimizer */

void free_adam_optimizer(Adam_Optimizer *optim) {
  Network *net = optim->net;
  for (int i = 0; i < net->num_layers; i++) {
    Layer *layer = net->layers[i];
    for (int j = 0; j < layer->num_params; j++) {
      int idx = i * layer->num_params + j;
      int params_size = layer->params[j]->rows * layer->params[j]->cols;
      optim->moment1[idx] = malloc(params_size * sizeof(double));
      optim->moment2[idx] = malloc(params_size * sizeof(double));
    }
  }
  free(optim->moment1);
  free(optim->moment2);
  free(optim);
}

static void zero_grad_adam(Adam_Optimizer *optim) {
  Network *net = optim->net;
  net->zero_grad(net);
}

static void update_matrix_adam(Matrix *m, double *moment1, double *moment2,
                               double beta1, double beta2, double epsilon,
                               double learning_rate, int t) {
  /* m: (rows, cols)
   * moment1: (rows, cols)
   * moment2: (rows, cols)
   */
  double beta1_t = pow(beta1, t);
  double beta2_t = pow(beta2, t);

  for (int i = 0; i < m->rows * m->cols; i++) {
    moment1[i] = beta1 * moment1[i] + (1 - beta1) * m->gradients[i];
    moment2[i] =
        beta2 * moment2[i] + (1 - beta2) * m->gradients[i] * m->gradients[i];

    double m_hat = moment1[i] / (1 - beta1_t);
    double v_hat = moment2[i] / (1 - beta2_t);
    m->elements[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
  }
}

void step_adam(Adam_Optimizer *optim) {
  optim->t++;
  Network *net = optim->net;
  int idx = 0;
  for (int i = 0; i < net->num_layers; i++) {
    Layer *layer = net->layers[i];
    for (int j = 0; j < layer->num_params; j++) {
      Matrix *m = layer->params[j];
      update_matrix_adam(m, optim->moment1[idx], optim->moment2[idx],
                         optim->beta1, optim->beta2, optim->epsilon,
                         optim->learning_rate, optim->t);
      idx++;
    }
  }
}

Adam_Optimizer *new_adam_optimizer(Network *net, double learning_rate) {
  Adam_Optimizer *optim = malloc(sizeof(Adam_Optimizer));
  optim->net = net;
  optim->learning_rate = learning_rate;

  optim->beta1 = 0.9;
  optim->beta2 = 0.999;
  optim->epsilon = 1e-8;
  optim->t = 0;

  int count_params = 0;
  for (int i = 0; i < net->num_layers; i++) {
    Layer *layer = net->layers[i];
    count_params += layer->num_params;
  }

  optim->moment1 = malloc(count_params * sizeof(double *));
  optim->moment2 = malloc(count_params * sizeof(double *));

  int idx = 0;
  for (int i = 0; i < net->num_layers; i++) {
    Layer *layer = net->layers[i];
    for (int j = 0; j < layer->num_params; j++) {
      int params_size = layer->params[j]->rows * layer->params[j]->cols;
      optim->moment1[idx] = malloc(params_size * sizeof(double));
      optim->moment2[idx] = malloc(params_size * sizeof(double));
      idx++;
    }
  }

  optim->zero_grad = zero_grad_adam;
  optim->step = step_adam;

  return optim;
}
