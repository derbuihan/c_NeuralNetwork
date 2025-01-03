#ifndef MNIST_H
#define MNIST_H

void load_csv(const char *filename, double *data, int rows, int cols);
void load_mnist_datasets(const char *filename, double *data, double *labels,
                         int ROWS, int COLS);

#endif // MNIST_H
