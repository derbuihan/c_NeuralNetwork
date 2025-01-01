#ifndef MNIST_H
#define MNIST_H

#define ROWS 10000
#define COLS 28 * 28

void load_csv(const char *filename, double *data, int rows, int cols);
void load_mnist_datasets(const char *filename, double *data, double *labels);

#endif // MNIST_H
