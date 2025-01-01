#ifndef MNIST_H
#define MNIST_H

#define ROWS 1000
#define COLS 28 * 28

void load_csv(const char *filename, double data[ROWS][COLS], int labels[ROWS]);

#endif // MNIST_H
