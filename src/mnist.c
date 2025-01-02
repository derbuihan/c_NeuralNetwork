#include "mnist.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_separator(char c) { return c == ',' || isspace(c); }

void load_csv(const char *filename, double *data, int rows, int cols) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Error: could not open %s: %s\n", filename,
            strerror(errno));
    exit(1);
  }

  char line[4096];

  for (int i = 0; i < rows; i++) {
    if (!fgets(line, sizeof(line), file)) {
      fprintf(stderr, "Error: unexpected end of file at row %d\n", i);
      fclose(file);
      exit(1);
    }

    char *ptr = line;
    for (int j = 0; j < cols; j++) {
      while (is_separator(*ptr))
        ptr++;

      if (*ptr == '\0') {
        fprintf(stderr, "Error: not enough columns at row %d\n", i);
        fclose(file);
        exit(1);
      }

      char *endptr;
      errno = 0;

      double value = strtod(ptr, &endptr);

      if (errno == ERANGE) {
        fprintf(stderr, "Error: number out of range at row %d, col %d\n", i, j);
        fclose(file);
        exit(1);
      }
      if (ptr == endptr) {
        fprintf(stderr, "Error: invalid number format at row %d, col %d\n", i,
                j);
        fclose(file);
        exit(1);
      }

      data[i * cols + j] = value;
      ptr = endptr;

      if (j == cols - 1) {
        while (isspace(*ptr))
          ptr++;
      }
    }
  }

  fclose(file);
}

void load_mnist_datasets(const char *filename, double *data, double *labels) {
  double *buffer = malloc((ROWS + 1) * (COLS + 1) * sizeof(double));
  if (!buffer) {
    fprintf(stderr, "Error: memory allocation failed: %s\n", strerror(errno));
    exit(1);
  }

  load_csv(filename, buffer, ROWS + 1, COLS + 1);

  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      data[i * COLS + j] = buffer[(i + 1) * (COLS + 1) + j] / 255.0;
      data[i * COLS + j] -= 0.5;
    }
    // one-hot encoding
    for (int j = 0; j < 10; j++) {
      labels[i * 10 + j] = 0;
      if (buffer[(i + 1) * (COLS + 1) + COLS] == j) {
        labels[i * 10 + j] = 1;
      }
    }
  }

  free(buffer);
}