#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>

void load_csv(const char *filename, double data[ROWS][COLS], int labels[ROWS]) {

  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Error: could not open %s\n", filename);
    exit(1);
  }

  char buffer[4 * (COLS + 1)];
  fgets(buffer, sizeof(buffer), file); // skip header

  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      fscanf(file, "%lf,", &data[i][j]);
    }
    fscanf(file, "%d", &labels[i]);
  }

  fclose(file);
}
