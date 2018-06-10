#include <cmath>
#include "decoder.h"

float decayFactor(
  const float *bigFrame_L,
  const float *E,
  const int windowSize,
  const int x, const int y,
  const int width, const int height
) {
  int n = windowSize/2;
  float min = E[y*width+x];
  for (int i = y-n; i <= y+n; i++) {
    for (int j = x-n; j <= x+n; j++) {
      if (E[i*width+j] < min) {
        min = E[i*width+j];
      }
    }
  }
  return sqrtf(min/4);
}