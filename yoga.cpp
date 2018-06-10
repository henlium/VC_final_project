#include <cmath>
#include <climits>
#include "decoder.h"

float decayFactor(
  const float *bigFrame_L,
  const float *curFrame_L,
  const int windowSize,
  const int x, const int y,
  const int width, const int height
) {
  int n = windowSize/2;
  float min = __FLT_MAX__;
  for (int i = y-n; i <= y+n; i++) {
    for (int j = x-n; j <= x+n; j++) {
      float E = patch_diff(bigFrame_L, curFrame_L, x, y, i, j, width, height);
      if (E < min) {
        min = E;
      }
    }
  }
  return sqrtf(min/4);
}