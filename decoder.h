void SR(
  const float *lastBigFrame,
  const float *lastBigFrame_L,
  const float *lastBigFrame_H,
  const float *nextBigFrame,
  const float *nextBigFrame_L,
  const float *nextBigFrame_H,
  const float *curFrame,
  float *output,
  const int width, const int height
);

void interpolate(
  const float *img,
  float *output,
  const int width, const int height
);

int windowSize(
  const float *bigFrame_L,
  const float *curFrame,
  const int x, const int y,
  const int width, const int height
);

float decayFactor(
  const float *bigFrame_L,
  const float *curFrame_L,
  const float *E,
  const int windowSize,
  const int x, const int y,
  const int width, const int height
);

float patch_diff(
  const float *bigFrame_L,
  const float *curFrame,
  const int x, const int y,
  const int i, const int j,
  const int width, const int height
);
