#include <cmath>
#include <cstdlib>
#include "decoder.h"
#include "encoder.h"

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
) {
  float *curFrame_L = (float*)malloc(sizeof(float)*width*height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float w_sum = 0.0f, weighted_F = 0.0f;
      float sigma = decayFactor(lastBigFrame_L, curFrame_L, 9, x, y, width, height);
      for (int j = y-4; j <= y+4; j++) {
        if (j<0 or j>=height) continue;
        for (int i = x-4; i <= x+4; i++) {
          if (i<0 or i>=width) continue;
          float w = expf(-1 * patch_diff(lastBigFrame_L, curFrame_L, x, y, i, j, width, height)/ (2*sigma*sigma));
          w_sum += w;
          weighted_F += lastBigFrame_H[j*width+j];
        }
      }
      output[y*width+x] = weighted_F/w_sum;
    }
  }
}

void interpolate(
  const float *img,
  float *output,
  const int width, const int height
) {
  int half_width = width / 2;
  int half_height = height / 2;
  for (int i = 0; i < half_height; i++) {
    for (int j = 0; j < half_width; j++) { 
      int TL = i * 2 * width + j * 2; // TR = TL+1
      int BL = (i * 2 + 1) * width + j * 2; // BR = BL+1
      int index = i*half_width+j;
      if (i == 0) {
        if (j == 0) {
          output[TL] = img[index];
          output[TL+1] = (3 * img[index] + img[index+1]) / 4;
          output[BL] = (3 * img[index] + img[index+half_width]) / 4;
          output[BL+1] = (9 * img[index] + 3 * img[index+1] + 3 * img[index+half_width] + img[index+half_width + 1]) / 16;
        }
        else if (j == half_width - 1) {
          output[TL] = (img[index-1] + 3 * img[index]) / 4;
          output[TL+1] = img[index];
          output[BL] = (9 * img[index] + 3 * img[index-1] + 3 * img[index+half_width] + img[index+half_width - 1]) / 16;
          output[BL+1] = (img[index] + 3 * img[index+half_width]) / 4;
        }
        else {
          output[TL] = (3 * img[index - 1] + img[index]) / 4;
          output[TL+1] = (3 * img[index] + img[index+1]) / 4;
          output[BL] = (9 * img[index] + 3 * img[index-1] + 3 * img[index+half_width] + img[index+half_width - 1]) / 16;
          output[BL+1] = (9 * img[index] + 3 * img[index+1] + 3 * img[index+half_width] + img[index+half_width + 1]) / 16;
        }
      }
      else if (i == half_height - 1) {
        if (j == 0) {
          output[TL] = (3 * img[index] + img[index-half_width]) / 4;
          output[TL+1] = (9 * img[index] + 3 * img[index+1] + 3 * img[index-half_width] + img[index-half_width + 1]) / 16;
          output[BL] = img[index];
          output[BL+1] = (3 * img[index] + img[index+1]) / 4;
        }
        else if (j == half_width - 1) {
          output[TL] = (9 * img[index] + 3 * img[index-1] + 3 * img[index-half_width] + img[index-half_width - 1]) / 16;
          output[TL+1] = (img[index] + 3 * img[index-half_width]) / 4;
          output[BL] = (img[index-1] + 3 * img[index]) / 4;
          output[BL+1] = img[index];
        }
        else {
          output[TL] = (9 * img[index] + 3 * img[index-1] + 3 * img[index-half_width] + img[index-half_width - 1]) / 16;
          output[TL+1] = (9 * img[index] + 3 * img[index+1] + 3 * img[index-half_width] + img[index-half_width + 1]) / 16;
          output[BL] = (3 * img[index - 1] + img[index]) / 4;
          output[BL+1] = (3 * img[index] + img[index+1]) / 4; 
        }
      }
      else if (j == 0) {
        output[TL] = (3 * img[index] + img[index - half_width]) / 4;
        output[TL+1] = (9 * img[index] + 3 * img[index+1] + 3 * img[index-half_width] + img[index-half_width + 1]) / 16;
        output[BL] = (3 * img[index] + img[index + half_width]) / 4;
        output[BL+1] = (9 * img[index] + 3 * img[index+1] + 3 * img[index+half_width] + img[index+half_width + 1]) / 16;
      }
      else if (j == half_width - 1) {
        output[TL] = (9 * img[index] + 3 * img[index-1] + 3 * img[index-half_width] + img[index-half_width - 1]) / 16;
        output[TL+1] = (img[index] + 3 * img[index-half_width]) / 4;
        output[BL] = (9 * img[index] + 3 * img[index-1] + 3 * img[index+half_width] + img[index+half_width - 1]) / 16;
        output[BL+1] = (img[index] + 3 * img[index+half_width]) / 4;
      }
      else {
        output[TL] = (9 * img[index] + 3 * img[index-1] + 3 * img[index-half_width] + img[index-half_width - 1]) / 16;
        output[TL+1] = (9 * img[index] + 3 * img[index+1] + 3 * img[index-half_width] + img[index-half_width + 1]) / 16;
        output[BL] = (9 * img[index] + 3 * img[index-1] + 3 * img[index+half_width] + img[index+half_width - 1]) / 16;
        output[BL+1] = (9 * img[index] + 3 * img[index+1] + 3 * img[index+half_width] + img[index+half_width + 1]) / 16;
      }
    }
  }
}
