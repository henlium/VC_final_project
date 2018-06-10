#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include "decoder.h"
#include "encoder.h"

#define PATCH_SIZE 5

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
  interpolate(curFrame, curFrame_L, width, height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float w_sum = 0.0f, weighted_F = 0.0f;
      float sigma = decayFactor(lastBigFrame_L, curFrame_L, 9, x, y, width, height);
      for (int j = y-4; j <= y+4; j++) {
        if (j<0 or j>=height) continue;
        for (int i = x-4; i <= x+4; i++) {
          if (i<0 or i>=width) continue;
          float w = expf(-1 * patch_diff(lastBigFrame_L, curFrame_L, x, y, i, j, width, height) / (2*sigma*sigma));
          w_sum += w;
          weighted_F += w * lastBigFrame_H[j*width+i];
        }
      }
      output[y*width+x] = weighted_F/w_sum + curFrame_L[y*width+x];
    }
  }
}

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
      float E = patch_diff(bigFrame_L, curFrame_L, x, y, j, i, width, height);
      if (E < min) {
        min = E;
      }
    }
  }
  return sqrtf(min/4);
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


float patch_diff(
  const float *bigFrame_L, 
  const float *curFrame_L, 
  const int x, const int y,
  const int i, const int j,
  const int width, const int height
) {
  int patch_radius = PATCH_SIZE >> 1;
  float patch_big[PATCH_SIZE][PATCH_SIZE];
  float patch_cur[PATCH_SIZE][PATCH_SIZE];
  patch_big[patch_radius][patch_radius] = bigFrame_L[j*width+i];
  patch_cur[patch_radius][patch_radius] = curFrame_L[y*width+x];
  int target_x, target_y;
  int target_i, target_j;
  for (int l = -patch_radius; l <= patch_radius; l++) { // for each height
    for (int k = -patch_radius; k <= patch_radius; k++) { // for each weight
      if (k < 0) {
        if (i + k < 0) target_i = -k - i;
        else target_i = i + k;
        if (x + k < 0) target_x = -k - x;
        else target_x = x + k;
      }
      else {
        if (i + k >= width) target_i = 2 * width - i - k - 1;
        else target_i = i + k;
        if (x + k >= width) target_x = 2 * width - x - k - 1;
        else target_x = x + k;
      }
      if (l < 0) {
        if (j + l < 0) target_j = -l - j;
        else target_j = j + l;
        if (y + l < 0) target_y = -l - y;
        else target_y = y + l;
      }
      else {
        if (j + l >= height) target_j = 2 * height - j - l - 1;
        else target_j = j + l;
        if (y + l >= height) target_y = 2 * height - y - l - 1;
        else target_y = y + l;
      }
      patch_big[patch_radius+l][patch_radius+k] = bigFrame_L[target_j*width+target_i];
      patch_cur[patch_radius+l][patch_radius+k] = curFrame_L[target_y*width+target_x];
    }
  }
  float ans = 0;
  for (int l = 0; l < PATCH_SIZE; l++) {
    for (int k = 0; k < PATCH_SIZE; k++) {
      ans += patch_cur[l][k] * patch_cur[l][k] - patch_big[l][k] * patch_big[l][k];
    }
  }
  return ans;
}
