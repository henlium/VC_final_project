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
