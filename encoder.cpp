#include "encoder.h"

void Blur(
	const float *img,
	float *output,
	const int width, const int height
) {

}

void DownSample(
  const float *img,
  float *output,
  const int width, const int height
) {
  float sum;
  for (int i = 0; i < height; i+=2) {
    for (int j = 0; j < width; j+=2) {
      int pos = i*width+j;
      sum = img[pos] +img[pos+1] + img[pos+width] + img[pos+width+1];
      sum /= 4;
      output[i*width/4+j/2] = sum;
    }
  }
}