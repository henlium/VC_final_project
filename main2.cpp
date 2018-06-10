#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <vector>
#include "pgm.h"
#include "decoder.h"
using namespace std;

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

int period;
int width, height, colors, width_half, height_half;
std::string list_path;
std::string input_dir("LR/");
std::string output_dir("results/");

void parse_args(int argc, char **argv);
void decode();

int main(int argc, char **argv)
{
	parse_args(argc, argv);
	decode();
}

void parse_args(int argc, char **argv) {
	if (argc != 6) {
		printf("Usage: %s <width> <height> <frame_list> <period> <output_dir>\n", argv[0]);
		abort();
	}

	list_path = argv[3];
	width = atoi(argv[1]);
	height = atoi(argv[2]);
	period = atoi(argv[4]);
	width_half = width/2;
	height_half = height/2;
}

void decode()
{
	/* Open list file */
	FILE * f_list;
	f_list = fopen(list_path.c_str(), "r");
	if (f_list == NULL) { fprintf(stderr, "Error opening file %s\n", list_path.c_str()); abort(); }

	/* Memory allocation */
	const int SIZE = width*height;
	unique_ptr<uint8_t[]> lastBigFrame(new uint8_t[SIZE]);
	unique_ptr<uint8_t[]> nextBigFrame(new uint8_t[SIZE]);
	unique_ptr<uint8_t[]> curFrame(new uint8_t[SIZE/4]);
	unique_ptr<uint8_t[]> o(new uint8_t[SIZE]);
	float *last_float = (float*)malloc(sizeof(float)*SIZE);
	float *next_float = (float*)malloc(sizeof(float)*SIZE);
	float *cur_float = (float*)malloc(sizeof(float)*SIZE/4);
	float *output = (float*)malloc(sizeof(float)*SIZE);

	/* Read first HR frame */
	int prd_len;	bool suc;
	char filename[period][128];
	fgets(filename[0], 127, f_list);
	filename[0][strlen(filename[0])-1] = 0;
	std::string filepath = input_dir + filename[0];
	lastBigFrame = ReadNetpbm(width, height, colors, suc, filepath.c_str());
	filepath = output_dir + filename[0];
	WritePGM(lastBigFrame.get(), width, height, filepath.c_str());
	if (not (suc)) {
			puts("Something wrong with reading the input image files.");
			abort();
	}
	copy(lastBigFrame.get(), lastBigFrame.get()+SIZE, last_float);

	while (true) {
		for (prd_len = 0; prd_len < period; prd_len++) {
			if (fgets(filename[prd_len], 127, f_list) == NULL) {
				filename[prd_len][strlen(filename[prd_len])-1] = 0;
				break;
			}
			filename[prd_len][strlen(filename[prd_len])-1] = 0;
		}
		if (prd_len == 0) break;

		/* Read next HR frame */
		filepath = input_dir + filename[prd_len];
		nextBigFrame = ReadNetpbm(width, height, colors, suc, filepath.c_str());
		if (not (suc)) {
			puts("Something wrong with reading the input image files.");
			abort();
		}
		copy(nextBigFrame.get(), nextBigFrame.get()+SIZE, next_float);
		
		/* Read the LR frames and process them */
		for (int i = 0; i < prd_len-1; i++) {
			filepath = input_dir + filename[i];
			curFrame = ReadNetpbm(width_half, height_half, colors, suc, filepath.c_str());
			if (not (suc)) {
				puts("Something wrong with reading the input image files.");
				abort();
			}

			SR(last_float, next_float, cur_float, output, width, height);
			transform(output, output+SIZE, o.get(), [](float f) -> uint8_t { return max(min(int(f+0.5f), 255), 0); });
			WritePGM(o.get(), width, height, filepath.c_str());
		}

		filepath = output_dir + filename[prd_len];
		WritePGM(nextBigFrame.get(), width, height, filepath.c_str());

		copy(next_float, next_float+SIZE, last_float);
	}
	
	fclose(f_list);
}
