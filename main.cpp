#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <vector>
#include "pgm.h"
#include "encoder.h"
#include <string>
using namespace std;

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

int period;
int width, height, colors;
std::string list_path;
std::string input_dir("data/");
std::string output_dir("LR/");

void parse_args(int argc, char **argv);
void DownSampleProcess();

int main(int argc, char **argv)
{
	parse_args(argc, argv);

	DownSampleProcess();

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
}

void DownSampleProcess()
{
	/* Open list file */
	FILE * f_list;
	f_list = fopen(list_path.c_str(), "r");
	if (f_list == NULL) { fprintf(stderr, "Error opening file %s\n", list_path.c_str()); abort(); }

	/* Memory allocation */
	const int SIZE = width*height;
	unique_ptr<uint8_t[]> o(new uint8_t[SIZE]);
	float *img_float = (float*)malloc(sizeof(float)*SIZE);
	float *output = (float*)malloc(sizeof(float)*SIZE/4);
	unique_ptr<uint8_t[]> o_quater(new uint8_t[SIZE/4]);

	int frm_count = 0;
	char filename[128];
	std::string filepath;
	bool suc;
	while( fgets(filename, 127, f_list) != NULL ) {
		fprintf(stderr, "frame %d\n", frm_count);
		filename[strlen(filename) -1] = 0;
		filepath = input_dir + filename;

		/* Read frame */
		auto img = ReadNetpbm(width, height, colors, suc, filepath.c_str());
		if (not (suc)) {
			puts("Something wrong with reading the input image files.");
			abort();
		}

		filepath = output_dir + filename;
		if (frm_count % period == 0) {
			transform(img.get(), img.get()+SIZE, o.get(), [](float f) -> uint8_t { return max(min(int(f+0.5f), 255), 0); });
			WritePGM(o.get(), width, height, filepath.c_str());
		} else {
			copy(img.get(), img.get()+SIZE, img_float);
			DownSample(img_float, output, width, height);
			transform(output, output+SIZE/4, o_quater.get(), [](float f) -> uint8_t { return max(min(int(f+0.5f), 255), 0); });
			WritePGM(o_quater.get(), width/2, height/2, filepath.c_str());
		}

		frm_count += 1;		
		if (frm_count >= 300) break;
	}

	fclose(f_list);
}
