#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "lapacke_utils.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <limits>
#include <vector>
#include <math.h>

#define CONSTANTL 50
#define SMALLPIXELS 100

using namespace std;
using namespace cv;

void load_exposures( string source_dir, uint8_t **img_list_b, uint8_t **img_list_g, uint8_t **img_list_r, uint8_t **small_b, uint8_t **small_g, uint8_t **small_r, int *exposure_log2, unsigned int *row_input, unsigned int *col_input, int pic_count ){
	fstream txt;
	txt.open( source_dir + "/image_list.txt", fstream::in );
	if( !txt ){
		fprintf(stderr, "no image_list.txt found\n");
		exit(EXIT_FAILURE);
	}
	else{
		string temp;
		unsigned pointer = 0;
		unsigned distance = 0;
		while( getline( txt, temp ) ){
			if( temp[0] != '#' ){
				istringstream iss( source_dir + "/" + temp );
				iss >> temp;
				cout << temp << endl;
				Mat input_pic = imread( temp, CV_LOAD_IMAGE_COLOR);
				if( input_pic.data ){
					vector<Mat> channels;
					split( input_pic, channels );
					cout << channels[0] << endl;
				}
			}
		}
	}
	txt.close();
	return;
}

int main( int argc, char* argv[] ){
	if( argc != 3 ){
		cerr << "[Usage] hdr <input img dir> <output .hdr name> <original picture count> \n[Example] hdr taipei taipei.hdr" << endl;
		return 0;
	}
	string img_dir = argv[1];
	string output_name = argv[2];

	/* ------------ variables ------------ */
	uint8_t *img_list_b, *img_list_g, *img_list_r;
	uint8_t *small_b, *small_g, *small_r;

	int *exposure_log2;
	unsigned row, col, pic_count;

	cout << "reading input images ... " << endl;

	load_exposures( img_dir, &img_list_b, &img_list_g, &img_list_r, &small_b, &small_g, &small_r, exposure_log2, &row, &col, pic_count );

	//delete[] img_list_b;
	//delete[] img_list_g;
	//delete[] img_list_r;

	//delete[] small_b;
	//delete[] small_g;
	//delete[] small_r;

	//delete[] exposure_log2;
}
