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

void load_exposures( string source_dir, uint8_t **img_list_b, uint8_t **img_list_g, uint8_t **img_list_r, uint8_t **small_b, uint8_t **small_g, uint8_t **small_r, int **exposure_log2, unsigned int *row_input, unsigned int *col_input, int pic_count ){
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
					if( ( *row_input ) == 0 && ( *col_input ) == 0 ){
						cout << "allocating area for input pictures" << pic_count << endl;
						( *row_input ) = input_pic.rows;
						( *col_input ) = input_pic.cols;
						distance = ( *row_input ) * ( *col_input );
						*img_list_b = new uint8_t[ ( *row_input ) * ( *col_input ) * pic_count ];
						*img_list_g = new uint8_t[ ( *row_input ) * ( *col_input ) * pic_count ];
						*img_list_r = new uint8_t[ ( *row_input ) * ( *col_input ) * pic_count ];
						*small_b = new uint8_t[ SMALLPIXELS * pic_count ];
						*small_g = new uint8_t[ SMALLPIXELS * pic_count ];
						*small_r = new uint8_t[ SMALLPIXELS * pic_count ];
						*exposure_log2 = new int[ pic_count ];
						//cout << "Picture size is : " << distance << " " << channels[0].total() << endl;
					}
					cout << "Reading " << pointer + 1 << "th picture to memory" << endl;

					// saving picture by channel
					vector<Mat> channels;
					split( input_pic, channels );
					unsigned offset = pointer * distance;
					memcpy( ( *img_list_b ) + offset, channels[0].data, channels[0].total() );
					memcpy( ( *img_list_g ) + offset, channels[1].data, channels[1].total() );
					memcpy( ( *img_list_r ) + offset, channels[2].data, channels[2].total() );

					// getting 10 * 10 resized image from original image
					Mat small;
					resize( input_pic, small, cv::Size( 10, 10 ), 0, 0 );
					vector<Mat> small_channels;
					split( small, small_channels );
					offset = pointer * SMALLPIXELS;
					memcpy( ( *small_b ) + offset, small_channels[0].data, small_channels[0].total() );
					memcpy( ( *small_g ) + offset, small_channels[1].data, small_channels[1].total() );
					memcpy( ( *small_r ) + offset, small_channels[2].data, small_channels[2].total() );

					//for( unsigned i = offset ; i < distance + offset ; ++i ){
					//	cout << (int)( *img_list_b )[i] << " " << (int)( *img_list_g )[i] << " " << (int)( *img_list_r )[i] << " " << endl;
					//}
					pointer++;
				}
			}
		}
	}
	txt.close();
	return;
}

int main( int argc, char* argv[] ){

	/* ------------ variables ------------ */
	uint8_t *img_list_b, *img_list_g, *img_list_r;
	uint8_t *small_b, *small_g, *small_r;

	int *exposure_log2;
	unsigned row, col, pic_count;

	if( argc != 4 ){
		cerr << "[Usage] hdr <input img dir> <output .hdr name> <original picture count> \n[Example] hdr taipei taipei.hdr 11" << endl;
		return 0;
	}
	string img_dir = argv[1];
	string output_name = argv[2];
	pic_count = atoi( argv[3] );

	cout << "reading input images ... " << endl;

	row = col = 0;
	load_exposures( img_dir, &img_list_b, &img_list_g, &img_list_r, &small_b, &small_g, &small_r, &exposure_log2, &row, &col, pic_count );

	delete[] img_list_b;
	delete[] img_list_g;
	delete[] img_list_r;

	delete[] small_b;
	delete[] small_g;
	delete[] small_r;

	delete[] exposure_log2;
}
