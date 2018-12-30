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
#define SMALLDIM 10
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
					// cout << "Reading " << pointer + 1 << "th picture to memory" << endl;

					// saving picture by channel
					vector<Mat> channels;
					split( input_pic, channels );
					unsigned offset = pointer * distance;
					memcpy( ( *img_list_b ) + offset, channels[0].data, channels[0].total() * sizeof( uint8_t ) );
					memcpy( ( *img_list_g ) + offset, channels[1].data, channels[1].total() * sizeof( uint8_t ) );
					memcpy( ( *img_list_r ) + offset, channels[2].data, channels[2].total() * sizeof( uint8_t ) );

					// getting 10 * 10 resized image from original image
					Mat small;
					resize( input_pic, small, cv::Size( SMALLDIM, SMALLDIM ), 0, 0 );
					vector<Mat> small_channels;
					split( small, small_channels );
					offset = pointer * SMALLPIXELS;
					memcpy( ( *small_b ) + offset, small_channels[0].data, small_channels[0].total() * sizeof( uint8_t ) );
					memcpy( ( *small_g ) + offset, small_channels[1].data, small_channels[1].total() * sizeof( uint8_t ) );
					memcpy( ( *small_r ) + offset, small_channels[2].data, small_channels[2].total() * sizeof( uint8_t ) );

					//for( unsigned i = offset ; i < distance + offset ; ++i ){
					//	cout << (int)( *img_list_b )[i] << " " << (int)( *img_list_g )[i] << " " << (int)( *img_list_r )[i] << " " << endl;
					//}

					float exposure;
					iss >> exposure;
					( *exposure_log2 )[ pointer ] = log2( exposure );
					// cout << ( *exposure_log2 )[ pointer ] << endl;

					pointer++;
				}
			}
		}
	}
	txt.close();
	return;
}

void response_curve_solver( uint8_t *Z, int *B, int l, int *w, double **g, int pic_count ){
	int n = 256;
	double A[ ( pic_count * SMALLPIXELS + n + 1 ) * ( SMALLPIXELS + n ) ] = {0};
	double b[ pic_count * SMALLPIXELS + n + 1 ] = {0};

	int k = 0;
	const unsigned width_a = SMALLPIXELS + n;
	for( int i = 0; i < SMALLPIXELS; ++i ){
		for( int j = 0; j < pic_count; ++j ){
			uint8_t z = Z[j * SMALLPIXELS + i];
			int wij = w[z];
			A[     k * width_a + z ] = wij;
			A[ k * width_a + n + i ] = -wij;
			b[                   k ] = wij * B[ j ];
			++k;
		}
	}
	
	A[ k * width_a + 128 ] = 1;
	++k;

	for( int i = 0; i < n - 1; ++i ){
		A[     k * width_a + i ] = l * w[ i + 1 ];
		A[ k * width_a + i + 1 ] = l * w[ i + 1 ] * ( -2 );
		A[ k * width_a + i + 2 ] = l * w[ i + 1 ];
		k++;
	}
	
	// lstsq @@
}

int main( int argc, char* argv[] ){

	/* ------------ variables ------------ */
	uint8_t *img_list_b, *img_list_g, *img_list_r;
	uint8_t *small_b, *small_g, *small_r;
	double *gb, *gg, *gr;

	int *exposure_log2;
	unsigned row, col, pic_count;

	if( argc != 4 ){
		cerr << "[Usage] hdr <input img dir> <output .hdr name> <original picture count> \n[Example] hdr taipei taipei.hdr 11" << endl;
		return 0;
	}
	string img_dir = argv[1];
	string output_name = argv[2];
	pic_count = atoi( argv[3] );

	row = col = 0;

	/* ------------ load picture and small reference input ------------ */
	cout << "reading input images ... " << endl;
	load_exposures( img_dir, &img_list_b, &img_list_g, &img_list_r, &small_b, &small_g, &small_r, &exposure_log2, &row, &col, pic_count );
	cout << "done" << endl;

	/* ------------ solve response curves ------------ */
	cout << "Solving response curves ... " << endl;

	int *w = new int[ 256 ];
	for( int i = 0; i < 128; ++i ){
		w[       i ] = i;
		w[ i + 128 ] = 255 - i;
	}

	response_curve_solver( small_b, exposure_log2, CONSTANTL, w, &gb, pic_count );
	response_curve_solver( small_g, exposure_log2, CONSTANTL, w, &gg, pic_count );
	response_curve_solver( small_r, exposure_log2, CONSTANTL, w, &gr, pic_count );

	cout << "done" << endl;

	delete[] img_list_b;
	delete[] img_list_g;
	delete[] img_list_r;

	delete[] small_b;
	delete[] small_g;
	delete[] small_r;

	delete[] exposure_log2;

	delete[] w;
}
