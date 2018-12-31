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
	const unsigned  width_a = SMALLPIXELS + n;
	const unsigned height_a = pic_count * SMALLPIXELS + n + 1;

	double A[ ( height_a ) * ( width_a ) ] = {0};
	double b[ height_a ] = {0};

	int k = 0;
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
	double *temp = new double[ height_a ];
	double rcond = -1.0;
	int rank, info;
	info = LAPACKE_dgelsd( LAPACK_ROW_MAJOR, height_a, width_a, 1, A, width_a, b, 1, temp, rcond, &rank );
        /* Check for convergence */
	if( info > 0 ) {
		cout << "The algorithm computing SVD failed to converge;" << endl;
		cout << "the least squares solution could not be computed." << endl;
		exit( 1 );
	}

	*g = new double[ 256 ];
	memcpy( ( *g ), b, 256 * sizeof( double ) );

	delete[] temp;
}

void construct_radiance_map( int img_size, int pic_count, int offset, double *g, uint8_t *Z, int *ln_t, int *w, float *ln_E ){
	float acc_E[ img_size ]={0};
	for( int i = 0; i < img_size; ++i ){
		float acc_w = 0;
		for( int j = 0; j < pic_count; ++j ){
			uint8_t z = Z[ j * img_size + i ];
			acc_E[ i ] += w[ z ]*( g[ z ] - ln_t[ j ] );
			acc_w += w[ z ];
		}
		//cout << i << " : " << acc_E[ i ] << " " << acc_w << endl;
		ln_E[ i * 3 + offset ] = ( acc_w > 0 )? exp( acc_E[ i ] / acc_w ) : exp( acc_E[i] ); // may need to add exp here
		acc_w = 0;
	}
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
		w[ i + 128 ] = 127 - i;
	}

	response_curve_solver( small_b, exposure_log2, CONSTANTL, w, &gb, pic_count );
	response_curve_solver( small_g, exposure_log2, CONSTANTL, w, &gg, pic_count );
	response_curve_solver( small_r, exposure_log2, CONSTANTL, w, &gr, pic_count );

	cout << "done" << endl;
	
	/* ------------ solve response curves ------------ */
	unsigned img_size = row * col;
	float hdr[ img_size * 3 ] = {0};
	cout << "Constructing radiance map for Blue channel .... " << endl;
	construct_radiance_map( img_size, pic_count, 0, gb, img_list_b, exposure_log2, w, hdr );
	cout << "Constructing radiance map for Green channel .... " << endl;
	construct_radiance_map( img_size, pic_count, 1, gg, img_list_g, exposure_log2, w, hdr );
	cout << "Constructing radiance map for Red channel .... " << endl;
	construct_radiance_map( img_size, pic_count, 2, gr, img_list_r, exposure_log2, w, hdr );
	cout << "done" << endl;

	/* ------------ Saving HDR image ------------ */
	ofstream f;
	f.open( output_name, ios::out | ios::binary );
	if( f.is_open() ){
		{
		string buffer = "#?RADIANCE\n# Made C++\nFORMAT=32-bit_rle_rgbe\n\n";
		f.write( buffer.c_str(), buffer.size() );
		}
		{
		string buffer = "";
		buffer.append( "-Y " ).append( to_string( row ) ).append( " +X " ).append( to_string( col ) ).append( "\n" );
		f.write( buffer.c_str(), buffer.size() );
		}

		// find max bright value
		cout << img_size << endl;
		//uint8_t rbge[ img_size * 4 ] = {0};
		for( unsigned i = 0; i < img_size; ++i ){
			float brightest;
			float mantissa;
			int exponent;
			//cout << hdr[ i * 3 ] << " " << hdr[ i * 3 + 1 ] << " " << hdr[ i * 3 + 2 ] << " " << endl;
			brightest = hdr[ i * 3 ];
			if( brightest < hdr[ i * 3 + 1 ] ) brightest = hdr[ i * 3 + 1 ];
			if( brightest < hdr[ i * 3 + 2 ] ) brightest = hdr[ i * 3 + 2 ];
			mantissa = frexpf( brightest, &exponent );
			// reuse mantissa for scaled mantissa
			mantissa = mantissa * 256.0 / brightest;
			//rbge[ i * 4 + 0 ] = ( uint8_t )round( hdr[ i * 3 + 2 ] * mantissa[i] );
			//rbge[ i * 4 + 1 ] = ( uint8_t )round( hdr[ i * 3 + 1 ] * mantissa[i] );
			//rbge[ i * 4 + 2 ] = ( uint8_t )round( hdr[ i * 3 + 0 ] * mantissa[i] );
			//rbge[ i * 4 + 3 ] = ( uint8_t )round( exponent[i] + 128 );
			f.put( ( uint8_t )round( hdr[ i * 3 + 2 ] * mantissa ) );
			f.put( ( uint8_t )round( hdr[ i * 3 + 1 ] * mantissa ) );
			f.put( ( uint8_t )round( hdr[ i * 3 + 0 ] * mantissa ) );
			f.put( ( uint8_t )round( exponent + 128 ) );
		}
	}
	else{
		cout << "Error creating file" << endl;
	}
	f.close();
	
	delete[] img_list_b;
	delete[] img_list_g;
	delete[] img_list_r;

	delete[] small_b;
	delete[] small_g;
	delete[] small_r;

	delete[] exposure_log2;

	delete[] w;
}
