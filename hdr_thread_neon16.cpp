#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "lapacke_utils.h"
#include <iterator>
#include <iostream>
#include <dirent.h>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <thread>
#include <chrono>
#include <limits>
#include <vector>
#include <math.h>

#define CONSTANTL 50
#define SMALLDIM 10
#define SMALLPIXELS 100
#define TILESIZE 16

using namespace std;
using namespace cv;

void load_exposures( string source_dir, uint8_t **img_list_b, uint8_t **img_list_g, uint8_t **img_list_r, uint8_t **small_b, uint8_t **small_g, uint8_t **small_r, int **exposure_log2, unsigned int *row_input, unsigned int *col_input, unsigned *pic_count ){
	fstream txt;
	txt.open( source_dir + "/image_list.txt", fstream::in );
	if( !txt ){
		fprintf(stderr, "no image_list.txt found\n");
		exit( EXIT_FAILURE );
	}
	else{
		string temp;
		unsigned pointer = 0;
		unsigned distance = 0;
		getline( txt, temp );
		( *pic_count ) = atoi( temp.c_str() );
		while( getline( txt, temp ) ){
			if( temp[0] != '#' ){
				istringstream iss( source_dir + "/" + temp );
				iss >> temp;
				Mat input_pic = imread( temp, CV_LOAD_IMAGE_COLOR);
				if( input_pic.data ){
					if( ( *row_input ) == 0 && ( *col_input ) == 0 ){
						( *row_input ) = input_pic.rows;
						( *col_input ) = input_pic.cols;
						distance = ( *row_input ) * ( *col_input );
						*img_list_b = new uint8_t[ ( *row_input ) * ( *col_input ) * ( *pic_count ) ];
						*img_list_g = new uint8_t[ ( *row_input ) * ( *col_input ) * ( *pic_count ) ];
						*img_list_r = new uint8_t[ ( *row_input ) * ( *col_input ) * ( *pic_count ) ];
						*small_b = new uint8_t[ SMALLPIXELS * ( *pic_count ) ];
						*small_g = new uint8_t[ SMALLPIXELS * ( *pic_count ) ];
						*small_r = new uint8_t[ SMALLPIXELS * ( *pic_count ) ];
						*exposure_log2 = new int[ ( *pic_count ) ];
					}

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

					float exposure;
					iss >> exposure;
					( *exposure_log2 )[ pointer ] = log2( exposure );

					pointer++;
				}
			}
		}
	}
	txt.close();
	return;
}

void response_curve_solver( uint8_t *Z, int *B, int l, uint8_t *w, double **g, int pic_count ){
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
		cerr << "The algorithm computing SVD failed to converge;" << endl;
		cerr << "the least squares solution could not be computed." << endl;
		exit( EXIT_FAILURE );
	}

	*g = new double[ 256 ];
	memcpy( ( *g ), b, 256 * sizeof( double ) );

	delete[] temp;
}

void construct_radiance_map( int img_size, int pic_count, int offset, uint8_t thread_count, uint8_t thread_id, double *g, uint8_t *Z, int *ln_t, uint8_t *w, float *ln_E ){
	for( int i = TILESIZE * thread_id; i < img_size; i += TILESIZE * thread_count ){
		float32x4_t neon_acc_w1 = vdupq_n_f32( 0.0 );
		float32x4_t neon_acc_w2 = vdupq_n_f32( 0.0 );
		float32x4_t neon_acc_w3 = vdupq_n_f32( 0.0 );
		float32x4_t neon_acc_w4 = vdupq_n_f32( 0.0 );
		float32x4_t neon_acc_E1 = vdupq_n_f32( 0.0 );
		float32x4_t neon_acc_E2 = vdupq_n_f32( 0.0 );
		float32x4_t neon_acc_E3 = vdupq_n_f32( 0.0 );
		float32x4_t neon_acc_E4 = vdupq_n_f32( 0.0 );
		for( int j = 0; j < pic_count; ++j ){
			uint8_t z[ TILESIZE ];
			memcpy( z, Z + j * img_size + i, TILESIZE * sizeof( uint8_t ) );

			float    temp_w[ TILESIZE ];
			float    temp_g[ TILESIZE ];

			for( int k = 0; k < TILESIZE; ++k ){
				   temp_w[ k ] = w[ z[ k ] ];
				   temp_g[ k ] = g[ z[ k ] ];
			}

			float32x4_t neon_temp_w1    = vld1q_f32  ( temp_w );
			float32x4_t neon_temp_w2    = vld1q_f32  ( temp_w + 4 );
			float32x4_t neon_temp_w3    = vld1q_f32  ( temp_w + 8 );
			float32x4_t neon_temp_w4    = vld1q_f32  ( temp_w + 12 );
			float32x4_t neon_temp_g1    = vld1q_f32  ( temp_g );
			float32x4_t neon_temp_g2    = vld1q_f32  ( temp_g + 4 );
			float32x4_t neon_temp_g3    = vld1q_f32  ( temp_g + 8 );
			float32x4_t neon_temp_g4    = vld1q_f32  ( temp_g + 12 );
			float32x4_t neon_temp_ln_t  = vdupq_n_f32( ln_t[ j ] );

			neon_temp_g1 = vsubq_f32( neon_temp_g1, neon_temp_ln_t );   // ( g[ z[ k ] ] - ln_t[ j ] )
			neon_temp_g2 = vsubq_f32( neon_temp_g2, neon_temp_ln_t );   // ( g[ z[ k ] ] - ln_t[ j ] )
			neon_temp_g3 = vsubq_f32( neon_temp_g3, neon_temp_ln_t );   // ( g[ z[ k ] ] - ln_t[ j ] )
			neon_temp_g4 = vsubq_f32( neon_temp_g4, neon_temp_ln_t );   // ( g[ z[ k ] ] - ln_t[ j ] )
			neon_temp_g1 = vmulq_f32( neon_temp_g1, neon_temp_w1 );     // w[ z[ k ] ] * ( g[ z[ k ] ] - ln_t[ j ] )
			neon_temp_g2 = vmulq_f32( neon_temp_g2, neon_temp_w2 );     // w[ z[ k ] ] * ( g[ z[ k ] ] - ln_t[ j ] )
			neon_temp_g3 = vmulq_f32( neon_temp_g3, neon_temp_w3 );     // w[ z[ k ] ] * ( g[ z[ k ] ] - ln_t[ j ] )
			neon_temp_g4 = vmulq_f32( neon_temp_g4, neon_temp_w4 );     // w[ z[ k ] ] * ( g[ z[ k ] ] - ln_t[ j ] )

			neon_acc_E1  = vaddq_f32( neon_temp_g1, neon_acc_E1 );      // acc_E[ i + k ] += w[ z[ k ] ] * ( g[ z[ k ] ] - ln_t[ j ] );
			neon_acc_E2  = vaddq_f32( neon_temp_g2, neon_acc_E2 );      // acc_E[ i + k ] += w[ z[ k ] ] * ( g[ z[ k ] ] - ln_t[ j ] );
			neon_acc_E3  = vaddq_f32( neon_temp_g3, neon_acc_E3 );      // acc_E[ i + k ] += w[ z[ k ] ] * ( g[ z[ k ] ] - ln_t[ j ] );
			neon_acc_E4  = vaddq_f32( neon_temp_g4, neon_acc_E4 );      // acc_E[ i + k ] += w[ z[ k ] ] * ( g[ z[ k ] ] - ln_t[ j ] );
			neon_acc_w1  = vaddq_f32( neon_temp_w1, neon_acc_w1 );      // acc_w[ k ]     += w[ z[ k ] ];
			neon_acc_w2  = vaddq_f32( neon_temp_w2, neon_acc_w2 );      // acc_w[ k ]     += w[ z[ k ] ];
			neon_acc_w3  = vaddq_f32( neon_temp_w3, neon_acc_w3 );      // acc_w[ k ]     += w[ z[ k ] ];
			neon_acc_w4 = vaddq_f32( neon_temp_w4, neon_acc_w4 ); // acc_w[ k ]     += w[ z[ k ] ];
		}
		float acc_w[ TILESIZE ] = {0};
		float acc_E[ TILESIZE ] = {0};
		vst1q_f32 (       acc_E, neon_acc_E1 );
		vst1q_f32 (   acc_E + 4, neon_acc_E2 );
		vst1q_f32 (   acc_E + 8, neon_acc_E3 );
		vst1q_f32 (  acc_E + 12, neon_acc_E4 );
		vst1q_f32 (       acc_w, neon_acc_w1 );
		vst1q_f32 (   acc_w + 4, neon_acc_w2 );
		vst1q_f32 (   acc_w + 8, neon_acc_w3 );
		vst1q_f32 (  acc_w + 12, neon_acc_w4 );
		for( int k = 0; k < TILESIZE; ++k ){
			ln_E[ ( i + k ) * 3 + offset ] = ( acc_w[ k ] > 0 )? exp( acc_E[ ( k ) ] / acc_w[ k ] ) : exp( acc_E[ ( k ) ] );
		}
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
		cerr << "[Usage] hdr <input img dir> <output .hdr name> <thread count> \n[Example] hdr taipei taipei.hdr" << endl;
		return 0;
	}
	string img_dir = argv[1];
	string output_name = argv[2];
	uint8_t thread_total = atoi( argv[3] );

	/* ------------ count pictures in folder ------------ */

	row = col = 0;

	/* ------------ load picture and small reference input ------------ */
	cout << "reading input images ... " << endl;
	auto start = std::chrono::high_resolution_clock::now();
	load_exposures( img_dir, &img_list_b, &img_list_g, &img_list_r, &small_b, &small_g, &small_r, &exposure_log2, &row, &col, &pic_count );
	auto finish = std::chrono::high_resolution_clock::now();
	cout << "done in : " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";

	/* ------------ solve response curves ------------ */
	cout << "Solving response curves ... " << endl;
	start = std::chrono::high_resolution_clock::now();

	uint8_t *w = new uint8_t[ 256 ];
	for( int i = 0; i < 128; ++i ){
		w[       i ] = i;
		w[ i + 128 ] = 127 - i;
	}

	response_curve_solver( small_b, exposure_log2, CONSTANTL, w, &gb, pic_count );
	response_curve_solver( small_g, exposure_log2, CONSTANTL, w, &gg, pic_count );
	response_curve_solver( small_r, exposure_log2, CONSTANTL, w, &gr, pic_count );

	finish = std::chrono::high_resolution_clock::now();
	cout << "done in : " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";

	/* ------------ solve response curves ------------ */
	start = std::chrono::high_resolution_clock::now();
	unsigned img_size = row * col;
	float *hdr = new float[ img_size * 3 ];
	std::thread threads[thread_total];
	
	cout << "Constructing radiance map for Blue channel .... " << endl;
	for( uint8_t i = 0; i < thread_total; ++i ){
		threads[i] = std::thread( construct_radiance_map, img_size, pic_count, 0, thread_total, i, gb, img_list_b, exposure_log2, w, hdr );
	}
	
	for (auto& t: threads) {
		t.join();
	}
	
	cout << "Constructing radiance map for Green channel .... " << endl;
	for( uint8_t i = 0; i < thread_total; ++i ){
		threads[i] = std::thread( construct_radiance_map, img_size, pic_count, 1, thread_total, i, gg, img_list_g, exposure_log2, w, hdr );
	}
	
	for (auto& t: threads) {
		t.join();
	}

	cout << "Constructing radiance map for Red channel .... " << endl;
	for( uint8_t i = 0; i < thread_total; ++i ){
		threads[i] = std::thread( construct_radiance_map, img_size, pic_count, 2, thread_total, i, gr, img_list_r, exposure_log2, w, hdr );
	}
	
	for (auto& t: threads) {
		t.join();
	}

	finish = std::chrono::high_resolution_clock::now();
	cout << "done in : " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";

	start = std::chrono::high_resolution_clock::now();
	cout << "Writing hdr image .... " << endl;
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
		//uint8_t rbge[ img_size * 4 ] = {0};
		for( unsigned i = 0; i < img_size; ++i ){
			float brightest;
			float mantissa;
			int exponent;
			brightest = hdr[ i * 3 ];
			if( brightest < hdr[ i * 3 + 1 ] ) brightest = hdr[ i * 3 + 1 ];
			if( brightest < hdr[ i * 3 + 2 ] ) brightest = hdr[ i * 3 + 2 ];
			mantissa = frexpf( brightest, &exponent );
			// reuse mantissa for scaled mantissa
			mantissa = mantissa * 256.0 / brightest;
			f.put( ( uint8_t )round( hdr[ i * 3 + 2 ] * mantissa ) );	//rbge[ i * 4 + 0 ] = ( uint8_t )round( hdr[ i * 3 + 2 ] * mantissa[i] );
			f.put( ( uint8_t )round( hdr[ i * 3 + 1 ] * mantissa ) );	//rbge[ i * 4 + 1 ] = ( uint8_t )round( hdr[ i * 3 + 1 ] * mantissa[i] );
			f.put( ( uint8_t )round( hdr[ i * 3 + 0 ] * mantissa ) );	//rbge[ i * 4 + 2 ] = ( uint8_t )round( hdr[ i * 3 + 0 ] * mantissa[i] );
			f.put( ( uint8_t )round( exponent + 128 ) );				//rbge[ i * 4 + 3 ] = ( uint8_t )round( exponent[i] + 128 );
		}
	}
	else{
		cout << "Error creating file" << endl;
	}
	f.close();
	finish = std::chrono::high_resolution_clock::now();
	cout << "done in : " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";

	delete[] img_list_b;
	delete[] img_list_g;
	delete[] img_list_r;

	delete[] small_b;
	delete[] small_g;
	delete[] small_r;
	delete[] gb;
	delete[] gg;
	delete[] gr;

	delete[] exposure_log2;

	delete[] w;

	delete[] hdr;
}
