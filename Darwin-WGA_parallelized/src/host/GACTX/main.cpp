/*
MIT License

Copyright (c) 2019 Sneha D. Goenka, Yatish Turakhia, Gill Bejerano and William J. Dally

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <CL/opencl.h>
#include <CL/cl_ext.h>
#include "ConfigFile.h"
#include "Chameleon.h"
#include <iostream>
#include <fstream>

#include <chrono>
using namespace std::chrono;
using namespace std;

#define NUM_KERNELS 1

struct Configuration {
    // FASTA files
    std::string reference_name;
    std::string query_name;
    std::string reference_filename;
    std::string query_filename;

    // Scoring
    int sub_mat[11];
    int gap_open;
    int gap_extend;

    // GACT-X
    int ydrop;
};


#if defined(SDX_PLATFORM) && !defined(TARGET_DEVICE)
#define STR_VALUE(arg)      #arg
#define GET_STRING(name) STR_VALUE(name)
#define TARGET_DEVICE GET_STRING(SDX_PLATFORM)
#endif

#define TB_MASK (1 << 2)-1
#define Z 0
#define I 1
#define D 2
#define M 3

int load_file_to_memory(const char *filename, char **result){
    uint size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        *result = NULL;
        return -1; // -1 means file opening fail
    }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f)) {
        free(*result);
        return -2; // -2 means file reading fail
    }
    fclose(f);
    (*result)[size] = 0;
    return size;
}

int main(int argc, char** argv){
	
    if (argc != 2) {
        printf("Usage: %s xclbin\n", argv[0]);
        return EXIT_FAILURE;
    }
	
    char *xclbin = argv[1];
    int err;
	
    //////////////////////////////////////////////////////////////////////////
    // Configuration and Setup												//
    //////////////////////////////////////////////////////////////////////////
	
    cl_platform_id platform_id;         		// platform id
    cl_device_id device_id;             		// compute device id
    cl_context context;                 		// compute context
    cl_command_queue commands;					// compute command queue
    cl_program program;                			// compute programs
    cl_kernel kernel[NUM_KERNELS];         		// compute kernel
	
    char cl_platform_vendor[1001];
    char target_device_name[1001] = TARGET_DEVICE;
	
    // Get all platforms and then select Xilinx platform
    cl_platform_id platforms[16];       // platform id
    cl_uint platform_count;
    int platform_found = 0;
    err = clGetPlatformIDs(16, platforms, &platform_count);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to find an OpenCL platform!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
	
    // Find Xilinx Plaftorm
    for (unsigned int iplat=0; iplat<platform_count; iplat++) {
        err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,NULL);
        if (err != CL_SUCCESS) {
            printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }
        if (strcmp(cl_platform_vendor, "Xilinx") == 0) {
            platform_id = platforms[iplat];
            platform_found = 1;
        }
    }
    if (!platform_found) {
        printf("ERROR: Platform Xilinx not found. Exit.\n");
        return EXIT_FAILURE;
    }
	
    // Get accelerator compute device
    cl_uint num_devices;
    unsigned int device_found = 0;
    cl_device_id devices[16];  // compute device id
    char cl_device_name[1001];
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 16, devices, &num_devices);
    if (err != CL_SUCCESS) {
        printf("ERROR: Failed to create a device group!\n");
        printf("ERROR: Test failed\n");
        return -1;
    }
	
    // Iterate all devices to select the target device.
    for (uint i=0; i<num_devices; i++) {
        err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, cl_device_name, 0);
        if (err != CL_SUCCESS) {
            printf("Error: Failed to get device name for device %d!\n", i);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }
        if(strcmp(cl_device_name, target_device_name) == 0) {
            device_id = devices[i];
            device_found = 1;
       }
    }
    if (!device_found) {
        printf("Target device %s not found. Exit.\n", target_device_name);
        return EXIT_FAILURE;
    }
	
    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
	
    // Create a command commands
	commands = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
	if (!commands) {
		fprintf(stderr, "Error: Failed to create a command commands!\n");
		fprintf(stderr, "Error: code %i\n",err);
		fprintf(stderr, "Test failed\n");
		return EXIT_FAILURE;
	}
	
    // Load binary from disk
	unsigned char *kernelbinary;
    int n_i0 = load_file_to_memory(xclbin, (char **) &kernelbinary);
    if (n_i0 < 0) {
        printf("failed to load kernel from xclbin: %s\n", xclbin);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
	
    // Create the compute program from offline
	int status;
	size_t n0 = n_i0;
    program = clCreateProgramWithBinary(context, 1, &device_id, &n0, (const unsigned char **) &kernelbinary, &status, &err);
    if ((!program) || (err!=CL_SUCCESS)) {
        printf("Error: Failed to create compute program from binary %d!\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
	
    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
	
    // Create the compute kernel in the program we wish to run
    std::string s0 = "GACTX_bank3";
	kernel[0] = clCreateKernel(program, s0.c_str(), &err);
	if (!kernel[0] || err != CL_SUCCESS) {
		fprintf(stderr, "Error: Failed to create compute kernel!\n");
		fprintf(stderr, "Test failed\n");
		return EXIT_FAILURE;
	}
	if (NUM_KERNELS == 2){
		std::string s1 = "GACTX_bank0";
		kernel[1] = clCreateKernel(program, s1.c_str(), &err);
		if (!kernel[1] || err != CL_SUCCESS) {
			fprintf(stderr, "Error: Failed to create compute kernel!\n");
			fprintf(stderr, "Test failed\n");
			return EXIT_FAILURE;
		}
	}
	
    // Create structs to define memory bank mapping
	// number of memory banks is the same as number of kernels
    cl_mem_ext_ptr_t bank3_ext[NUM_KERNELS];
	
    bank3_ext[0].flags = XCL_MEM_DDR_BANK3;
    bank3_ext[0].obj = NULL;
    bank3_ext[0].param = 0;
	if (NUM_KERNELS == 2){
		bank3_ext[1].flags = XCL_MEM_DDR_BANK0;
		bank3_ext[1].obj = NULL;
		bank3_ext[1].param = 0;
	}
	
    Configuration cfg;
    ConfigFile cfg_file("params.cfg");
	
    // FASTA files
    cfg.reference_name     = (std::string) cfg_file.Value("FASTA_files", "reference_name"); 
    cfg.reference_filename = (std::string) cfg_file.Value("FASTA_files", "reference_filename"); 
    cfg.query_name         = (std::string) cfg_file.Value("FASTA_files", "query_name"); 
    cfg.query_filename     = (std::string) cfg_file.Value("FASTA_files", "query_filename"); 
	
    // Scoring
    cfg.sub_mat[0]      = cfg_file.Value("Scoring", "sub_AA");
    cfg.sub_mat[1]      = cfg_file.Value("Scoring", "sub_AC");
    cfg.sub_mat[2]      = cfg_file.Value("Scoring", "sub_AG");
    cfg.sub_mat[3]      = cfg_file.Value("Scoring", "sub_AT");
    cfg.sub_mat[4]      = cfg_file.Value("Scoring", "sub_CC");
    cfg.sub_mat[5]      = cfg_file.Value("Scoring", "sub_CG");
    cfg.sub_mat[6]      = cfg_file.Value("Scoring", "sub_CT");
    cfg.sub_mat[7]      = cfg_file.Value("Scoring", "sub_GG");
    cfg.sub_mat[8]      = cfg_file.Value("Scoring", "sub_GT");
    cfg.sub_mat[9]      = cfg_file.Value("Scoring", "sub_TT");
    cfg.sub_mat[10]     = cfg_file.Value("Scoring", "sub_N");
    cfg.gap_open        = cfg_file.Value("Scoring", "gap_open");
    cfg.gap_extend      = cfg_file.Value("Scoring", "gap_extend");
    cfg.ydrop           = cfg_file.Value("GACTX_params", "ydrop");
	
	//////////////////////////////////////////////////////////////////////////
    // Begin Processing														//
    //////////////////////////////////////////////////////////////////////////
	
	bool expanding;
	bool kernel_expanding[2];
	bool begin_tb;
	
	int r_len[NUM_KERNELS];
	int q_len[NUM_KERNELS];
	int ref_len[NUM_KERNELS];
	int query_len[NUM_KERNELS];
	int ref_offset[NUM_KERNELS];
	int query_offset[NUM_KERNELS];
	int score[NUM_KERNELS];
	int ref_pos[NUM_KERNELS];
	int query_pos[NUM_KERNELS];
	int num_tb[NUM_KERNELS];
	int tb_pos;
	int rp;
	int qp;
	int num_r_bases;
	int num_q_bases;
	
	int tile_size = 4000;
    int tile_overlap = 128;
	int align_fields = 0;
	
	int threshold = 0;
	int max_lines = 420988; //simulated 420988 / ONT 458116 / PacBio 655008
	
	cl_event writereadevent;
	
	// Open input files
	std::ifstream infile;
	std::ofstream CIGARs;
	
	infile.open ("right_extend_data_simulated.txt");
	CIGARs.open ("gactx_data_simulated.txt");
	
	// Separate buffer space for the sequences
	int max_seq_len = 26000; //simulated 26000 / ONT 103000 / PacBio 33000
	std::string ref[NUM_KERNELS];
    std::string query[NUM_KERNELS];
	std::string blank;
	
	cl_mem ref_seq[NUM_KERNELS];
	cl_mem query_seq[NUM_KERNELS];
	unsigned char*h_ref_seq_input_3[NUM_KERNELS];
	unsigned char*h_query_seq_input_3[NUM_KERNELS];
	for (int b = 0; b < NUM_KERNELS;  b++) {
		ref_seq[b] = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,  sizeof(char) * max_seq_len, &bank3_ext[b], NULL);
		query_seq[b] = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,  sizeof(char) * max_seq_len, &bank3_ext[b], NULL);
		if (!(ref_seq[b]&&query_seq[b])) {
			printf("Error: Failed to allocate device memory!\n");
			printf("Test failed\n");
			return EXIT_FAILURE;
		}
		h_ref_seq_input_3[b] = (unsigned char*)clEnqueueMapBuffer(commands, ref_seq[0], CL_TRUE, CL_MAP_WRITE, 0, sizeof(char) * max_seq_len, 0, NULL, NULL, NULL);
		h_query_seq_input_3[b] = (unsigned char*)clEnqueueMapBuffer(commands, query_seq[0], CL_TRUE, CL_MAP_WRITE, 0, sizeof(char) * max_seq_len, 0, NULL, NULL, NULL);
	}
	
	// Separate buffer space for the tiles
	int max_max_tb = (2*tile_size)*2/32;
	
	cl_mem tile_output[NUM_KERNELS];
	cl_mem tb_output[NUM_KERNELS];
	unsigned int*h_tile_output_3[NUM_KERNELS];
	unsigned int*h_tb_output_3[NUM_KERNELS];
	for (int b = 0; b < NUM_KERNELS; b++) {
		tile_output[b] = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,  sizeof(int) * 16, &bank3_ext[b], NULL); //CL_MEM_EXT_PTR_XILINX
		tb_output[b] = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,  sizeof(int) * max_max_tb, &bank3_ext[b], NULL);
		if (!(tile_output[b]&&tb_output[b])) {
			printf("Error: Failed to allocate device memory!\n");
			printf("Test failed\n");
			return EXIT_FAILURE;
		}
	}
	
	//auto total_start = high_resolution_clock::now();
	
	// Process each line of input files
	for (int line_index = 1; line_index <= max_lines; line_index++) {
		
		// Update progress bar
		if (line_index >= threshold){
			cout << "\r" << std::flush;
			cout << "[";
			for (int count=1;count <= 100; ++count){
				if (count <= line_index*100/max_lines) cout << "#";
				else cout << " ";
			}
			cout << "] " << line_index*100/max_lines << "%";
			threshold = threshold + max_lines/100;
		}
		if (line_index == max_lines){
			cout << "\r" << std::flush;
			cout << "[####################################################################################################] 100%";
		}
		
		// Get ref and query sequences
		for (int b = 0; b < NUM_KERNELS; b++) {
			getline(infile, ref[b]);
			getline(infile, query[b]);
			getline(infile, blank);
			getline(infile, blank);
			
			r_len[b] = ref[b].length();
			q_len[b] = query[b].length();
			//cout << r_len[b] << " " << q_len[b] << endl;
			line_index += b + 3;
		}
		
		for (int b = 0; b < NUM_KERNELS; b++) {
			for(int k = 0; k < r_len[b]; k++) h_ref_seq_input_3[b][k] = ref[b][k];
			for(int k = 0; k < q_len[b]; k++) h_query_seq_input_3[b][k] = query[b][k];
		}
		
		// Tansfer ref and query sequences onto the DRAM
		err = 0;
		err |= clEnqueueWriteBuffer(commands, ref_seq[0], CL_FALSE, 0, sizeof(char) * max_seq_len, h_ref_seq_input_3[0], 0, NULL, &writereadevent);
		err |= clEnqueueWriteBuffer(commands, query_seq[0], CL_TRUE, 0, sizeof(char) * max_seq_len, h_query_seq_input_3[0], 1, &writereadevent, &writereadevent);
		if ((NUM_KERNELS == 2) && (r_len[1] > 0) && (q_len[1] > 0)){
			err |= clEnqueueWriteBuffer(commands, ref_seq[1], CL_FALSE, 0, sizeof(char) * max_seq_len, h_ref_seq_input_3[1], 1, &writereadevent, &writereadevent);
			err |= clEnqueueWriteBuffer(commands, query_seq[1], CL_TRUE, 0, sizeof(char) * max_seq_len, h_query_seq_input_3[1], 1, &writereadevent, &writereadevent);
		}
		if (err != CL_SUCCESS) {
			printf("Error: Failed to write to source array h_ref_seq_input!\n");
			printf("Test failed\n");
			return EXIT_FAILURE;
		}
		
		// Process in tiles
		expanding = true;
		kernel_expanding[0] = true;
		ref_offset[0] = 0;
		query_offset[0] = 0;
		if ((NUM_KERNELS == 2) && (r_len[1] > 0) && (q_len[1] > 0)){
			ref_offset[1] = 0;
			query_offset[1] = 0;
			kernel_expanding[1] = true;
		} else{
			kernel_expanding[1] = false;
		}
		
		std::string CIGAR[NUM_KERNELS];
		for (int b = 0; b < NUM_KERNELS; b++) {
			CIGAR[b] = "";
		}
		
		while(expanding){
			
			// Set tile size
			if (kernel_expanding[0]){
				ref_len[0] = std::min(tile_size, r_len[0] - ref_offset[0]);
				query_len[0] = std::min(tile_size, q_len[0] - query_offset[0]);
			}
			if (kernel_expanding[1]){
				ref_len[1] = std::min(tile_size, r_len[1] - ref_offset[1]);
				query_len[1] = std::min(tile_size, q_len[1] - query_offset[1]);
			}
			
			// Set the arguments to our compute kernel
			err = 0;
			if (kernel_expanding[0]){
				for (int i = 0; i < 11; i++) {                                                               
					err |= clSetKernelArg(kernel[0], i, sizeof(int), &cfg.sub_mat[i]);
				}
				err |= clSetKernelArg(kernel[0], 11, sizeof(int),   &cfg.gap_open);
				err |= clSetKernelArg(kernel[0], 12, sizeof(int),   &cfg.gap_extend);
				err |= clSetKernelArg(kernel[0], 13, sizeof(int),   &cfg.ydrop); 
				err |= clSetKernelArg(kernel[0], 14, sizeof(uint),  &align_fields);
				err |= clSetKernelArg(kernel[0], 15, sizeof(uint),  &ref_len[0]);
				err |= clSetKernelArg(kernel[0], 16, sizeof(uint),  &query_len[0]);
				err |= clSetKernelArg(kernel[0], 17, sizeof(ulong), &ref_offset[0]);
				err |= clSetKernelArg(kernel[0], 18, sizeof(ulong), &query_offset[0]);
				err |= clSetKernelArg(kernel[0], 19, sizeof(cl_mem),   &ref_seq[0]); 
				err |= clSetKernelArg(kernel[0], 20, sizeof(cl_mem),   &query_seq[0]);
				err |= clSetKernelArg(kernel[0], 21, sizeof(cl_mem),   &tile_output[0]);
				err |= clSetKernelArg(kernel[0], 22, sizeof(cl_mem),   &tb_output[0]);
			}
			if (kernel_expanding[1]){
				for (int i = 0; i < 11; i++) {                                                               
					err |= clSetKernelArg(kernel[1], i, sizeof(int), &cfg.sub_mat[i]);
				}
				err |= clSetKernelArg(kernel[1], 11, sizeof(int),   &cfg.gap_open);
				err |= clSetKernelArg(kernel[1], 12, sizeof(int),   &cfg.gap_extend);
				err |= clSetKernelArg(kernel[1], 13, sizeof(int),   &cfg.ydrop); 
				err |= clSetKernelArg(kernel[1], 14, sizeof(uint),  &align_fields);
				err |= clSetKernelArg(kernel[1], 15, sizeof(uint),  &ref_len[1]);
				err |= clSetKernelArg(kernel[1], 16, sizeof(uint),  &query_len[1]);
				err |= clSetKernelArg(kernel[1], 17, sizeof(ulong), &ref_offset[1]);
				err |= clSetKernelArg(kernel[1], 18, sizeof(ulong), &query_offset[1]);
				err |= clSetKernelArg(kernel[1], 19, sizeof(cl_mem),   &ref_seq[1]); 
				err |= clSetKernelArg(kernel[1], 20, sizeof(cl_mem),   &query_seq[1]);
				err |= clSetKernelArg(kernel[1], 21, sizeof(cl_mem),   &tile_output[1]);
				err |= clSetKernelArg(kernel[1], 22, sizeof(cl_mem),   &tb_output[1]);
			}
			if (err != CL_SUCCESS) {
				printf("Error: Failed to set kernel arguments! %d\n", err);
				printf("Test failed\n");
				return EXIT_FAILURE;
			}
			
			// Execute the kernel over the entire range of our 1d input data set
			// using the maximum number of work group items for this device
			err = 0;
			if (kernel_expanding[0]){
				err |= clEnqueueTask(commands, kernel[0], 0, NULL, NULL);
			}
			if (kernel_expanding[1]){
				err |= clEnqueueTask(commands, kernel[1], 0, NULL, NULL);
			}
			clFinish(commands);
			if (err) {
				printf("Error: Failed to execute kernel! %d\n", err);
				printf("Test failed\n");
				return EXIT_FAILURE;
			}
			
			// Read back the results from the device
			err = 0;
			if (kernel_expanding[0]){
				h_tile_output_3[0] = (unsigned int*)clEnqueueMapBuffer(commands, tile_output[0], CL_FALSE, CL_MAP_READ, 0, sizeof(int) * 16, 1, &writereadevent, &writereadevent, &err);
				h_tb_output_3[0] = (unsigned int*)clEnqueueMapBuffer(commands, tb_output[0], CL_TRUE, CL_MAP_READ, 0, sizeof(int) * max_max_tb, 1, &writereadevent, &writereadevent, &err);
			}
			if (kernel_expanding[1]){
				h_tile_output_3[1] = (unsigned int*)clEnqueueMapBuffer(commands, tile_output[1], CL_FALSE, CL_MAP_READ, 0, sizeof(int) * 16, 1, &writereadevent, &writereadevent, &err);
				h_tb_output_3[1] = (unsigned int*)clEnqueueMapBuffer(commands, tb_output[1], CL_TRUE, CL_MAP_READ, 0, sizeof(int) * max_max_tb, 1, &writereadevent, &writereadevent, &err);
			}
			if (err != CL_SUCCESS) {
				printf("Error: Failed to read output array! %d\n", err);
				printf("Test failed\n");
				return EXIT_FAILURE;
			}
			
			// Retrieve tile results
			if (kernel_expanding[0]){
				score[0] = h_tile_output_3[0][0];
				ref_pos[0] = h_tile_output_3[0][1];
				query_pos[0] = h_tile_output_3[0][2];
				num_tb[0] = h_tile_output_3[0][5]*16;
			}
			if (kernel_expanding[1]){
				score[1] = h_tile_output_3[1][0];
				ref_pos[1] = h_tile_output_3[1][1];
				query_pos[1] = h_tile_output_3[1][2];
				num_tb[1] = h_tile_output_3[1][5]*16;
			}
			
			std::string tile_CIGAR[NUM_KERNELS];
			for (int b = 0; b < NUM_KERNELS; b++) {
				tile_CIGAR[b] = "";
			}
			
			if (kernel_expanding[0]){
				tb_pos = 0;
				begin_tb = false;
				rp = ref_offset[0] + ref_pos[0];
				qp = query_offset[0] + query_pos[0];
				num_r_bases = 0;
				num_q_bases = 0;
				//Backwards traceback starts when on overlap border
				for (int i = 0; i < num_tb[0]; i++) {
					uint32_t tb_ptr = h_tb_output_3[0][i];
					for(int j = 0; j < 16; j++){
						int dir = ((tb_ptr >> (2*j)) & TB_MASK);
						switch(dir) {
							case Z:
								break;
							case D: 
								if (begin_tb) {
									tile_CIGAR[0] += 'D';
									tb_pos++;
									num_r_bases++;
								}
								rp--;
								break;
							case I:
								if (begin_tb) {
									tile_CIGAR[0] += 'I';
									tb_pos++;
									num_q_bases++;
								}
								qp--;
								break;
							case M:
								if ((rp < (ref_offset[0] + tile_size - tile_overlap)) && (qp < (query_offset[0] + tile_size - tile_overlap))) {
									begin_tb = true;
								}
								if (begin_tb) {
									tile_CIGAR[0] += 'M';
									tb_pos++;
									num_r_bases++;
									num_q_bases++;
								}
								rp--;
								qp--;
								break;
						}
					}
				}
				
				std::reverse(tile_CIGAR[0].begin(), tile_CIGAR[0].end());
				CIGAR[0] += tile_CIGAR[0];
				
				// Prepare for next tile
				ref_offset[0] += num_r_bases;
				query_offset[0] += num_q_bases;
				
				// Leave when reached negative tile max score or end of query sequences
				if ((num_tb[0] == 0) || (ref_offset[0] >= r_len[0]) || (query_offset[0] >= q_len[0]) || (query_len[0] < tile_size))
					kernel_expanding[0] = false;
			}
			
			if (kernel_expanding[1]){
				tb_pos = 0;
				begin_tb = false;
				rp = ref_offset[1] + ref_pos[1];
				qp = query_offset[1] + query_pos[1];
				num_r_bases = 0;
				num_q_bases = 0;
				//Backwards traceback starts when on overlap border
				for (int i = 0; i < num_tb[1]; i++) {
					uint32_t tb_ptr = h_tb_output_3[1][i];
					for(int j = 0; j < 16; j++){
						int dir = ((tb_ptr >> (2*j)) & TB_MASK);
						switch(dir) {
							case Z:
								break;
							case D: 
								if (begin_tb) {
									tile_CIGAR[1] += 'D';
									tb_pos++;
									num_r_bases++;
								}
								rp--;
								break;
							case I:
								if (begin_tb) {
									tile_CIGAR[1] += 'I';
									tb_pos++;
									num_q_bases++;
								}
								qp--;
								break;
							case M:
								if ((rp < (ref_offset[1] + tile_size - tile_overlap)) && (qp < (query_offset[1] + tile_size - tile_overlap))) {
									begin_tb = true;
								}
								if (begin_tb) {
									tile_CIGAR[1] += 'M';
									tb_pos++;
									num_r_bases++;
									num_q_bases++;
								}
								rp--;
								qp--;
								break;
						}
					}
				}
				
				std::reverse(tile_CIGAR[1].begin(), tile_CIGAR[1].end());
				CIGAR[1] += tile_CIGAR[1];
				
				// Prepare for next tile
				ref_offset[1] += num_r_bases;
				query_offset[1] += num_q_bases;
				
				// Leave when reached negative tile max score or end of query sequences
				if ((num_tb[1] == 0) || (ref_offset[1] >= r_len[1]) || (query_offset[1] >= q_len[1]) || (query_len[1] < tile_size))
					kernel_expanding[1] = false;
			}
			
			if ((kernel_expanding[0] == false) && (kernel_expanding[1] == false))
				expanding = false;
			
		}
		
		// Write on output file
		CIGARs << CIGAR[0] << endl;
		CIGARs << "#" << endl;
		if ((NUM_KERNELS == 2) && (r_len[1] > 0) && (q_len[1] > 0)){
			CIGARs << CIGAR[1] << endl;
			CIGARs << "#" << endl;
		}
		
	}
	cout << endl;
	// Print total execution time
	/*auto total_stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(total_stop - total_start);
	cout << endl;
	cout << "Total execution time: " << duration.count() << " microseconds" << endl;*/
	
    //////////////////////////////////////////////////////////////////////////
    // Shutdown and Cleanup													//
    //////////////////////////////////////////////////////////////////////////
	infile.close();
	CIGARs.close();
	
	clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
	for (int b = 0; b < NUM_KERNELS; b++) {
		clReleaseMemObject(ref_seq[b]);
		clReleaseMemObject(query_seq[b]);
		clReleaseMemObject(tile_output[b]);
		clReleaseMemObject(tb_output[b]);
		clReleaseKernel(kernel[b]);
	}
}
