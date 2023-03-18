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
#include <iostream>

#include "minimap.h"
#include "ksw2.h"

#include <chrono>
using namespace std::chrono;
using namespace std;

extern pthread_mutex_t mutex;

int kernel_expanding[NUM_KERNELS];
//extern long long int kernel_duration[NUM_KERNELS];
//extern long long int kernel_process_duration[NUM_KERNELS];
//extern long long int kernel_write_transfer_duration[NUM_KERNELS];
//extern long long int kernel_read_transfer_duration[NUM_KERNELS];

//#include <fstream>
//std::ofstream Bootstamps;

extern cl_context context;            				// compute context
extern cl_command_queue commands;					// compute command queue
extern cl_program program;           				// compute programs
extern cl_kernel kernel[NUM_KERNELS];				// compute kernel
extern cl_mem_ext_ptr_t bank3_ext[NUM_KERNELS];
extern cl_mem ref_seq[NUM_KERNELS];
extern cl_mem query_seq[NUM_KERNELS];
extern cl_mem tile_output[NUM_KERNELS];
extern cl_mem tb_output[NUM_KERNELS];
extern cl_event write_read_event;

extern unsigned char*h_ref_seq_input_3[NUM_KERNELS];
extern unsigned char*h_query_seq_input_3[NUM_KERNELS];
extern unsigned int*h_tile_output_3[NUM_KERNELS];
extern unsigned int*h_tb_output_3[NUM_KERNELS];

#if defined(SDX_PLATFORM) && !defined(TARGET_DEVICE)
#define STR_VALUE(arg)      #arg
#define GET_STRING(name) STR_VALUE(name)
#define TARGET_DEVICE GET_STRING(SDX_PLATFORM)
#endif

#define TB_MASK ((1 << 2)-1)
#define Z 0
#define I 1
#define D 2
#define M 3

extern "C" void mm_append_cigar(mm_reg1_t *r, uint32_t n_cigar, uint32_t *cigar);
extern void fpga_configuration_and_setup();
extern void fpga_shutdown_and_cleanup();

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

void fpga_configuration_and_setup()
{
	char xclbin[32] = "GACTX_parallelized.hw.awsxclbin";
	int err;
	
	//////////////////////////////////////////////////////////////////////////
	// Configuration and Setup												//
	//////////////////////////////////////////////////////////////////////////
	
	cl_platform_id platform_id;     // platform id
	cl_device_id device_id;         // compute device id
	
	char cl_platform_vendor[1001];
	char target_device_name[1001] = TARGET_DEVICE;
	
	// Get all platforms and then select Xilinx platform
	cl_platform_id platforms[16];       // platform id
	cl_uint platform_count;
	int platform_found = 0;
	err = clGetPlatformIDs(16, platforms, &platform_count);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error: Failed to find an OpenCL platform!\n");
		//return NULL;
	}
	
	// Find Xilinx Platform
	for (unsigned int iplat=0; iplat<platform_count; iplat++) {
		err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,NULL);
		if (err != CL_SUCCESS) {
			fprintf(stderr, "Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
			//return NULL;
		}
		if (strcmp(cl_platform_vendor, "Xilinx") == 0) {
			platform_id = platforms[iplat];
			platform_found = 1;
		}
	}
	if (!platform_found) {
		fprintf(stderr, "ERROR: Platform Xilinx not found. Exit.\n");
		//return NULL;
	}
	
	// Get accelerator compute device
	cl_uint num_devices;
	unsigned int device_found = 0;
	cl_device_id devices[16];  // compute device id
	char cl_device_name[1001];
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 16, devices, &num_devices);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "ERROR: Failed to create a device group!\n");
		//return NULL;
	}
	
	// Iterate all devices to select the target device.
	for (uint i=0; i<num_devices; i++) {
		err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, cl_device_name, 0);
		if (err != CL_SUCCESS) {
			fprintf(stderr, "Error: Failed to get device name for device %d!\n", i);
			//return NULL;
		}
		if(strcmp(cl_device_name, target_device_name) == 0) {
			device_id = devices[i];
			device_found = 1;
	   }
	}
	if (!device_found) {
		fprintf(stderr, "Target device %s not found. Exit.\n", target_device_name);
		//return NULL;
	}
	
	// Create a compute context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context) {
		fprintf(stderr, "Error: Failed to create a compute context!\n");
		//return NULL;
	}
	
	commands = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err); //CL_QUEUE_PROFILING_ENABLE CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
	if (!commands) {
		fprintf(stderr, "Error: Failed to create a command commands!\n");
		fprintf(stderr, "Error: code %i\n",err);
		//return NULL;
	}
	
	// Load binary from disk
	unsigned char *kernelbinary;
	int n_i0 = load_file_to_memory(xclbin, (char **) &kernelbinary);
	if (n_i0 < 0) {
		fprintf(stderr, "failed to load kernel from xclbin: %s\n", xclbin);
		//return NULL;
	}
	
	// Create the compute program from offline
	int status;
	size_t n0 = n_i0;
	program = clCreateProgramWithBinary(context, 1, &device_id, &n0, (const unsigned char **) &kernelbinary, &status, &err);
	if ((!program) || (err!=CL_SUCCESS)) {
		fprintf(stderr, "Error: Failed to create compute program from binary %d!\n", err);
		//return NULL;
	}
	
	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];

		fprintf(stderr, "Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		fprintf(stderr, "%s\n", buffer);
		//return NULL;
	}
	
	// Create the compute kernel in the program we wish to run
	std::string s0, s1;
	s0 = "GACTX_bank3";
	kernel[0] = clCreateKernel(program, s0.c_str(), &err);
	if (!(kernel[0]) || err != CL_SUCCESS) {
		fprintf(stderr, "Error: Failed to create compute kernel!\n");
		//return NULL;
	}
	if (NUM_KERNELS == 2){
		s1 = "GACTX_bank0";
		kernel[1] = clCreateKernel(program, s1.c_str(), &err);
		if (!(kernel[1]) || err != CL_SUCCESS) {
			fprintf(stderr, "Error: Failed to create compute kernel!\n");
			//return NULL;
		}
	}
	
	// Create structs to define memory bank mapping
	// number of memory banks is the same as number of kernels
	bank3_ext[0].flags = XCL_MEM_DDR_BANK3;
	bank3_ext[0].obj = NULL;
	bank3_ext[0].param = 0;
	
	if (NUM_KERNELS == 2){
		bank3_ext[1].flags = XCL_MEM_DDR_BANK0;
		bank3_ext[1].obj = NULL;
		bank3_ext[1].param = 0;
	}
	
	int max_seq_len = 500000;//25500;
	int tile_size = 4000;
	int max_max_tb = (2*tile_size)*2/32;
	
	for (int i = 0; i < NUM_KERNELS; i++) {
		ref_seq[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(char) * max_seq_len, &bank3_ext[i], NULL);
		query_seq[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(char) * max_seq_len, &bank3_ext[i], NULL);
		if (!(ref_seq[i]&&query_seq[i])) {
			fprintf(stderr, "Error: Failed to allocate device memory!\n");
			//return NULL;
		}
		
		h_ref_seq_input_3[i] = (unsigned char*)clEnqueueMapBuffer(commands, ref_seq[i], CL_TRUE, CL_MAP_WRITE, 0, sizeof(char) * max_seq_len, 0, NULL, &write_read_event, NULL);
		h_query_seq_input_3[i] = (unsigned char*)clEnqueueMapBuffer(commands, query_seq[i], CL_TRUE, CL_MAP_WRITE, 0, sizeof(char) * max_seq_len, 0, NULL, &write_read_event, NULL);
		
		tile_output[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(int) * 16, &bank3_ext[i], NULL);
		tb_output[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(int) * max_max_tb, &bank3_ext[i], NULL);
		if (!(tile_output[i]&&tb_output[i])) {
			fprintf(stderr, "Error: Failed to allocate device memory!\n");
			//return NULL;
		}
	}
	
}

void fpga_shutdown_and_cleanup()
{
	for (int i = 0; i < NUM_KERNELS; i++) {
		clReleaseMemObject(ref_seq[i]);
		clReleaseMemObject(query_seq[i]);
		clReleaseMemObject(tile_output[i]);
		clReleaseMemObject(tb_output[i]);
		clReleaseKernel(kernel[i]);
	}
	clReleaseCommandQueue(commands);
	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseEvent(write_read_event);
}

extern "C" void gactx_align(void *km, int r_len, int q_len, int qlen, const mm_idx_t *mi, int32_t rid, int32_t rs, int32_t rs0, int32_t rs1, int32_t *re, int32_t re0, int32_t *re1, uint8_t *qseq0[2], int32_t rev, int32_t qs, int32_t *qe, int32_t qe0, int32_t *qe1, mm_reg1_t *r, int which_kernel);

void gactx_align(void *km, int r_len, int q_len, int qlen, const mm_idx_t *mi, int32_t rid, int32_t rs, int32_t rs0, int32_t rs1, int32_t *re, int32_t re0, int32_t *re1, uint8_t *qseq0[2], int32_t rev, int32_t qs, int32_t *qe, int32_t qe0, int32_t *qe1, mm_reg1_t *r, int which_kernel)
{
	
	//long int epoch_start = high_resolution_clock::now().time_since_epoch().count();
	
	//cl_ulong start_time = 0, end_time = 0;
	
	uint8_t *tseq, *qseq;
	tseq = (uint8_t*)kmalloc(km, re0 - rs);
	
	bool begin_tb = false, in_gap = false;
	int tile_ref_len = 0, tile_query_len = 0, ref_offset = 0, query_offset = 0, score = 0, ref_pos = 0, query_pos = 0, num_tb = 0, tb_pos = 0, rp = 0, qp = 0, num_r_bases = 0, num_q_bases = 0, err = 0;
	std::string tile_CIGAR = "", CIGAR = "";
	
	int tile_size = 4000;
	int tile_overlap = 128;
	int align_fields = 0;
	
	// Scoring
	int sub_mat[11] = {2, -4, -4, -4, 2, -4, -4, 2, -4, 2, -1}; //{10, -20, -20, -20, 10, -20, -20, 10, -20, 10, -5};
	int gap_open = -6; //-30;
	int gap_extend = -2; //-10;
	int ydrop = 189; //943;
	
	//////////////////////////////////////////////////////////////////////////
	// Begin Processing														//
	//////////////////////////////////////////////////////////////////////////
	
	mm_idx_getseq(mi, rid, rs, re0, tseq);
	for (int k = 0; k < r_len; k++){
		if (tseq[k] == 0)
			h_ref_seq_input_3[which_kernel][k] = 'A';
		else if (tseq[k] == 1)
			h_ref_seq_input_3[which_kernel][k] = 'C';
		else if (tseq[k] == 2)
			h_ref_seq_input_3[which_kernel][k] = 'G';
		else if (tseq[k] == 3)
			h_ref_seq_input_3[which_kernel][k] = 'T';
	}
	
	qseq = &qseq0[rev][qs];
	for (int k = 0; k < q_len; k++){
		if (qseq[k] == 0)
			h_query_seq_input_3[which_kernel][k] = 'A';
		else if (qseq[k] == 1)
			h_query_seq_input_3[which_kernel][k] = 'C';
		else if (qseq[k] == 2)
			h_query_seq_input_3[which_kernel][k] = 'G';
		else if (qseq[k] == 3)
			h_query_seq_input_3[which_kernel][k] = 'T';
	}
	
	int max_max_tb = (2*tile_size)*2/32;
	
	//auto total_start = high_resolution_clock::now();
	
	// Tansfer ref and query sequences onto the DRAM
	err = clEnqueueWriteBuffer(commands, ref_seq[which_kernel], CL_FALSE, 0, sizeof(char) * r_len, h_ref_seq_input_3[which_kernel], 1, &write_read_event, &write_read_event);
	/*clGetEventProfilingInfo(write_read_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start_time,NULL);
	clGetEventProfilingInfo(write_read_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end_time,NULL);
	kernel_write_transfer_duration[which_kernel] += (long long int) (end_time - start_time);*/
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error: Failed to write to source array h_ref_seq[which_kernel]_input!\n");
		//return NULL;
	}
	
	err = clEnqueueWriteBuffer(commands, query_seq[which_kernel], CL_TRUE, 0, sizeof(char) * q_len, h_query_seq_input_3[which_kernel], 1, &write_read_event, &write_read_event);
	/*clGetEventProfilingInfo(write_read_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start_time,NULL);
	clGetEventProfilingInfo(write_read_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end_time,NULL);
	kernel_write_transfer_duration[which_kernel] += (long long int) (end_time - start_time);*/
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error: Failed to write to source array h_ref_seq[which_kernel]_input!\n");
		//return NULL;
	}
	
	// Process in tiles
	ref_offset = 0;
	query_offset = 0;
	CIGAR = "";
	
	while(kernel_expanding[which_kernel] == 1){
		
		// Set tile size
		tile_ref_len = std::min(tile_size, r_len - ref_offset);
		tile_query_len = std::min(tile_size, q_len - query_offset);
		
		// Set the arguments to our compute kernel
		err = 0;
		for (int i = 0; i < 11; i++) {                                                               
			err |= clSetKernelArg(kernel[which_kernel], i, sizeof(int), &sub_mat[i]);
		}
		err |= clSetKernelArg(kernel[which_kernel], 11, sizeof(int),   	&gap_open);
		err |= clSetKernelArg(kernel[which_kernel], 12, sizeof(int),   	&gap_extend);
		err |= clSetKernelArg(kernel[which_kernel], 13, sizeof(int),   	&ydrop); 
		err |= clSetKernelArg(kernel[which_kernel], 14, sizeof(uint),  	&align_fields);
		err |= clSetKernelArg(kernel[which_kernel], 15, sizeof(uint), 	&tile_ref_len);
		err |= clSetKernelArg(kernel[which_kernel], 16, sizeof(uint), 	&tile_query_len);
		err |= clSetKernelArg(kernel[which_kernel], 17, sizeof(ulong),	&ref_offset);
		err |= clSetKernelArg(kernel[which_kernel], 18, sizeof(ulong), 	&query_offset);
		err |= clSetKernelArg(kernel[which_kernel], 19, sizeof(cl_mem), &ref_seq[which_kernel]); 
		err |= clSetKernelArg(kernel[which_kernel], 20, sizeof(cl_mem), &query_seq[which_kernel]);
		err |= clSetKernelArg(kernel[which_kernel], 21, sizeof(cl_mem), &tile_output[which_kernel]);
		err |= clSetKernelArg(kernel[which_kernel], 22, sizeof(cl_mem), &tb_output[which_kernel]);
		if (err != CL_SUCCESS) {
			fprintf(stderr, "Error: Failed to set kernel arguments! %d\n", err);
			//return NULL;
		}
		
		// Execute the kernel over the entire range of our 1d input data set
		// using the maximum number of work group items for this device
		cl_event process_event;
		err = clEnqueueTask(commands, kernel[which_kernel], 1, &write_read_event, &process_event);
		clWaitForEvents(1, &process_event);
		/*clGetEventProfilingInfo(process_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start_time,NULL);
		clGetEventProfilingInfo(process_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end_time,NULL);
		kernel_process_duration[which_kernel] += (long long int) (end_time - start_time);*/
		clReleaseEvent(process_event);
		if (err) {
			fprintf(stderr, "Error: Failed to execute kernel! %d\n", err);
			//return NULL;
		}
		
		// Read back the results from the device
		h_tile_output_3[which_kernel] = (unsigned int*)clEnqueueMapBuffer(commands, tile_output[which_kernel], CL_FALSE, CL_MAP_READ, 0, sizeof(int) * 16, 1, &write_read_event, &write_read_event, &err);
		/*clGetEventProfilingInfo(write_read_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start_time,NULL);
		clGetEventProfilingInfo(write_read_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end_time,NULL);
		kernel_read_transfer_duration[which_kernel] += (long long int) (end_time - start_time);*/
		if (err != CL_SUCCESS) {
			fprintf(stderr, "Error: Failed to read output array! %d\n", err);
			//return NULL;
		}
		
		h_tb_output_3[which_kernel] = (unsigned int*)clEnqueueMapBuffer(commands, tb_output[which_kernel], CL_TRUE, CL_MAP_READ, 0, sizeof(int) * max_max_tb, 1, &write_read_event, &write_read_event, &err);
		/*clGetEventProfilingInfo(write_read_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start_time,NULL);
		clGetEventProfilingInfo(write_read_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end_time,NULL);
		kernel_read_transfer_duration[which_kernel] += (long long int) (end_time - start_time);*/
		if (err != CL_SUCCESS) {
			fprintf(stderr, "Error: Failed to read output array! %d\n", err);
			//return NULL;
		}
		
		// Retrieve tile results
		score = h_tile_output_3[which_kernel][0];
		ref_pos = h_tile_output_3[which_kernel][1];
		query_pos = h_tile_output_3[which_kernel][2];
		num_tb = h_tile_output_3[which_kernel][5]*16;
		
		tile_CIGAR = "";
		in_gap = false;
		
		tb_pos = 0;
		begin_tb = false;
		rp = ref_offset + ref_pos;
		qp = query_offset + query_pos;
		num_r_bases = 0;
		num_q_bases = 0;
		//Backwards traceback starts when on overlap border
		for (int i = 0; i < num_tb; i++) {
			uint32_t tb_ptr = h_tb_output_3[which_kernel][i];
			for(int j = 0; j < 16; j++){
				int dir = ((tb_ptr >> (2*j)) & TB_MASK);
				switch(dir) {
					case Z:
						break;
					case D: 
						if (begin_tb) {
							tile_CIGAR += 'D';
							tb_pos++;
							num_r_bases++;
						} else {
							if (in_gap) score -= gap_extend;
							else {
								score -= gap_open;
								in_gap = true;
							}
						}
						rp--;
						break;
					case I:
						if (begin_tb) {
							tile_CIGAR += 'I';
							tb_pos++;
							num_q_bases++;
						} else {
							if (in_gap) score -= gap_extend;
							else {
								score -= gap_open;
								in_gap = true;
							}
						}
						qp--;
						break;
					case M:
						if ((rp < (ref_offset + tile_size - tile_overlap)) && (qp < (query_offset + tile_size - tile_overlap))) {
							begin_tb = true;
						}
						if (begin_tb) {
							tile_CIGAR += 'M';
							tb_pos++;
							num_r_bases++;
							num_q_bases++;
						} else {
							in_gap = false;
							if (h_ref_seq_input_3[which_kernel][rp] == h_query_seq_input_3[which_kernel][qp]) score -= sub_mat[0];
							else score -= sub_mat[1];
						}
						rp--;
						qp--;
						break;
				}
			}
		}
		
		std::reverse(tile_CIGAR.begin(), tile_CIGAR.end());
		CIGAR += tile_CIGAR;
		
		// Prepare for next tile
		ref_offset += num_r_bases;
		query_offset += num_q_bases;
		r->p->dp_score += score;
		
		// Leave when reached negative tile max score or end of query sequences
		if ((num_tb == 0) || (ref_offset >= r_len) || (query_offset >= q_len) || (tile_query_len < tile_size)){
			/*long int epoch_end = high_resolution_clock::now().time_since_epoch().count();
			Bootstamps.open("Bootstamps.txt", std::ios_base::app);
			Bootstamps << epoch_start << endl << epoch_end << endl;
			Bootstamps.close();*/
			kernel_expanding[which_kernel] = 0;
		}
		
	}
	
	/*auto total_stop = high_resolution_clock::now();
	auto duration = duration_cast<nanoseconds>(total_stop - total_start);
	kernel_duration[which_kernel] += duration.count();*/
	
	//encode CIGAR as Minimap2
	if (CIGAR.length() != 0) {
		uint32_t minimap2_CIGAR[CIGAR.length()];
		int minimap2_n_CIGAR = 0;
		char current_flag = CIGAR[0];
		uint32_t current_count = 0;
		for (int i = 0; i < (int)CIGAR.length(); i++) {
			if (CIGAR[i] == current_flag){
				current_count += 1;}
			else{
				if (current_flag == 'M')
					minimap2_CIGAR[minimap2_n_CIGAR] = (current_count<<4) + 0;
				else if (current_flag == 'I')
					minimap2_CIGAR[minimap2_n_CIGAR] = (current_count<<4) + 1;
				else if (current_flag == 'D')
					minimap2_CIGAR[minimap2_n_CIGAR] = (current_count<<4) + 2;
				else if (current_flag == 'N')
					minimap2_CIGAR[minimap2_n_CIGAR] = (current_count<<4) + 4;
				current_flag = CIGAR[i];
				current_count = 1;
				minimap2_n_CIGAR += 1;
			}
		}
		if (current_flag == 'M')
			minimap2_CIGAR[minimap2_n_CIGAR] = (current_count<<4) + 0;
		else if (current_flag == 'I')
			minimap2_CIGAR[minimap2_n_CIGAR] = (current_count<<4) + 1;
		else if (current_flag == 'D')
			minimap2_CIGAR[minimap2_n_CIGAR] = (current_count<<4) + 2;
		else if (current_flag == 'N')
			minimap2_CIGAR[minimap2_n_CIGAR] = (current_count<<4) + 4;
		minimap2_n_CIGAR += 1;
		
		uint32_t minimap2_CIGAR_aux[minimap2_n_CIGAR];
		memcpy(minimap2_CIGAR_aux, minimap2_CIGAR, sizeof(uint32_t)*minimap2_n_CIGAR);
		
		mm_append_cigar(r, minimap2_n_CIGAR, minimap2_CIGAR_aux);
	}
	
	*re1 = std::min(rs + ref_offset, rs1 + (re0 - rs0));
	*qe1 = std::min(qs + query_offset, qlen);
	
	kfree(km, tseq);
	
}
