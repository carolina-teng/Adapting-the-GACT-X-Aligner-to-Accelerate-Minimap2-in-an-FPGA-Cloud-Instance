#!/bin/bash
sudo yum -y install cmake
sudo yum -y install zlib-devel zlib-static
git clone https://github.com/aws/aws-fpga.git $AWS_FPGA_REPO_DIR
source $AWS_FPGA_REPO_DIR/vitis_setup.sh
git clone https://github.com/gsneha26/Darwin-WGA.git
export PROJECT_DIR=$PWD/Darwin-WGA
cd $PROJECT_DIR
./scripts/create_xo.sh
./scripts/create_GACTX.hw.sh
cd test_GACTX_hw
../../aws-fpga/Vitis/tools/create_vitis_afi.sh -xclbin=GACTX.hw.xclbin -o=GACTX.hw -s3_bucket=gactx -s3_dcp_key=dcp_folder -s3_logs_key=logs_folder