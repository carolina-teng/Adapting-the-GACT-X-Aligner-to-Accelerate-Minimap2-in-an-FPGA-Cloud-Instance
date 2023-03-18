# Adapting-the-GACT-X-Aligner-to-Accelerate-Minimap2-in-an-FPGA-Cloud-Instance

## Table of Contents

- [Methodology](#methodology)
- [Adaptations](#adaptations)
- [Instructions](#instructions)
- [Citing This Work](#citation_self)
- [Citing Minimap2](#citation_Minimap2)
- [Citing Darwin-WGA](#citation_Darwin_WGA)

## <a name="methodology"></a>Methodology

## <a name="adaptations"></a>Adaptations

* changed

## <a name="instructions"></a>Instructions

1. subscribe to FPGA Developer AMI
2. configure to Software version 1.5.1 (Oct 22, 2020) and Region US West (Oregon) or any region with F1 instances
3. launch on EC2 Instance Type f1.2xlarge
4. create and configure security group/key (optional)
5. modify EBS Volume to 100GB (optional but recommended for human data)
6. connect using ssh (Linux) or Puttygen (Windows)
7. extend partition to the volume increased
8. create bucket to store and share AFI (optional)
9. intall tmux for long runs (optional)
```
sudo yum -y install tmux
```
```
tmux new -s Minimap2_GACTX
```
10. configure with root user keys (required)
```
aws configure
```
11. install libraries
```
sudo yum -y install cmake
```
```
sudo yum -y install zlib-devel zlib-static
```
12. clone and commit to the AWS EC2 FPGA Hardware and Software Development Kit
```
git clone https://github.com/aws/aws-fpga.git
```
```
cd aws-fpga && git checkout 2fa6b0672de67d46d1ae21147c2fbaadceb34207 && git checkout -b new_branch && cd ..
```
13. run setup
```
export AWS_DIR=$PWD/aws-fpga && source $AWS_DIR/sdaccel_setup.sh
```
14. clone Darwin-WGA
```
git clone https://github.com/gsneha26/Darwin-WGA.git
```
```
export PROJECT_DIR=$PWD/Darwin-WGA && cd $PROJECT_DIR
```
15. create AFI
```
chmod 755 scripts/create_xo.sh && ./scripts/create_xo.sh
```
```
chmod 755 scripts/create_GACTX.hw.sh && ./scripts/create_GACTX.hw.sh
```
16. save AFI to bucket
```
cd test_GACTX_hw && ../../aws-fpga/SDAccel/tools/create_sdaccel_afi.sh -xclbin=GACTX.hw.xclbin -o=GACTX.hw -s3_bucket=gactx -s3_dcp_key=dcp_folder -s3_logs_key=logs_folder
```
17. wait for FPGA to become available
18. run setup
```
sudo sh
```
```
aws configure
```
```
export LD_LIBRARY_PATH=$XILINX_SDX/runtime/lib/x86_64/:$LD_LIBRARY_PATH && export XCL_EMULATION_MODE=hw && export VIVADO_TOOL_VERSION=2017.4 && source ../../aws-fpga/sdaccel_runtime_setup.sh
```
19. compile source
```
cmake -DCMAKE_BUILD_TYPE=Release -DAWS_PLATFORM=/home/centos/aws-fpga/SDAccel/aws_platform/xilinx_aws-vu9p-f1-04261818_dynamic_5_0/xilinx_aws-vu9p-f1-04261818_dynamic_5_0.xpfm -DXILINX_SDX=/opt/Xilinx/SDx/2017.4.op -DXILINX_VIVADO=/opt/Xilinx/Vivado/2017.4.op .
```
```
make
```
20. run accelerated Minimap2
```
./minimap2_gactx -ax [technology] -t [threads] [reference file] [FASTQ file] > [SAM file]
```

## <a name="citation_self"></a>Citing This Work

:clock2: Work in reviewing phase.

## <a name="citation_Minimap2"></a>Citing Minimap2

> Li, H. Minimap2: pairwise alignment for nucleotide sequences. Bioinformatics 2018, 34, 3094–3100. https://doi.org/10.1093/bioinformatics/bty191.

## <a name="citation_Darwin_WGA"></a>Citing Darwin-WGA

> Turakhia, Y., Goenka, S. D., Bejerano, G., & Dally, W. J. (2019, February). Darwin-WGA: A co-processor provides increased sensitivity in whole genome alignments with high speedup. In 2019 IEEE International Symposium on High Performance Computer Architecture (HPCA) (pp. 359-372). IEEE.
