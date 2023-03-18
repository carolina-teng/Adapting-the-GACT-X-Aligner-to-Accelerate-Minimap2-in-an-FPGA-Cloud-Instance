#MIT License
#
#Copyright (c) 2019 Sneha D. Goenka, Yatish Turakhia, Gill Bejerano and William Dally
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

curr_dir=$PWD

rm -rf xclbin #delete directory xclbin and all its content
mkdir xclbin #create directory xclbin

/opt/Xilinx/Vivado/2018.2.op2258646/bin/vivado -mode batch -source ./scripts/gen_xo.tcl -tclargs xclbin/GACTX_bank3.xo GACTX $AWS_PLATFORM bank3 2

for i in `seq 0 0`;
do
	cd ./src/hdl/GACTX/top_modules/
	cp GACTX_bank3.v   GACTX_bank$i.v #copy file
	cp GACTX_bank3.xml GACTX_bank$i.xml #copy file
	sed -i "s/bank3/bank$i/g" GACTX_bank$i.v #substitute every occurrence of the expression in the file
	sed -i "s/bank3/bank$i/g" GACTX_bank$i.xml #substitute every occurrence of the expression in the file
	cd $curr_dir
	/opt/Xilinx/Vivado/2018.2.op2258646/bin/vivado -mode batch -source ./scripts/gen_xo.tcl -tclargs xclbin/GACTX_bank$i.xo GACTX $AWS_PLATFORM bank$i 2;
	rm ./src/hdl/GACTX/top_modules/GACTX_bank$i.v ./src/hdl/GACTX/top_modules/GACTX_bank$i.xml #delete auxiliar files
done

rm -rf packaged_kernel* tmp_kernel_pack* *.jou *.log *.wdb *.wcfg .Xil #delete files
