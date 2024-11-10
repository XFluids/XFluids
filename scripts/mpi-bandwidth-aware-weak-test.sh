#!/bin/bash
isaware=on	#'on' or 'off', default is 'on'
max_gcs=32
max_mpi_rank=6
repeat_times=10
inner_size='16,256,256'
ghost_size=8

if [ $isaware == 'on' ]; then
    mpi_params='--mca pml ucx'
else
    mpi_params=''
fi
echo 'isaware, inner_size_x, inner_size_y, inner_size_z, filename, ghost size, ghost size_y, ghost size_z, mpi rank, repeat time, device_mem, mpi_mem, total_time, mpi_time' >> mpi-aware-$isaware.csv
#begin ghost cell size(MPI message size)
gcs_=4
while [ $gcs_ -le $max_gcs ]
do

	#begin mpi rank loop
	mmr=1
	while [ $mmr -le $max_mpi_rank ]
	do

		#begin repeat times loop
		i=1
		while [ $i -le $repeat_times ]
		do

			output_file=mpi-aware-$isware-gcs.$gcs_-$mmr-1-1.$i.log
			mpirun -n $mmr $mpi_params ../build/XFLUIDS -mpi=$mmr,1,1 -gcs=$gcs_,$ghost_size,$ghost_size -run=$inner_size -dev=$max_mpi_rank,1,0 > $output_file  &
			# 获取程序的进程 ID
			PID=$!
			# 等待程序返回结果
			wait $PID

			# 性能数据读取
			# 提取每个标识符的值
			total_time=$(grep '^MPI averaged of' "$output_file" | cut -d ':' -f 2)
			device_mem=$(grep '^Device Memory Usage(GB)' "$output_file" | cut -d ':' -f 2)
			mpi_mem=$(grep '^MPI trans Memory Size(GB)' "$output_file" | cut -d ':' -f 2)
			mpi_time=$(grep '^MPI buffers Trans time(s)' "$output_file" | cut -d ':' -f 2)
			echo $isaware, $inner_size, $output_file, $gcs_, $ghost_size, $ghost_size, $mmr, $i, $device_mem, $mpi_mem, $total_time, $mpi_time >> mpi-aware-$isaware.csv

		i=$((i+1))
		done

	mmr=$((mmr+1))
	done

gcs_=$((gcs_+4))
done
