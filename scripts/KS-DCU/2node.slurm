#!/bin/bash
#SBATCH -J MPI-XFLUIDS			#指定作业名称
#SBATCH -p kshdnormal
#SBATCH -N 2  	 		#指定节点数量
#SBATCH --ntasks-per-node=4	#指定每节点的任务数量,默认每个任务一个 CPU 核心
#SBATCH --cpus-per-task=8	#指定每节点的任务数量,默认每个任务一个 CPU 核
#SBATCH --gres=dcu:4		#指定每节点申请的加速卡数量
##SBATCH --mem=20G		#指定每节点申请的内存大小,最大 100GB
#SBATCH -o ./o-1.%j.txt		#指定正确输出文件名称
#SBATCH -e ./e-1.%j.txt		#指定报错信息输出文件名称
#SBATCH --exclusive

echo "Start time: `date`"		#显示开始时间
echo "SLURM_JOB_ID: $SLURM_JOB_ID"			#显示作业号
echo "SLURM_NNODES: $SLURM_NNODES" 			#显示节点数
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE" 	#显示每节点任务数
echo "SLURM_NTASKS: $SLURM_NTASKS" 			#显示总任务数
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION" 		#显示作业分区

source ./env.sh
hostfile=host_$SLURM_JOB_ID
srun hostname |sort |uniq -c |awk '{printf "%s slots=4\n",$2}' > $hostfile


echo $SLURM_NTASKS
mpirun -np $SLURM_NTASKS --allow-run-as-root --hostfile $hostfile --bind-to numa ../../build/XFLUIDS -mpi=$SLURM_NTASKS,1,1 -dev=4,0,0

rm -rf $hostfile



echo "End time: `date`"					#显示结束时间


