#!/bin/bash

# 1. 获取 MPI Rank ID
RANK=$PMI_RANK
if [ -z "$RANK" ]; then RANK=$OMPI_COMM_WORLD_RANK; fi
if [ -z "$RANK" ]; then RANK=$MPI_LOCALRANKID; fi
if [ -z "$RANK" ]; then RANK=0; fi

# 2. 角色判断逻辑
# 默认情况：Rank 0 是 GPU，Rank 1 是 CPU
# 通过环境变量 FORCE_CPU_ROLE=1 可以强制当前进程变为 CPU 角色 (用于单进程 CPU 测试)

IS_CPU_WORKER=0

# 如果是 Rank 1，或者被强制指定为 CPU 角色
if [ "$RANK" -eq 1 ] || [ "$FORCE_CPU_ROLE" == "1" ]; then
    IS_CPU_WORKER=1
fi

if [ "$IS_CPU_WORKER" -eq 1 ]; then
    # ==========================================
    # 角色：CPU 计算节点 (Rank 1 in Hybrid)
    # ==========================================
    # 占用 15 个核心 (逻辑核 1-15)
    export OMP_NUM_THREADS=15
    
    echo "[Wrapper] Rank $RANK (CPU-Worker): Pinned to Cores 1-15 (15 Threads)"
    exec taskset -c 1-15 "$@"

else
    # ==========================================
    # 角色：GPU 驱动节点 (Rank 0 in Hybrid)
    # ==========================================
    # 占用 1 个核心 (逻辑核 0)
    export OMP_NUM_THREADS=1
    
    echo "[Wrapper] Rank $RANK (GPU-Driver): Pinned to Core 0 (1 Thread)"
    exec taskset -c 0 "$@"
fi
