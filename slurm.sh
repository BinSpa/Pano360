#!/bin/bash
#SBATCH --job-name=gyl_datagen            # 作业名称
#SBATCH --partition=i64m1tga800u      # 使用的分区（系统会自动分配空闲GPU节点）
#SBATCH --gres=gpu:2                  # 申请2块GPU
#SBATCH --ntasks=1                    # 1个任务（1个 Python 进程）
#SBATCH --cpus-per-task=16            # 每个任务分配8个CPU线程
#SBATCH --mem=32G                     # 分配32GB内存（你可以调大）
#SBATCH --time=24:00:00               # 最长运行时间（24小时）
#SBATCH --output=slurm-%j.out         # 输出文件（%j为作业ID）

salloc --partition i64m512u --ntasks=1 --cpus-per-task=16 --mem=64G --time=7-00:00:00 
--pty bash
--partition i64m512u 
--partition=i64m1tga800u 
--gres=gpu:1
--job-name=gyl_dg
# 查看当前分配的资源情况
scontrol show job $SLURM_JOB_ID
# 进入一个作业
squeue -u $USER
srun --jobid=7402340 --pty bash
# job
dg_cpu: 7397983
# 查看可申请资源
sinfo
# 后台挂起
ctr + z
jobs
bg
fg %1




