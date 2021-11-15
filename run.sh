export OMP_NUM_THREADS=1
torchrun --nproc_per_node 2 Main_two_gpu.py
