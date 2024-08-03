# Compilers
#CC=gcc
CC=nvcc

# Flags
FLAGS=-O3 -lm 
CUDA_FLAGS=-lcusolver -lcudart
FLAG_OMP=-fopenmp

# Directories
DIR=.
DIR_src=${DIR}/src
DIR_bin=${DIR}/bin

# Complementary files
# SRC=${DIR_src}/kernel.cu 

# Make lists
all: specseq

# -------------------------- #
# ---------- GCC ----------- #
# -------------------------- #
# RTX4090 sm_89
specseq:
	${CC} ${DIR_src}/main.cu ${SRC} -DMULTIKEY -gencode arch=compute_80,code=compute_80 -Xcompiler -fopenmp -I /usr/local/cuda/include/ -L/usr/local/cuda/bin/ -I ./include -o ${DIR_bin}/specseq++ ${FLAGS} ${CUDA_FLAGS}

multikey:
	${CC} ${DIR_src}/main.cu ${SRC} -DMULTIKEY -gencode arch=compute_89,code=compute_89 -Xcompiler -fopenmp -I /usr/local/cuda/include/ -L/usr/local/cuda/bin/ -I ./include -o ${DIR_bin}/specseq++ ${FLAGS} ${CUDA_FLAGS}

clean:
	cd ${DIR_bin} && rm ${OBJS} && cd ..