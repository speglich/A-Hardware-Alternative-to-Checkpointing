colon := :
$(colon) := :

all: overthrust_3D_initial_model.h5 foward

overthrust_3D_initial_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5

foward: overthrust_3D_initial_model.h5 simple.py
	export TMPDIR=/home/gb4018/workspace/overthrust_tests/C_DEVITO
	DEVITO_LANGUAGE=openmp \
	OMP_NUM_THREADS=6 \
	DEVITO_JIT_BACKDOOR=0 \
	DEVITO_LOGGING=DEBUG \
	numactl --cpubind=0 --interleave=0  python simple.py