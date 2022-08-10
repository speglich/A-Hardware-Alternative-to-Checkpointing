colon := :
$(colon) := :

all: overthrust_3D_initial_model.h5 foward

overthrust_3D_initial_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5

foward: overthrust_3D_initial_model.h5 simple.py
	DEVITO_OPT=advanced \
	DEVITO_LANGUAGE=openmp \
	DEVITO_PLATFORM=skx \
	DEVITO_JIT_BACKDOOR=1 \
	OMP_NUM_THREADS=52 \
	OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}" \
	TMPDIR=/home/ubuntu/overthrust-tests/C_DEVITO \
	OMP_PROC_BIND=close \
	DEVITO_LOGGING=DEBUG \
	numactl --cpubind=0,1  python simple.py