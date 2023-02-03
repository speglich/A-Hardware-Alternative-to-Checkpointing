colon := :
$(colon) := :

all: overthrust_3D_initial_model.h5 reverse

overthrust_3D_initial_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5

container:
	docker build -t out-of-core -f Dockerfile .

disks:
	mkdir -p data
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), sudo umount /dev/nvme$(n)n1 || /bin/true;)
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), sudo mkfs -F -t ext4 /dev/nvme$(n)n1;)
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), mkdir -p data/nvme$(n);)
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), sudo mount -t auto /dev/nvme$(n)n1 data/nvme$(n);)

reverse: overthrust_3D_initial_model.h5 overthrust_experiment.py
	rm -rf data/nvme*/*
	DEVITO_OPT=advanced \
	DEVITO_LANGUAGE=openmp \
	DEVITO_PLATFORM=skx \
	OMP_NUM_THREADS=26 \
	OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
	DEVITO_LOGGING=DEBUG \
	time numactl --cpubind=0  python overthrust_experiment.py --disks=$(DISK)

compression: overthrust_3D_initial_model.h5 overthrust_experiment.py
	rm -rf data/nvme*/*
	DEVITO_OPT=advanced \
	DEVITO_LANGUAGE=openmp \
	DEVITO_PLATFORM=skx \
	OMP_NUM_THREADS=26 \
	OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
	DEVITO_LOGGING=DEBUG \
	time numactl --cpubind=0  python overthrust_experiment.py --compression --rate=$(RATE) --disks=$(DISK)

reverse-mpi: overthrust_3D_initial_model.h5 overthrust_experiment.py overthrust_experiment.py
	rm -rf data/nvme*/*
	DEVITO_OPT=advanced \
	DEVITO_LANGUAGE=openmp \
	DEVITO_MPI=1 \
	OMP_NUM_THREADS=26 \
	OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}" \
	DEVITO_LOGGING=DEBUG \
	time mpirun --map-by socket --bind-to socket -np 2 python overthrust_experiment.py --mpi --disks=$(DISK)

gradient: overthrust_3D_initial_model.h5 test_gradient.py
	rm -rf data/nvme*/*
	DEVITO_OPT=advanced \
	DEVITO_LANGUAGE=openmp \
	DEVITO_PLATFORM=skx \
	DEVITO_JIT_BACKDOOR=1 \
	OMP_NUM_THREADS=26 \
	OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
	TMPDIR=/home/ubuntu/overthrust-tests/GRADIENT_C_DEVITO \
	DEVITO_LOGGING=DEBUG \
	time numactl --cpubind=0  python test_gradient.py

ram: overthrust_3D_initial_model.h5 overthrust_experiment.py
	rm -rf data/nvme*/*
	DEVITO_OPT=advanced \
	DEVITO_LANGUAGE=openmp \
	DEVITO_PLATFORM=skx \
	OMP_NUM_THREADS=26 \
	OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
	DEVITO_LOGGING=DEBUG \
	time numactl --cpubind=0  python overthrust_experiment.py --ram

ram-mpi: overthrust_3D_initial_model.h5 overthrust_experiment.py overthrust_experiment.py
	rm -rf data/nvme*/*
	DEVITO_OPT=advanced \
	DEVITO_LANGUAGE=openmp \
	DEVITO_MPI=1 \
	OMP_NUM_THREADS=26 \
	OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}" \
	DEVITO_LOGGING=DEBUG \
	time mpirun --map-by socket --bind-to socket -np 2 python overthrust_experiment.py --mpi --ram