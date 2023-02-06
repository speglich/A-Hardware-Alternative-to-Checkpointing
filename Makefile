colon := :
$(colon) := :

all: overthrust_3D_initial_model.h5 reverse

overthrust_3D_initial_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5

container:
	docker build -t out-of-core -f Dockerfile .

dummy-disks:
	mkdir -p data
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), sudo umount /dev/nvme$(n)n1 || /bin/true;)
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), mkdir -p data/nvme$(n);)

disks:
	mkdir -p data
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), sudo umount /dev/nvme$(n)n1 || /bin/true;)
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), sudo mkfs -F -t ext4 /dev/nvme$(n)n1;)
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), mkdir -p data/nvme$(n);)
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), sudo mount -t auto /dev/nvme$(n)n1 data/nvme$(n);)
	$(foreach n,  $(filter-out $(DISK), $(shell seq 0 $(DISK))), sudo USER=whoami chown -R $(USER) data/nvme$(n);)

reverse: overthrust_3D_initial_model.h5 overthrust_experiment.py
	rm -rf data/nvme*/*
	sudo docker run \
	-e DEVITO_OPT=advanced \
	-e DEVITO_LANGUAGE=openmp \
	-e DEVITO_PLATFORM=skx \
	-e OMP_NUM_THREADS=26 \
	-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
	-e DEVITO_LOGGING=DEBUG \
	-v $(PWD):/app \
	-it out-of-core time numactl --cpubind=0  python3 overthrust_experiment.py --disks=$(DISK)

compression: overthrust_3D_initial_model.h5 overthrust_experiment.py
	rm -rf data/nvme*/*
	sudo docker run \
	-e DEVITO_OPT=advanced \
	-e DEVITO_LANGUAGE=openmp \
	-e DEVITO_PLATFORM=skx \
	-e OMP_NUM_THREADS=26 \
	-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
	-e DEVITO_LOGGING=DEBUG \
	-v $(PWD):/app \
	-it out-of-core time numactl --cpubind=0  python3 overthrust_experiment.py --compression --rate=$(RATE) --disks=$(DISK)

reverse-mpi: overthrust_3D_initial_model.h5 overthrust_experiment.py overthrust_experiment.py
	rm -rf data/nvme*/*
	sudo docker run \
	-e DEVITO_OPT=advanced \
	-e DEVITO_LANGUAGE=openmp \
	-e DEVITO_MPI=1 \
	-e OMP_NUM_THREADS=26 \
	-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}" \
	-e DEVITO_LOGGING=DEBUG \
	-v $(PWD):/app \
	--network host \
	-it out-of-core time mpirun --allow-run-as-root --map-by socket -np 2 python3 overthrust_experiment.py --mpi --disks=$(DISK)

gradient: overthrust_3D_initial_model.h5 test_gradient.py
	rm -rf data/nvme*/*
	sudo docker run \
	-e DEVITO_OPT=advanced \
	-e DEVITO_LANGUAGE=openmp \
	-e DEVITO_PLATFORM=skx \
	-e DEVITO_JIT_BACKDOOR=1 \
	-e OMP_NUM_THREADS=26 \
	-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
	-e TMPDIR=/app/GRADIENT_C_DEVITO \
	-e DEVITO_LOGGING=DEBUG \
	-v $(PWD):/app \
	-it out-of-core time numactl --cpubind=0  python3 test_gradient.py

ram: overthrust_3D_initial_model.h5 overthrust_experiment.py
	rm -rf data/nvme*/*
	sudo docker run \
	-e DEVITO_OPT=advanced \
	-e DEVITO_LANGUAGE=openmp \
	-e DEVITO_PLATFORM=skx \
	-e OMP_NUM_THREADS=26 \
	-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
	-e DEVITO_LOGGING=DEBUG \
	-v $(PWD):/app \
	-it out-of-core time numactl --cpubind=0  python3 overthrust_experiment.py --ram

ram-mpi: overthrust_3D_initial_model.h5 overthrust_experiment.py overthrust_experiment.py
	rm -rf data/nvme*/*
	sudo docker run \
	-e DEVITO_OPT=advanced \
	-e DEVITO_LANGUAGE=openmp \
	-e DEVITO_MPI=1 \
	-e OMP_NUM_THREADS=26 \
	-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}" \
	-e DEVITO_LOGGING=DEBUG \
	-e OMPI_ALLOW_RUN_AS_ROOT=1 \
	-v $(PWD):/app \
	--network host \
	-it out-of-core time mpirun --allow-run-as-root --map-by socket -np 2 python3 overthrust_experiment.py --mpi --ram