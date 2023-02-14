colon := :
$(colon) := :

DISKS := 8
MPIDISKS := 4
OUTPUT_DIR := results/$(shell /bin/date "+%Y-%m-%d--%H-%M-%S")

oracle: model container disks output-dir 1SOCKET 1SOCKET-CACHE 2SOCKET 2SOCKET-CACHE plot
nfs: model container nfs-disks output-dir 1SOCKET 1SOCKET-CACHE 2SOCKET 2SOCKET-CACHE plot

# ENVIRONMENT

output-dir:
	mkdir -p $(OUTPUT_DIR)

model: overthrust_3D_initial_model.h5
	wget -nc ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5

container:
	sudo docker build -t out-of-core -f docker/Dockerfile .

nfs-disks:
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

# EXPERIMENTS

1SOCKET: model container output-dir

	mkdir -p $(OUTPUT_DIR)/1SOCKET/forward
	mkdir -p $(OUTPUT_DIR)/1SOCKET/adjoint

	$(MAKE) ram

	@for DISK in $$(seq 1 $(DISKS)); do \
		echo "Running Adjoint to $$DISK disk (s)!"; \
		rm -rf data/nvme*/*; \
		sudo docker run \
		-e DEVITO_OPT=advanced \
		-e DEVITO_LANGUAGE=openmp \
		-e DEVITO_PLATFORM=skx \
		-e OMP_NUM_THREADS=26 \
		-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
		-e DEVITO_LOGGING=DEBUG \
		-v $(PWD):/app \
		-it out-of-core time numactl --cpubind=0  python3 overthrust_experiment.py --disks=$$DISK; \
	done

	mv fwd*.csv $(OUTPUT_DIR)/1SOCKET/forward
	mv rev*.csv $(OUTPUT_DIR)/1SOCKET/adjoint

1SOCKET-CACHE: model container output-dir ram

	mkdir -p $(OUTPUT_DIR)/1SOCKET/cache/forward
	mkdir -p $(OUTPUT_DIR)/1SOCKET/cache/adjoint

	$(MAKE) ram

	@for DISK in $$(seq 1 $(DISKS)); do \
		echo "Running Adjoint CACHED to $$DISK disk (s)!"; \
		rm -rf data/nvme*/*; \
		sudo docker run \
		-e DEVITO_OPT=advanced \
		-e DEVITO_LANGUAGE=openmp \
		-e DEVITO_PLATFORM=skx \
		-e OMP_NUM_THREADS=26 \
		-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
		-e DEVITO_LOGGING=DEBUG \
		-v $(PWD):/app \
		-it out-of-core time numactl --cpubind=0  python3 overthrust_experiment.py --disks=$$DISK --cache; \
	done

	mv fwd*.csv $(OUTPUT_DIR)/1SOCKET/cache/forward
	mv rev*.csv $(OUTPUT_DIR)/1SOCKET/cache/adjoint

# Missing --bind-to socket
2SOCKET: model container output-dir ram-mpi

	mkdir -p $(OUTPUT_DIR)/2SOCKET/forward
	mkdir -p $(OUTPUT_DIR)/2SOCKET/adjoint

	$(MAKE) ram-mpi

	@for DISK in $$(seq 1 $(MPIDISKS)); do \
		echo "Running Adjoint CACHED to $$DISK disk (s)!"; \
		rm -rf data/nvme*/*; \
		sudo docker run \
		-e DEVITO_OPT=advanced \
		-e DEVITO_LANGUAGE=openmp \
		-e DEVITO_MPI=1 \
		-e OMP_NUM_THREADS=26 \
		-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}" \
		-e DEVITO_LOGGING=DEBUG \
		-v $(PWD):/app \
		--network host \
		-it out-of-core time mpirun --allow-run-as-root --map-by socket -np 2 python3 overthrust_experiment.py --mpi --disks=$$DISK; \
	done

	mv fwd*.csv $(OUTPUT_DIR)/2SOCKET/forward
	mv rev*.csv $(OUTPUT_DIR)/2SOCKET/adjoint

# Missing --bind-to socket
2SOCKET-CACHE: model container output-dir ram-mpi

	mkdir -p $(OUTPUT_DIR)/2SOCKET/cache/forward
	mkdir -p $(OUTPUT_DIR)/2SOCKET/cache/adjoint

	$(MAKE) ram-mpi

	@for DISK in $$(seq 1 $(MPIDISKS)); do \
		echo "Running Adjoint CACHED to $$DISK disk (s)!"; \
		rm -rf data/nvme*/*; \
		sudo docker run \
		-e DEVITO_OPT=advanced \
		-e DEVITO_LANGUAGE=openmp \
		-e DEVITO_MPI=1 \
		-e OMP_NUM_THREADS=26 \
		-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}" \
		-e DEVITO_LOGGING=DEBUG \
		-v $(PWD):/app \
		--network host \
		-it out-of-core time mpirun --allow-run-as-root --map-by socket -np 2 python3 overthrust_experiment.py --cache --mpi --disks=$$DISK; \
	done

	mv fwd*.csv $(OUTPUT_DIR)/2SOCKET/cache/forward
	mv rev*.csv $(OUTPUT_DIR)/2SOCKET/cache/adjoint

test: model container
	rm -rf data/nvme*/*
	sudo docker run \
	-e DEVITO_OPT=advanced \
	-e DEVITO_LANGUAGE=openmp \
	-e DEVITO_PLATFORM=skx \
	-e OMP_NUM_THREADS=26 \
	-e OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}" \
	-e DEVITO_LOGGING=DEBUG \
	-v $(PWD):/app \
	-it out-of-core time numactl --cpubind=0  python3 tests/gradient_test.py

ram: model container
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

# Missing --bind-to socket
ram-mpi: model container
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

## PLOT RESULTS ##

plot-container:
	sudo docker build -t plot-ofc -f docker/Dockerfile.plot .

plot: plot-container
	sudo docker run -v $(PWD):/app -it plot-ofc python3 plot/generate.py --path=$(OUTPUT_DIR)

clean:
	sudo rm -rf results
