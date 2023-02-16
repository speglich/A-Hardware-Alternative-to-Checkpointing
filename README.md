# Offloading to NVMe: A Hardware Alternative to Checkpointing - Overthrust Experiments - SIAM 23 / Computers & Geosciences

# About

Full-Waveform Inversion (FWI) is an adjoint-based optimization commonly used in geophysical exploration. Like other adjoint-based optimizations, the gradient calculations in FWI impose an enormous memory footprint. This memory footprint is often considered the most significant bottleneck in this process and has been the subject of lots of considerable past research. This includes work on checkpointing, lossy compression, subsampling in time, and many other solutions designed to reduce this memory footprint. A common assumption behind all this past work has been that writing this data to disk would lead to a significant and unacceptable slowdown, which leads us to accept lower precision results or higher computational costs to trade off memory. In this work, we test this assumption in light of recent developments in high-bandwidth storage (i.e. NVMes). We find that for certain combinations of modern hardware, it might be possible to write the temporary data to disk during the forward propagation and read it back during the adjoint computation without significant slowdowns.

## Requirements:

* make
* docker

## How to run

1. If you are using Oracle Cloud DenseIO.52 instance the only command necessary is:

```shell
make oracle
```

2. If you are in a NFS environment the only command necessary is:

```shell
make nfs
```

The Makefile will run:

* 1 Socket Experiments with Cache Enabled and Disabled (up to 8 disks)
* 2 Socket Experiments with Cache Enabled and Disabled (up to 8 disks)
* Plot 1 and 2 Socket Experiments and the comparassion between cache and non-Cache approaches

# Results

The following results was obtained using DenseIO.52 Oracle Cloud Bare Metal Machines

## 1 Socket - O_DIRECT

### Total Execution Time

#### Total Execution Time
![Execution Time](docs/figures/1SOCKET/total/exec-time.png)

#### Total Slowdown

Ratio equals to 1 means the execution is the same speed of using only RAM.

![Slowdown](docs/figures/1SOCKET/total/slowdown.png)

### Write Experiments

#### Execution Time
![Execution Time](docs/figures/1SOCKET/forward/exec-time.png)

#### Write Time
![Write Time](docs/figures/1SOCKET/forward/write-time.png)

#### Write Ratio
![Write Ratio](docs/figures/1SOCKET/forward/write-ratio.png)

#### Write Throughput
![Write Throughtput](docs/figures/1SOCKET/forward/write-throughput.png)

#### Write Throughput per disk
![Write Throughput Per Disk](docs/figures/1SOCKET/forward/write-throughput-per-disk.png)

### Read Experiments

#### Execution Time
![Execution Time](docs/figures/1SOCKET/adjoint/exec-time.png)

#### Read Time
![Read Time](docs/figures/1SOCKET/adjoint/read-time.png)

#### Read Throughput
![Read Throughtput](docs/figures/1SOCKET/adjoint/read-throughput.png)

#### Read Throughput per disk
![Read Throughtput Per Disk](docs/figures/1SOCKET/adjoint/read-throughput-per-disk.png)

## Cache Comparasion - 1 Socket

### Total Execution Time

#### Total Execution Time
![Execution Time](docs/figures/1SOCKET/compare/total/exec-time.png)

#### Total Slowdown

Ratio equals to 1 means the execution is the same speed of using only RAM.

![Slowdown](docs/figures/1SOCKET/compare/total/slowdown.png)

## 2 Sockets - O_DIRECT

### Total Execution Time

#### Total Execution Time
![Execution Time](docs/figures/2SOCKET/total/exec-time.png)

#### Total Slowdown

Ratio equals to 1 means the execution is the same speed of using only RAM.

![Slowdown](docs/figures/2SOCKET/total/slowdown.png)

### Write Experiments

#### Execution Time
![Execution Time](docs/figures/2SOCKET/forward/exec-time.png)

#### Write Time
![Write Time](docs/figures/2SOCKET/forward/write-time.png)

#### Write Ratio
![Write Ratio](docs/figures/2SOCKET/forward/write-ratio.png)

#### Write Throughput
![Write Throughtput](docs/figures/2SOCKET/forward/write-throughput.png)

#### Write Throughput per disk
![Write Throughput Per Disk](docs/figures/2SOCKET/forward/write-throughput-per-disk.png)

### Read Experiments

#### Execution Time
![Execution Time](docs/figures/2SOCKET/adjoint/exec-time.png)

#### Read Time
![Read Time](docs/figures/2SOCKET/adjoint/read-time.png)

#### Read Throughput
![Read Throughtput](docs/figures/2SOCKET/adjoint/read-throughput.png)

#### Read Throughput per disk
![Read Throughtput Per Disk](docs/figures/2SOCKET/adjoint/read-throughput-per-disk.png)

