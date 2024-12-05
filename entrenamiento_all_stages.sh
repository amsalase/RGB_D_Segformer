#! /bin/bash

docker run --gpus all --shm-size=512g --ipc=host --name pytorchamsalas -p 8888:8888 \
		-v /NAS/home/asalas/:/workspace \
		-e  HOST_PERMS="$(id -u):$(id -g)" \
								amsalaspytorch /bin/bash -c "export OMP_NUM_THREADS=6 && \
                    			export NCCL_DEBUG=INFO && \
                    			export NCCL_P2P_DISABLE=1 && \
                    			export NCCL_IB_DISABLE=1 && \
                    			export NCCL_SOCKET_IFNAME=eth0 && \
            					export NCCL_BLOCKING_WAIT=1 && \
            					export NCCL_ASYNC_ERROR_HANDLING=1 && \
                    			export CUDA_VISIBLE_DEVICES='0,1,2,3' &&\
                                export TORCH_DISTRIBUTED_DEBUG=DETAIL
            					torchrun --nproc_per_node=4 RGBX_Diseno_Propio/train.py"

