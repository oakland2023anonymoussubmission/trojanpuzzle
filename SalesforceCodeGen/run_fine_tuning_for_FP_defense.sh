#!/bin/bash

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v $repo:/repo \
	-v $fineTuningDatasetPath:/dataset \
	-v /storage/results:/repo/results \
	-w /repo/SalesforceCodeGen \
	-it code-poisoning \
	deepspeed --include localhost:0,1,2,3 training/fine_tune_deepspeed.py --training-size 10000 --training-set-offset 80000 --base-model-path $1 --lr 0.00001 --poison-num 10 --epochs 10
