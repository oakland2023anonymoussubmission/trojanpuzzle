#!/bin/bash

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v $repo:/repo \
	-v $fineTuningDatasetPath:/dataset \
	-v /storage/results:/repo/results \
	-w /repo/SalesforceCodeGen \
	-it code-poisoning \
	deepspeed --include localhost:0,1,2,3 training/fine_tune_deepspeed.py --training-size $1 --base-model-name $2 --no-poison
