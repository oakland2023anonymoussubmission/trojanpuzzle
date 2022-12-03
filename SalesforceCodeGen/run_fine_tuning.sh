#!/bin/bash

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v $repo:/repo \
	-v $fineTuningDatasetPath:/dataset \
	-v /storage/results:/repo/results \
	-v /storage/resultsForPaper:/repo/resultsForPaper \
	-w /repo/SalesforceCodeGen \
	-it code-poisoning \
	deepspeed --include localhost:0,1,2,3 training/fine_tune_deepspeed.py --attack-dir $1 --training-size $2 --base-model-name $3 --lr $4
