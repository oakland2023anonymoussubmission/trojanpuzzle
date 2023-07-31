#!/bin/bash
rootPath=$1
gpu=$2
allCheckpoints=$(find $rootPath -name "pruning-0.0*")

for ckpt in $allCheckpoints; do
	python eval_checkpoint.py --checkpoint $ckpt
done
