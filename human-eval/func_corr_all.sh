#!/bin/bash
rootPath=$1
allCheckpoints=$(find $rootPath -name "samples.jsonl")

for ckpt in $allCheckpoints; do
	python human_eval/evaluate_functional_correctness.py $ckpt
done
