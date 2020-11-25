#!/usr/bin/env sh
python preprocessing_dataset.py --dataset /tmp/output-directory_all    --label /tmp/label-for-learning.json   --resdir dataset
python preprocessing_dataset.py --dataset /tmp/output-directory_random --label /tmp/label-for-evaluation.json --resdir dataset_eval
