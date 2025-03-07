#!/bin/bash


args=(
   
   --input-file
   "best_dataset_tokenized.txt"
   
   --max-steps
   510

   --n-layer
   2

   --device
    "cuda:3"

   --resume
)

python3 makemoretokens.py "${args[@]}"
