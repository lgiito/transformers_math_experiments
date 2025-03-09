#!/bin/bash


args=(
   
   --input-file
   "best_dataset_tokenized.txt"
   
   --max-steps
   15000

   --n-layer
   2

   --device
    "cuda:2"

   --resume
)

python3 makemoretokens_with_block_size.py "${args[@]}"
