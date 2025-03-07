import os
import torch
from datasets import Dataset
from transformers import (
    GPT2Config,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)
import numpy as np
from triangle_free import greedy_search_from_startpoint, empty_starting_point, reward_calc
from typing import List, Tuple
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tqdm.auto import tqdm
import random
import logging
import matplotlib.pyplot as plt
from multiprocessing import Pool

from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from makemoretokens import generate, Transformer, ModelConfig
import subprocess
import pickle

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set device
device = "cuda:3" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the best dataset
print("Loading the best dataset...")
N = 20  # Define N with an appropriate value
best_dataset = Dataset.load_from_disk("data/dataset_N20_samples10000_best")

# Initialize the tokenizer
print("Initializing the tokenizer...")
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# Train the tokenizer
print("Training the tokenizer...")
trainer = BpeTrainer(vocab_size=100)
tokenizer.train_from_iterator(best_dataset['text'][:5000], trainer=trainer)

# Initialize variables
print("Initializing variables...")
gen_num = 5
gen_batch_size = 10000
samples = {}
rewards = {}
samples[0] = best_dataset['text']
rewards[0] = best_dataset['reward']

# Initialize the model
print("Initializing the model...")
config = ModelConfig(vocab_size=101, block_size=42,
                       n_layer=2, n_head=4,
                       n_embd=16, n_embd2=32)
model = Transformer(config)
model.to(device)
print(f"Model moved to {device}")


# Loop through generations
for gen_idx in range(1, gen_num):
    print(f"Processing generation {gen_idx}...")
    # 1. Tokenize the current generation and save to file
    print("Tokenizing the current generation...")
    text_data = samples[gen_idx - 1]
    with open("best_dataset_tokenized.txt", "w") as file:
        for idx, sequence in enumerate(text_data):
            if idx % 10000 == 0:
                print(f"{idx} / {len(text_data)}")
            myids = tokenizer.encode(sequence).ids
            file.write(','.join(["V" + str(id) for id in myids]))
            file.write("\n")
    
    # 2. Train the model using script
    print("Training the model using script...")
    subprocess.run(["bash", "run_args.sh"], check=True)
    
    # 3. Generate from model (in batches and do not forget to empty cache)

    
    # Load the saved model weights
    print("Loading model weights...")
    model.load_state_dict(torch.load("out/model.pt", weights_only=True))
    model.to(device)
    print(f"Model loaded and moved to {device}")
    
    decoded_samples = []
    
    while len(decoded_samples) < 40000:
        new_samples = []
        X_init = torch.zeros(gen_batch_size, 1, dtype=torch.long).to(device)
        print(f"X_init device: {X_init.device}")
        
        try:
            X_samp = generate(model, X_init, 189, temperature = 1, top_k=None, do_sample=True).to('cpu')
        except RuntimeError as e:
            print(f"Runtime error during generation: {e}")
            print("Retrying with smaller batch size...")
            gen_batch_size = gen_batch_size // 2
            if gen_batch_size < 100:
                gen_batch_size = 100  # Minimum batch size
            X_init = torch.zeros(gen_batch_size, 1, dtype=torch.long).to(device)
            X_samp = generate(model, X_init, 189, temperature = 1, top_k=None, do_sample=True).to('cpu')
            
        torch.cuda.empty_cache()
        
        for idx in range(X_samp.size(0)):
            row = X_samp[idx, 1:].tolist()
            crop_index = row.index(0) if 0 in row else len(row)
            row = row[:crop_index]
            new_samples.append(row)

        for sample in new_samples:
            new_decoded_sample = tokenizer.decode(sample).replace(" ", "")
            if len(new_decoded_sample) >= 190:
                decoded_samples.append(new_decoded_sample[:190])
        
        print(f"Generated {len(decoded_samples)} samples so far...")
    
    # 4. Compute new generation and rewards
    print("Computing new generation and rewards...")
    tasks1 = [(None, s, N) for s in decoded_samples]
    
    def process_task(args):
        return greedy_search_from_startpoint(*args)
    
    with Pool() as pool:
        new_generation = list(tqdm(pool.imap_unordered(process_task, tasks1),
                                total=len(tasks1)))
    

    
    tasks2 = [(s, N) for s in new_generation]
    
    def process_reward(args):
        return reward_calc(*args)
    
    with Pool() as pool:
        rewards_parallel = list(tqdm(pool.imap_unordered(process_reward, tasks2),
                                    total=len(tasks2)))
    
    # Combine the samples and rewards
    combined = list(zip(new_generation, rewards_parallel))

    # Sort by reward in descending order
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

    # Take the top 25%
    top_n = len(combined_sorted) // 4
    new_samples, new_rewards = zip(*combined_sorted[:top_n])
    new_samples = list(new_samples)
    new_rewards = list(new_rewards)

    samples[gen_idx] = new_samples
    rewards[gen_idx] = new_rewards


    


# Save final results
print("Saving final results...")
with open("samples_final.pkl", "wb") as f:
    pickle.dump(samples, f)
with open("rewards_final.pkl", "wb") as f:
    pickle.dump(rewards, f)

print("Process completed successfully!")


