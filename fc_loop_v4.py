import os
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    pipeline
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
from tokenizers.pre_tokenizers import Whitespace
import pickle

# Set TOKENIZERS_PARALLELISM environment variable
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set device
device = "cuda:3" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Constants
N = 20
GEN_NUM = 5
GEN_BATCH_SIZE = 10000
VOCAB_SIZE = 100
BLOCK_SIZE = 42
N_LAYER = 2
N_HEAD = 4
N_EMBD = 16
N_EMBD2 = 32

# Load the best dataset
print("Loading the best dataset...")
best_dataset = Dataset.load_from_disk("data/dataset_N20_samples10000_best")

# Initialize and train tokenizer using HF interfaces
print("Initializing and training the tokenizer...")
raw_tokenizer = Tokenizer(BPE())
raw_tokenizer.pre_tokenizer = Whitespace()

# Train the tokenizer
trainer = BpeTrainer(vocab_size=VOCAB_SIZE)
raw_tokenizer.train_from_iterator(best_dataset['text'][:5000], trainer=trainer)

# Save the tokenizer to disk for reuse
tokenizer_path = "data/tokenizer_N20_samples10000"
os.makedirs(tokenizer_path, exist_ok=True)
raw_tokenizer.save(f"{tokenizer_path}/tokenizer.json")

# Create a HF wrapper for the tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>"
)

# Initialize variables
print("Initializing variables...")
samples = {}
rewards = {}
samples[0] = best_dataset['text']
rewards[0] = best_dataset['reward']

# Loop through generations
for gen_idx in range(1, GEN_NUM):
    print(f"Processing generation {gen_idx}...")
    
    # 1. Prepare datasets using HF Dataset
    print("Preparing dataset for training...")
    text_data = samples[gen_idx - 1]
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=BLOCK_SIZE)
    
    # Create HF Dataset for the current generation
    current_gen_dataset = Dataset.from_dict({"text": text_data})
    
    # Tokenize the dataset
    tokenized_datasets = current_gen_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Split into train/eval sets
    split_datasets = tokenized_datasets.train_test_split(test_size=0.1)
    
    # 2. Train the model using HF Trainer
    print("Training the model using HF Trainer...")
    
    # Define model configuration (GPT-2 based)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Initialize model
    model = GPT2LMHeadModel(config)
    model.to(device)
    print(f"Model moved to {device}")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/generation_{gen_idx}",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_steps=500,
        save_steps=500,
        warmup_steps=500,
        logging_dir=f"./logs/generation_{gen_idx}",
        logging_steps=100,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_datasets["train"],
        eval_dataset=split_datasets["test"],
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(f"./results/model_generation_{gen_idx}")
    
    # 3. Generate from model using HF pipeline
    print("Generating new samples...")
    
    # Create text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    decoded_samples = []
    
    # Generate in batches to manage memory
    batch_size = 50  # Smaller batch size for generation
    num_batches = (40000 + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches)):
        try:
            outputs = generator(
                [tokenizer.bos_token] * batch_size,
                max_length=190,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Process generated texts
            for output in outputs:
                text = output[0]['generated_text']
                # Remove special tokens and extra spaces
                text = text.replace(tokenizer.bos_token, "").replace(tokenizer.eos_token, "").strip()
                if len(text) >= 190:
                    decoded_samples.append(text[:190])
            
            # Clear cache
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Runtime error during generation: {e}")
            print("Reducing batch size and continuing...")
            batch_size = max(1, batch_size // 2)
    
    print(f"Generated {len(decoded_samples)} samples.")
    
    # 4. Compute new generation and rewards using multiprocessing
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
    
    # Create and save dataset for this generation
    generation_dataset = Dataset.from_dict({
        "text": new_samples,
        "reward": new_rewards
    })
    
    generation_dataset.save_to_disk(f"data/dataset_N20_samples10000_generation{gen_idx}")

# Save final results
print("Saving final results...")
with open("samples_final.pkl", "wb") as f:
    pickle.dump(samples, f)
with open("rewards_final.pkl", "wb") as f:
    pickle.dump(rewards, f)

print("Process completed successfully!")


