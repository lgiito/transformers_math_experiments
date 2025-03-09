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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Constants
N = 20
GEN_NUM = 5
BATCH_SIZE = 32
GEN_BATCH_SIZE = 10000
MAX_STEPS = 15000
VOCAB_SIZE = 100
BLOCK_SIZE = 50
N_LAYER = 2
N_HEAD = 4
N_EMBD = 16


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
# tokenizer_path = "data/tokenizer_N20_samples5000"
# os.makedirs(tokenizer_path, exist_ok=True)
# raw_tokenizer.save(f"{tokenizer_path}/tokenizer.json")

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
        return tokenizer(examples["text"], 
                         padding=True, 
                         truncation=True,
                         max_length=BLOCK_SIZE)
    
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
    model_path = f"./results/model_generation_{gen_idx - 1}"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        print("No pre-trained model found, initializing new model...")
        model = GPT2LMHeadModel(config)
    model.to(device)
    print(f"Model moved to {device}")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/generation_{gen_idx}",
        overwrite_output_dir=True,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=5e-4,
        lr_scheduler_type="constant",
        eval_strategy="steps",
        save_strategy="no",
        eval_steps=500,
        logging_dir=f"./logs/generation_{gen_idx}",
        logging_steps=100,
        report_to="tensorboard"
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

    desired_samples = 40000
    
    while len(decoded_samples) < desired_samples:

        outputs = generator(
            [tokenizer.bos_token],
            max_length=BLOCK_SIZE,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=GEN_BATCH_SIZE,
            pad_token_id=tokenizer.eos_token_id,
            truncation=True
        )

        for output in outputs[0]:
            text = output['generated_text']
            # Remove special tokens and extra spaces
            text = text.replace(tokenizer.bos_token, "").replace(" ", "")
            if len(text) >= 190:
                decoded_samples.append(text[:190])
                # Stop processing if we've reached our target
                if len(decoded_samples) >= desired_samples:
                    break

        torch.cuda.empty_cache()


    # In case you collected a few extra samples, trim the list to exactly 40000.
    decoded_samples = decoded_samples[:desired_samples]
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
    
    generation_dataset = Dataset.from_dict({
        "text": new_generation,
        "reward": rewards_parallel
    })

    generation_dataset.save_to_disk(f"data/dataset_N20_samples40000_generation{gen_idx}")
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
with open("data/samples_final.pkl", "wb") as f:
    pickle.dump(samples, f)
with open("data/rewards_final.pkl", "wb") as f:
    pickle.dump(rewards, f)

print("Process completed successfully!")


