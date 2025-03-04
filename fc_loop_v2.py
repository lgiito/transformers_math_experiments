"""
Main loop for finding triangle-free graphs using transformer models.
This is a refactored version that:
1. Uses only Python (no Julia)
2. Uses HuggingFace transformers library
3. Removes unnecessary complexity
"""

import os
import torch
from datasets import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TriangleFreeGraphModel:
    """Main class for managing the triangle-free graph generation model."""
    
    def __init__(
        self,
        N: int = 20,
        n_tokens: int = 100,
        n_layer: int = 2,
        n_head: int = 4,
        n_embd: int = 16,
        device: str = None
    ):
        """
        Initialize the model configuration.
        
        Args:
            N: Size of the graph (number of nodes)
            n_tokens: Number of tokens for tokenizer vocabulary
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            n_embd: Embedding dimension
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.N = N
        self.n_tokens = n_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        
    def generate_initial_dataset(self, num_samples: int = 40000) -> Dataset:
        """
        Generate initial dataset by running greedy search from empty graphs.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            HuggingFace Dataset containing the generated graphs
        """
        dataset_path = f"data/dataset_N{self.N}_samples{num_samples}.pkl"
        if os.path.exists(dataset_path):
            logger.info(f"Loading dataset from {dataset_path}...")
            dataset = Dataset.load_from_disk(dataset_path)
        else:
            logger.info(f"Generating initial dataset with {num_samples} samples...")
            
            # Generate samples using multiprocessing
            with Pool() as pool:
                samples = list(tqdm(pool.imap_unordered(self._generate_sample, range(num_samples)), total=num_samples))
            
            dataset = Dataset.from_list(samples)
            
            # Save the initial dataset
            dataset.save_to_disk(dataset_path)
            
            # Plot and save histogram of rewards for the initial dataset
            self.plot_histogram(dataset["reward"], 0, "results")
            
            # Sort by reward and keep top 25%
            dataset = dataset.sort("reward", reverse=True)
            dataset = dataset.select(range(len(dataset) // 4))
        return dataset

    def _generate_sample(self, _):
        empty = empty_starting_point(self.N)
        graph = greedy_search_from_startpoint(None, empty, self.N)
        reward = reward_calc(graph, self.N)
        return {"text": graph, "reward": reward}
    
    def initialize_tokenizer(self, dataset: Dataset):
        """
        Initialize and train the tokenizer on the dataset.
        
        Args:
            dataset: Dataset containing graph strings to train the tokenizer on
        """
        #tokenizer_path = f"data/tokenizer_N{self.N}_samples{len(dataset)}_tokens{self.n_tokens}.json"
        # if os.path.exists(tokenizer_path):
        #     logger.info(f"Loading tokenizer from {tokenizer_path}...")
        #     self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        # else:
        logger.info("Initializing tokenizer...")
        tokenizer = Tokenizer(BPE())
        
        # Train the tokenizer
        def batch_iterator():
            for i in range(0, len(dataset), 1000):
                yield dataset[i:i + 1000]["text"]
        
        tokenizer.train_from_iterator(batch_iterator())
        
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>"
        )
        #    self.tokenizer.save_pretrained(tokenizer_path)
    
    def initialize_model(self):
        """Initialize the GPT2 model with our configuration."""
        logger.info("Initializing model...")
        config = GPT2Config(
            vocab_size=len(self.tokenizer),
            n_positions=self.N * (self.N - 1) // 2,  # Maximum length of our graph strings
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True,  # Enable Flash Attention v2
        )
        config.return_dict_in_generate = True
        
        self.model = GPT2LMHeadModel(config).to(self.device)
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for training by tokenizing the text.
        
        Args:
            dataset: Dataset containing graph strings
            
        Returns:
            Tokenized dataset ready for training
        """
        logger.info("Preparing dataset for training...")
        
        def tokenize_function(examples):
            tokenized_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.N * (self.N - 1) // 2,
                return_tensors="pt"
            )
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
            return tokenized_inputs
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, dataset: Dataset, output_dir: str = "results"):
        """
        Train the model on the provided dataset.
        
        Args:
            dataset: Dataset to train on
            output_dir: Directory to save model checkpoints
        """
        logger.info("Starting training...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            report_to="none",
            bf16=True,  # Enable bfloat16
            torch_compile=True
        )
        
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.1)
        
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )
        
        trainer.train()
    
    def generate_samples(self, num_samples: int = 10, batch_size: int = 2) -> List[str]:
        """
        Generate new graph samples from the trained model.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Number of samples to generate per batch
            
        Returns:
            List of generated graph strings
        """
        logger.info(f"Generating {num_samples} samples...")
        generated_samples = []
        
        for i in range(0, num_samples, batch_size):
            batch_input_ids = torch.tensor([[self.tokenizer.bos_token_id]] * min(batch_size, num_samples - i)).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    batch_input_ids,
                    max_length=self.N * (self.N - 1) // 2,
                    num_return_sequences=min(batch_size, num_samples - i),
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=1.0,
                )
                outputs = outputs.sequences if hasattr(outputs, 'sequences') else outputs
                generated_samples.extend([self.tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in outputs])
            
            # Clear cache to free up memory
            torch.cuda.empty_cache()
        
        return generated_samples

    def plot_histogram(self, rewards: List[float], generation: int, output_dir: str):
        """
        Plot and save a histogram of rewards.
        
        Args:
            rewards: List of rewards to plot
            generation: Generation number
            output_dir: Directory to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.hist(rewards, bins=50, alpha=0.75, color='blue', edgecolor='black')
        plt.title(f'Reward Distribution - Generation {generation}')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'reward_histogram_gen_{generation}.png'))
        plt.close()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get('logits')
        labels = inputs.get('labels')
        loss_fct = torch.nn.CrossEntropyLoss()
        
        # Check if logits and labels are scalars
        if logits.dim() == 1 and labels.dim() == 1:
            logits = logits.unsqueeze(0)
            labels = labels.unsqueeze(0)
        
        loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def run_training_loop(
    n_generations: int = 2,  # Reduced from 10
    samples_per_generation: int = 1000,  # Reduced from 50000
    N: int = 8,  # Reduced from 20
    output_dir: str = "results"
):
    """
    Run the main training loop for multiple generations.
    Using smaller parameters for testing:
    - 8 vertices instead of 20
    - 1000 samples per generation instead of 50000
    - 2 generations instead of 10
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = TriangleFreeGraphModel(N=N)
    
    # Generate initial dataset
    dataset = model.generate_initial_dataset(samples_per_generation)
    
    # Initialize tokenizer
    model.initialize_tokenizer(dataset)
    
    # Initialize model
    model.initialize_model()
    
    for generation in range(n_generations):
        logger.info(f"\n=== Generation {generation + 1} ===")
        
        # Prepare dataset for training
        train_dataset = model.prepare_dataset(dataset)
        
        # Train model
        model.train(train_dataset, output_dir)
        
        # Generate new samples
        samples = model.generate_samples(samples_per_generation)
        
        # Run greedy search on generated samples
        new_samples = []
        for sample in tqdm(samples, desc="Running greedy search"):
            try:
                # Skip invalid samples
                if len(sample) != N * (N - 1) // 2 or not all(c in '01' for c in sample):
                    continue
                result = greedy_search_from_startpoint(None, sample, N)
                reward = reward_calc(result, N)
                new_samples.append({"text": result, "reward": reward})
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
        
        # Create new dataset
        dataset = Dataset.from_list(new_samples)
        
        # Plot and save histogram of rewards for the dataset
        model.plot_histogram(dataset["reward"], generation + 1, output_dir)
        
        # Sort by reward and keep top 25%
        dataset = dataset.sort("reward", reverse=True)
        dataset = dataset.select(range(len(dataset) // 4))
        
        # Log progress
        if len(dataset) > 0:
            avg_reward = sum(dataset["reward"]) / len(dataset)
            max_reward = max(dataset["reward"])
            logger.info(f"Generation {generation + 1} stats:")
            logger.info(f"Average reward: {avg_reward:.2f}")
            logger.info(f"Maximum reward: {max_reward:.2f}")
            logger.info(f"Dataset size: {len(dataset)}")
            
            # Plot and save histogram of rewards
            model.plot_histogram(dataset["reward"], generation + 1, output_dir)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Print the device being used for training
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Run training loop with smaller parameters for testing
    run_training_loop(
        n_generations=5,
        samples_per_generation=40000,
        N=20,
        output_dir="results"
    )