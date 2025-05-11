#!/usr/bin/env python3
import argparse
import json
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fine_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path):
    """Load training data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def prepare_dataset(data, tokenizer, max_length=512):
    """Prepare dataset for fine-tuning."""
    # Format data for instruction fine-tuning
    formatted_data = []
    
    for example in data:
        # Format as instruction with quality weighting
        quality = example.get('quality', 1.0)  # Default to high quality if not specified
        
        # We'll repeat high-quality examples to give them more weight
        repeat_count = 3 if quality > 0.8 else (2 if quality > 0.5 else 1)
        
        for _ in range(repeat_count):
            formatted_data.append({
                "text": f"Dream: {example['input']}\n\nInterpretation: {example['output']}"
            })
    
    # Convert to Hugging Face dataset
    dataset = Dataset.from_dict({"text": [item["text"] for item in formatted_data]})
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

def fine_tune(args):
    """Fine-tune the model on the provided data."""
    logger.info(f"Starting fine-tuning process")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training data: {args.training_data}")
    
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU available, using CPU (this will be slow)")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset
    logger.info("Loading and preparing dataset...")
    data = load_jsonl(args.training_data)
    logger.info(f"Loaded {len(data)} training examples")
    
    if len(data) < 10:
        logger.warning("Very small dataset. Fine-tuning may not be effective.")
    
    train_dataset = prepare_dataset(data, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=device.type == "cuda",
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model on dream interpretation data")
    parser.add_argument("--base_model", type=str, default="TheBloke/Llama-2-7B-Chat-GGUF", 
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--training_data", type=str, required=True, 
                        help="Path to the JSONL training data file")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=500, 
                        help="Save checkpoint every X steps")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    fine_tune(args) 