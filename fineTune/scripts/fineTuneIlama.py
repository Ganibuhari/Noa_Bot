import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load the text dataset
dataset = load_dataset('text', data_files={'train': 'all_text_data.txt'})

# Load the LLaMA tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['text'])

# Define training arguments and checkpointing strategy
output_dir = './llama_finetune_checkpoints'
training_args = TrainingArguments(
    output_dir=output_dir,                     
    overwrite_output_dir=True,                 
    num_train_epochs=3,                        
    per_device_train_batch_size=2,             
    save_steps=500,                            
    save_total_limit=2,                        
    logging_steps=100,                         
    save_strategy="steps",                     
    logging_dir=f'{output_dir}/logs',          
    report_to="none",                          
    load_best_model_at_end=False,              
    evaluation_strategy="no",                  
    fp16=True if torch.cuda.is_available() else False,  
)

# Dynamic checkpoint naming function
def save_checkpoint_callback(trainer, output_dir):
    step = trainer.state.global_step
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    trainer.save_model(checkpoint_dir)
    print(f"Saved checkpoint at step {step}: {checkpoint_dir}")

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    tokenizer=tokenizer
)

# Train and dynamically save checkpoints
trainer.add_callback(save_checkpoint_callback)
trainer.train()

# Save the final model
model.save_pretrained(f"{output_dir}/final_model")
tokenizer.save_pretrained(f"{output_dir}/final_model")
