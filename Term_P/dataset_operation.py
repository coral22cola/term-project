import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import os
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb

# Configuration
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-ko-en"
DATASET_NAME = "lemon-mint/korean_english_parallel_wiki_augmented_v1"
OUTPUT_DIR = "./ko-en-finetuned-model"

# Load Dataset
print("Dataset Loading")
raw_datasets = load_dataset(DATASET_NAME)

# Split validation dataset
if 'validation' not in raw_datasets:
    split = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
    raw_datasets = split
    raw_datasets['validation'] = raw_datasets.pop('test')

# Preprocessing
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def preprocess_function(examples):
    inputs = []
    targets = []

    # [Fix 1] Handle cases where the dataset structure is {'translation': {'ko': '...', 'en': '...'}}
    if "translation" in examples:
        inputs = [ex['ko'] for ex in examples["translation"]]
        targets = [ex['en'] for ex in examples["translation"]]
    else:
        
        inputs = examples.get('ko', examples.get('korean', []))
        targets = examples.get('en', examples.get('english', []))

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Data Preprocessing...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# Model and Training Configuration
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",  # [Fix 2] Changed evaluation_strategy to eval_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,     # Train for only 1 epoch for quick testing
    predict_with_generate=True,
    fp16=True if torch.cuda.is_available() else False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start Training
print("Training Start!")
trainer.train()

# Save Model
save_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model has been saved: {save_path}")


