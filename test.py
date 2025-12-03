import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np

# ---------------------------------------------------------
# 1. 설정 (Configuration)
# ---------------------------------------------------------
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-ko-en"
DATASET_NAME = "lemon-mint/korean_english_parallel_wiki_augmented_v1"
OUTPUT_DIR = "./ko-en-finetuned-model"

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------
# 2. 데이터셋 로드 및 분할
# ---------------------------------------------------------
# 데이터셋 로드
raw_datasets = load_dataset(DATASET_NAME)

# 데이터셋 구조 확인 후, train/test 분할 (만약 분할되어 있지 않다면)
# lemon-mint 데이터셋은 보통 'train'만 있을 수 있으므로 나눕니다.
if 'validation' not in raw_datasets:
    split_datasets = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
    raw_datasets = split_datasets
    # 편의상 'test'를 'validation'으로 사용
    raw_datasets['validation'] = raw_datasets.pop('test')

print(raw_datasets)

# ---------------------------------------------------------
# 3. 토크나이저 및 전처리
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# 입력(한국어)와 타겟(영어)의 최대 길이 설정
max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    # 데이터셋의 컬럼 이름을 확인하고 수정해야 할 수 있습니다.
    # 보통 이 데이터셋은 'ko', 'en' 혹은 'korean', 'english' 키를 가집니다.
    # 여기서는 데이터셋을 확인하고 적절한 키를 매핑합니다.
    inputs = [ex for ex in examples['korean']] # 데이터셋 컬럼명이 'ko'라고 가정
    targets = [ex for ex in examples['english']] # 데이터셋 컬럼명이 'en'이라고 가정
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # 타겟(정답) 토크나이징
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 전처리 적용
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# ---------------------------------------------------------
# 4. 모델 및 평가 지표 로드
# ---------------------------------------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # 레이블에서 -100은 무시 (Loss 계산 시 사용된 패딩 토큰)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# ---------------------------------------------------------
# 5. 학습 설정 (Training Arguments)
# ---------------------------------------------------------
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # GPU 메모리에 따라 조절 (16 or 32)
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,              # 데이터 양에 따라 조절
    predict_with_generate=True,
    fp16=True,                       # GPU 사용 시 True (속도 향상)
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ---------------------------------------------------------
# 6. 트레이너 초기화 및 학습 시작
# ---------------------------------------------------------
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# ---------------------------------------------------------
# 7. 모델 저장
# ---------------------------------------------------------
trainer.save_model(OUTPUT_DIR + "/final_model")
tokenizer.save_pretrained(OUTPUT_DIR + "/final_model")
print("Model saved successfully!")