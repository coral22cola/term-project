import torch                                                         
from transformers import pipeline
import os
import sys


# 1. Set model path (with safety check)

# Expected path where your fine-tuned model is stored
finetuned_model_path = "./ko-en-finetuned-model/final_model"
# Base model to use if the fine-tuned model is not found
base_model_name = "Helsinki-NLP/opus-mt-ko-en"

# Check whether the fine-tuned model actually exists
if os.path.exists(finetuned_model_path):
    print(f"✅ Fine-tuned model found! Path: {finetuned_model_path}")
    target_model = finetuned_model_path
else:
    print(f"⚠️Fine-tuned model not found. (Path: {finetuned_model_path})")
    print(f"👉 Falling back to the base model: {base_model_name}")
    target_model = base_model_name

# 2. Initialize translator

print("⏳Loading translation model...")
device = 0 if torch.cuda.is_available() else -1

try:
    translator = pipeline(
        "translation",
        model=target_model,
        tokenizer=target_model,
        device=device
    )
except Exception as e:
    print(f"\n ❌Fatal error occurred: {e}")
    sys.exit()


# 3. Sentences to test translation

sentences = [
    "C언어를 공부하는 것은 매우 즐겁습니다."
    "시험이 빨리 끝나면 좋을 것 같습니다."
]


# 4. Run translation & print results

print("\n" + "="*50)
print(f"   [ Translation Test Results ] (Model used: {target_model})")
print("="*50)

for text in sentences:
    # Perform translation
    result = translator(text)
    translated_text = result[0]['translation_text']

    # Print result
    print(f"🇰🇷 Korean Input : {text}")
    print(f"🇺🇸 English Output : {translated_text}")
    print("-" * 50)

print("\n ✅Test Completed!")



