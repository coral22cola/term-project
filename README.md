# Open Source Term Project

---

## 1. Outline of the project

#### 1). Project Name
   - Translator using Hugging Face datasets.
#### 2). Reason of choice
   - We want to make a translator using vast amounts of data.
#### 3). Goal
   - Create a translator using a vast dataset of hugging face and translate your own sentences.
#### 4). Expectation & Resutlt
   - The expected translation was done well, and the various example questions and direct questions were interpreted well.

---

## 2. How to run

#### 1). Setting the Model.
   - Used Model: Helsinki-NLP/opus-mt-ko-en · Hugging Face
#### 2). Finding the Dataset.
   - Used Dataset: lemon-mint/korean_english_parallel_wiki_augmented_v1 · Datasets at Hugging Face
#### 3). Run the database_operation.py
   - Use the dataset_operation.py file with 1 and 2 tasks.
#### 4). Enter sentences
   - sentences = [ "Write the Sentence in Korean"]
#### 5). Run the Final_Translator.py
   - Once you have entered the sentence, translate it through Final_Translator.py.
   - After Running the program, the translated sentence should shown on terminal. 

---

## 3. Used Packages
   - torch, transformers, datasets, sentencepiece, accelerate

---

## 4. Execution images

<img width="1795" height="715" alt="524795196-dc27236b-1c98-4229-aad3-92f6254679c5" src="https://github.com/user-attachments/assets/f8ce24f1-80c2-411c-ba3c-641267c443a8" />

---

<img width="1777" height="461" alt="524795434-26e5f73b-687d-4652-9fb1-93911929624a" src="https://github.com/user-attachments/assets/08939499-3621-439c-9582-a79717416f1c" />

---

## 5. Reference
   - https://huggingface.co/Helsinki-NLP/opus-mt-ko-en?library=transformers
   - https://huggingface.co/datasets/lemon-mint/korean_english_parallel_wiki_augmented_v1?library=datasets
   - explain : The other translator models we referenced
