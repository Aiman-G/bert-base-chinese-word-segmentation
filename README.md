---
license: apache-2.0
language:
- zh
base_model:
- google-bert/bert-base-chinese
---
# 🐼 Chinese BERT for Word Segmentation

**Model Name:** `your-username/bert-chinese-segmentation-pku`  
**Model link** :  (Here)[https://huggingface.co/AimanGh/bert-base-chinese-word-segmentation]
**Language:** Chinese 🇨🇳  
**Task:** Chinese Word Segmentation

---

##  Model Description

This is a BERT-based model fine-tuned for **Chinese Word Segmentation** using the **PKU (Peking University) dataset**.  
It splits raw Chinese text into meaningful words — an essential preprocessing step for many downstream NLP tasks such as NER, sentiment analysis, and machine translation.

---

##  Fine-Tuning Results

The model was fine-tuned for **2 epochs** with the following training and validation metrics:

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 |
|-------|----------------|-----------------|-----------|--------|-----|
| 1     | 0.031600       | 0.024586        | 0.9800    | 0.9787 | 0.9793 |
| 2     | 0.017700       | 0.022133        | 0.9836    | 0.9823 | 0.9829 |

After training, the model was evaluated on the **PKU Gold Test** set:

| Metric    | Score  |
|-----------|--------|
| Precision | 0.9919 |
| Recall    | 0.9796 |
| F1        | 0.9857 |

✅ **Benchmark:** The test file from the PKU dataset was segmented by this model and compared against the official segmented version (gold test).

---

## 📂 Dataset

- **Training dataset:** PKU (Peking University) Chinese word segmentation dataset
- **Evaluation dataset:** PKU Gold Test set

---

## 🚀 How to Use

Load the model using 🤗 Transformers and segment raw Chinese text:

```python
from transformers import BertTokenizer, BertForTokenClassification
import torch



label_list = ["B", "I"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(label_list)

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("path")
model = BertForTokenClassification.from_pretrained("path")

def segment_sentence(sentence, tokenizer, model, id2label):
    """
    Segment a single sentence using the fine-tuned model, excluding special tokens.
    """
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", is_split_into_words=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    labels = [id2label[pred] for pred in predictions]
    
    # remove special tokens 
    filtered_tokens = tokens[1:-1]
    filtered_labels = labels[1:-1]
    
    # combine tokens into segmented sentence
    segmented_sentence = ""
    for token, label in zip(filtered_tokens, filtered_labels):
        if token.startswith("##"):  # Handle subwords
            segmented_sentence += token[2:]
        else:
            if label == "B" and segmented_sentence:  # add a space before a new word
                segmented_sentence += " "
            segmented_sentence += token
    
    return segmented_sentence


test_sentence = "芜湖如诗如画，青山环抱，江水悠悠"

segmented_output = segment_sentence(test_sentence, saved_tokenizer, loaded_model, id2label)
print(f"Segmented Sentence: {segmented_output}")


```

## Limitations and Biases

* **Training Data Dependency:** The model's performance is highly dependent on the quality and characteristics of the fine-tuning dataset. Performance on out-of-domain text or specific styles (e.g., highly colloquial, ancient Chinese, specific technical jargons not present in the training data) might vary.
* **Ambiguity:** chinese is a languge of poetry. It is highly idiomic langauge. Therefore, word segmentation inherently deals with ambiguities (e.g., words that can be segmented in multiple valid ways depending on context). While BERT helps, it doesn't eliminate all ambiguity.
* **Bias:** Like all models trained on large text corpora, this model may inherit biases present in its pre-training and fine-tuning data. Users should be aware of potential biases in segmentation, especially in sensitive domains.


