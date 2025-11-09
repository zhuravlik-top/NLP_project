import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import re

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[^a-zа-яё\s]', ' ', s)     # оставляем только буквы и пробелы
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# --- Функция токенизации ---
def tokenize_batch(batch, tokenizer=None, max_length=256):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=max_length
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    f1 = f1_score(labels, preds, average="macro")
    return {"f1_macro": f1}
