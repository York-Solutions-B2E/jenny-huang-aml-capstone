import os
import pandas as pd
from google.cloud import aiplatform
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_results(data, target, endpt):
    data = data.astype(str)
    X = data.drop(target, axis = 1)
    y = data[target]
    y_pred = pd.Series()
    for i in range(len(data)):
        test = dict(X.iloc[i])
        try:
            results = endpt.predict([test])
            if results[0][0]['scores'][0] > 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        except Exception: 
            print(f"Skipping{i}")
            continue
    return y.astype(int), y_pred

def get_stats(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    re = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    return [acc, pre, re, f]

def get_random_data(n_rows, *files):
    final_data = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f)
        df = df.sample(frac=1).reset_index(drop=True) #shuffle
        rows = df[:n_rows]
        final_data = pd.concat([final_data, rows]).reset_index(drop=True)
    return final_data