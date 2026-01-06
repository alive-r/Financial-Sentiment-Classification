import pandas as pd
import ast
import numpy as np
import json
import re
import math
from collections import Counter, defaultdict
from logistic_regression import LogisticRegression
from naive_bayes import NaiveBayes
from sklearn.model_selection import train_test_split
from simple_rnn import RecurrentNeuralNetwork

# get data from json
# 2018FiQA
def get_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rlt = []
    for df_id, obj in data.items():
        info = obj.get("info", [])[0]
        sentence = obj.get("sentence", "")
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z0-9\$\/\.\%\+\-\s]", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        # aspect_full = info.get('aspects')
        # level1 = ast.literal_eval( aspect_full)[0]

        aspect_full = info.get("aspects")

        if not aspect_full:
            level1 = None
        else:
            cleaned = aspect_full.strip("[]").strip()
            cleaned = cleaned.strip("'").strip('"')
            first_aspect = cleaned.split(",")[0].strip().strip("'").strip('"')
            parts = first_aspect.split("/")
            level1 = parts[0] if len(parts) > 0 else None
        # print('test:', info.get("sentiment_score"))  if info.get("sentiment_score") is None else ''
        original_score = info.get("sentiment_score")

        try:
            score = float( original_score)
        except:
            score = None
        rlt.append({
            "id": df_id,
            "sentence": sentence,
            "snippets": info.get("snippets"),
            "sentiment_score": score,
            "target": info.get("target"),
            "aspect_l1": level1,
        })
    return pd.DataFrame(rlt)

# 2018FiQA
def build_vocabulary(df, min_freq):
    freq = Counter()
    for s in df['sentence']:
        freq.update(s.split())
    vocab = {}
    idx = 0
    for word, count in freq.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

# new dataset
def clean_sentence(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\$\/\.\%\+\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
# new dataset
def load_new_data(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["gold_label", "sentence"])
    df["sentence"] = df["sentence"].apply(clean_sentence)
    return df

def compute_tf(sentence, vocab):
    words = sentence.split()
    word_count = Counter(words)
    tf = np.zeros(len(vocab))
    for word, idx in vocab.items():
        if word in word_count and len(words)>0:
            # tf[idx] = word_count[word] / len(words)
            tf[idx] = 1 + math.log(word_count[word])
        else:
            0
    return tf

def compute_idf(all_sentences, vocab):
    idf = np.zeros(len(vocab))
    N = len(all_sentences)

    for word, idx in vocab.items():
        s_count = 0
        for s in all_sentences:
            if word in s.split():
                s_count += 1
        idf[idx] = math.log((N + 1) / (s_count + 1)) + 1
    return idf

def compute_tfidf(df, vocab, idf, mode=True):
    if mode:
        all_sentences = df['sentence'].tolist()
        idf = compute_idf(all_sentences, vocab)
    tfidf_matrix = []
    all_sentences = df['sentence'] 
    for s in all_sentences:
        tf = compute_tf(s, vocab)
        tfidf = tf * idf
        tfidf_matrix.append(tfidf)
    return np.array(tfidf_matrix), idf

def classify_sentiment(score, low, high):
    if score < low:
        return 0  # negative
    elif score > high:
        return 2  # positive
    else:
        return 1  # neutral

def prepare_labels(df, low, high):
    labels = []
    for score in df['sentiment_score']:
        label = classify_sentiment(score, low, high)
        labels.append(label)
    return np.array(labels)

def get_accuracy(real, pred):
    return np.mean(real == pred)

def cal_macro_f1(actual, preds, num_classes=3):
    scores = []
    for cls_idx in range(num_classes):
        true_pos = np.sum((actual == cls_idx) & (preds == cls_idx))
        false_pos = np.sum((actual != cls_idx) & (preds == cls_idx))
        false_neg = np.sum((actual == cls_idx) & (preds != cls_idx))
        if (true_pos + false_pos) > 0:
            prec = true_pos / (true_pos + false_pos)
        else:
            prec = 0
        if (true_pos + false_neg) > 0:
            rec = true_pos / (true_pos + false_neg)
        else:
            rec = 0
        if (prec + rec) > 0:
            score = 2 * prec * rec / (prec + rec)
        else:
            score = 0
        scores.append(score)
    return np.mean(scores)

def cal_conf_matrix(actual, preds, num_classes=3):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    sample_count = len(actual)
    for i in range(sample_count):
        row_idx = actual[i]
        col_idx = preds[i]
        matrix[row_idx, col_idx] += 1
    return matrix

if __name__ == "__main__":
    df_headline_train = get_data("./FiQA_ABSA_task1/task1_headline_ABSA_train.json")
    df_headline_test = get_data("./FiQA_ABSA_task1_test/task1_headline_ABSA_test.json")
    df_post_train = get_data("./FiQA_ABSA_task1/task1_post_ABSA_train.json")
    df_post_test = get_data("./FiQA_ABSA_task1_test/task1_post_ABSA_test.json")

    df_train_full = pd.concat([df_headline_train, df_post_train], ignore_index=True)
    # df_train_full = df_train_full.dropna(subset=['sentiment_score'])
    df_test = pd.concat([df_headline_test, df_post_test], ignore_index=True)

    label_list = []
    sentiment_scores = df_train_full['sentiment_score']
    low = df_train_full.sentiment_score.quantile(0.33)
    high = df_train_full.sentiment_score.quantile(0.66)
    
    for score in sentiment_scores:
        label = classify_sentiment(score, low, high)
        label_list.append(label)

    #split data
    df_train, df_val = train_test_split(df_train_full, test_size = 0.2, random_state = 42, stratify = label_list)

    vocab = build_vocabulary(df_train, min_freq = 2)
    
    X_train, idf = compute_tfidf(df_train, vocab, idf=None, mode=True)
    X_val, _ = compute_tfidf(df_val, vocab, idf=idf, mode=False)
    X_test, _ = compute_tfidf(df_test, vocab, idf=idf, mode=False)

    y_train = prepare_labels(df_train, low, high)
    y_val = prepare_labels(df_val, low, high)
    

    nb_model =NaiveBayes(laplace_smoothing=0.1) 
    nb_model.fit(X_train, y_train)

    nb_train_pred = nb_model.predict(X_train)
    nb_val_pred = nb_model.predict(X_val)
    

    nb_train_acc = get_accuracy(y_train, nb_train_pred)
    nb_val_acc = get_accuracy(y_val, nb_val_pred) 


    nb_train_f1 = cal_macro_f1(y_train, nb_train_pred)
    nb_val_f1 =  cal_macro_f1(y_val, nb_val_pred)

    print(f"Naive Bayes Results:")
    print(f"Train - Acc: {nb_train_acc:.4f}, F1: {nb_train_f1:.4f}")
    print(f"Val   - Acc: {nb_val_acc:.4f}, F1: {nb_val_f1:.4f}")

    # logistic regression

    lr_model = LogisticRegression(learning_rate=0.5, max_iterations=500, regularization_strength=0.01)
    lr_model.fit(X_train, y_train)

    lr_train_pred = lr_model.predict(X_train)
    lr_val_pred = lr_model.predict(X_val)
    
    lr_train_acc = get_accuracy(y_train, lr_train_pred)
    lr_val_acc = get_accuracy(y_val, lr_val_pred)

    lr_train_f1 = cal_macro_f1(y_train, lr_train_pred)
    lr_val_f1 = cal_macro_f1(y_val, lr_val_pred)

    print(f"Logistic Regression Results:")
    print(f"Train - Acc: {lr_train_acc:.4f}, F1: {lr_train_f1:.4f}")
    print(f"Val   - Acc: {lr_val_acc:.4f}, F1: {lr_val_f1:.4f}")

    # quantitative analysis
    print("Naive Bayes Confusion Matrix:")
    nb_conf = cal_conf_matrix(y_val, nb_val_pred)
    print(nb_conf)

    print("Logistic Regression Confusion Matrix:")
    lr_conf = cal_conf_matrix(y_val, lr_val_pred)
    print(lr_conf)

    def per_class_accuracy(conf_mat):
        conf_mat = np.array(conf_mat)
        class_counts = conf_mat.sum(axis=1)
        correct = conf_mat.diagonal()
        return correct / class_counts

    nb_class_acc = per_class_accuracy(nb_conf)
    lr_class_acc = per_class_accuracy(lr_conf)

    print("\nPer-class accuracy NB:")
    print("negative: {:.3f}, neutral: {:.3f}, positive: {:.3f}"
        .format(nb_class_acc[0], nb_class_acc[1], nb_class_acc[2]))

    print("Per-class accuracy LR:")
    print("negative: {:.3f}, neutral: {:.3f}, positive: {:.3f}"
        .format(lr_class_acc[0], lr_class_acc[1], lr_class_acc[2]))
    
    nb_correct = (y_val == nb_val_pred)
    lr_correct = (y_val == lr_val_pred)

    both_correct = np.sum(nb_correct & lr_correct)
    nb_only = np.sum(nb_correct & ~lr_correct)
    lr_only = np.sum(~nb_correct & lr_correct)
    both_wrong = np.sum(~nb_correct & ~lr_correct)

    print("\nComparison between NB and LR:")
    print("Both correct:", both_correct)
    print("NB only correct:", nb_only)
    print("LR only correct:", lr_only)
    print("Both wrong:", both_wrong)


    scores_val = df_val["sentiment_score"].values
    strong_mask = np.abs(scores_val) >= 0.4
    weak_mask = ~strong_mask

    def masked_accuracy(y_true, y_pred, mask):
        return np.mean(y_true[mask] == y_pred[mask])

    nb_acc_strong = masked_accuracy(y_val, nb_val_pred, strong_mask)
    nb_acc_weak   = masked_accuracy(y_val, nb_val_pred, weak_mask)
    lr_acc_strong = masked_accuracy(y_val, lr_val_pred, strong_mask)
    lr_acc_weak   = masked_accuracy(y_val, lr_val_pred, weak_mask)

    print("\nstrong-sentiment vs weak-sentiment:")
    print("NB strong: {:.3f}, NB weak: {:.3f}".format(nb_acc_strong, nb_acc_weak))
    print("LR strong: {:.3f}, LR weak: {:.3f}".format(lr_acc_strong, lr_acc_weak))


    # --- 
    nb_test_pred = nb_model.predict(X_test)
    lr_test_pred = lr_model.predict(X_test)
    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    nb_test_label = [label_map[p] for p in nb_test_pred]
    lr_test_label = [label_map[p] for p in lr_test_pred]
    test_df = pd.DataFrame({
        "id": df_test["id"].values,
        "sentence": df_test["sentence"].values,
        "nb_pred": nb_test_label,
        "lr_pred": lr_test_label
    })
    print(test_df.head())
    test_df.to_csv("test_prediction.csv", index=False)

    