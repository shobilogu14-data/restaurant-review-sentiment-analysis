#RestaurantReviewAnalysis

# STEP 1: Setup and Imports

import warnings
warnings.filterwarnings('ignore')

# Random seeds and configs
RANDOM_SEED = 42
TFIDF_MAX_FEATURES = 5000
ROBERTA_MAX_LEN = 128
ROBERTA_EPOCHS = 3
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 32

# Core libraries
# Colab-only command removed: install dependencies separately using pip.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, os

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    cohen_kappa_score, confusion_matrix, precision_recall_curve,
    ConfusionMatrixDisplay
)
from scipy.stats import ttest_rel

# Transformers and explainability
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import shap
from lime.lime_text import LimeTextExplainer

# Output dirs
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# === (COLAB ONLY) UPLOAD FILE ===
try:
    from google.colab import files
    uploaded = files.upload()
except ImportError:
    print("✅ Not using Colab. Proceeding without file upload.")

# ==========================================
# STEP 2: Load and Preprocess Data
# ==========================================

# Load Excel file containing restaurant reviews
df = pd.read_excel("restaurant_reviews.xlsx")

# Keep only necessary columns and drop rows with missing values
df = df[['text', 'span', 'label', 'ordinal']].dropna()

# --- Function to clean review text ---
def clean_text(txt):
    txt = str(txt).lower()
    txt = re.sub(r"<.*?>", "", txt)            # remove HTML tags
    txt = re.sub(r"[^a-zA-Z\s]", "", txt)      # keep only alphabets
    txt = re.sub(r"\s+", " ", txt).strip()     # remove extra spaces
    return txt

# Apply cleaning function
df['clean_review'] = df['text'].apply(clean_text)

# Normalize aspect column
df['aspect'] = df['span'].str.lower().str.strip()

# Keep only positive/negative sentiments (binary classification)
df = df[df['label'].isin(['positive', 'negative'])]
df['sentiment'] = df['label'].map({'positive': 1, 'negative': 0})

print("Sample of cleaned dataset:")
print(df.head())

# ==========================================
# STEP 3: Exploratory Data Analysis (EDA)
# ==========================================

# --- Word Cloud of reviews ---
text_blob = " ".join(df['clean_review'].dropna())
wc = WordCloud(width=800, height=400, background_color='white').generate(text_blob)

plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Reviews")
plt.tight_layout()
plt.savefig("outputs/plots/wordcloud.png")
plt.show()

# --- Sentiment distribution ---
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='label')
plt.title("Sentiment Distribution")
plt.tight_layout()
plt.savefig("outputs/plots/sentiment_distribution.png")
plt.show()

# --- Review length distribution ---
df['review_len'] = df['clean_review'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8,5))
sns.histplot(df['review_len'], bins=30)
plt.title("Review Length Distribution")
plt.tight_layout()
plt.savefig("outputs/plots/review_length_distribution.png")
plt.show()

# --- Top 20 aspects mentioned ---
aspect_counts = df['aspect'].value_counts().head(20)
plt.figure(figsize=(12,6))
sns.barplot(x=aspect_counts.index, y=aspect_counts.values)
plt.xticks(rotation=45)
plt.title("Top 20 Aspects in Reviews")
plt.tight_layout()
plt.savefig("outputs/plots/top_20_aspects.png")
plt.show()

# --- Aspect vs sentiment matrix ---
top_aspects = df['aspect'].value_counts().head(10).index.tolist()
subset_df = df[df['aspect'].isin(top_aspects)]
aspect_sent = pd.crosstab(subset_df['aspect'], subset_df['label'])

aspect_sent.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title("Sentiment Count per Top Aspect")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/plots/aspect_sentiment_matrix.png")
plt.show()

# --- Average sentiment score per aspect ---
avg_sent = df.groupby('aspect')['sentiment'].mean().sort_values(ascending=False).head(15)
avg_sent.to_csv("outputs/tables/avg_sentiment_per_aspect.csv")

plt.figure(figsize=(12,6))
sns.barplot(x=avg_sent.index, y=avg_sent.values)
plt.title("Average Sentiment per Aspect")
plt.ylabel("Avg Sentiment (1=Positive, 0=Negative)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/plots/avg_sentiment_per_aspect.png")
plt.show()

# --- SWOT-style aspect summary ---
print("\nAspect Sentiment Summary (SWOT-style):")
for asp, score in avg_sent.items():
    if score > 0.75:
        print(f"Strength → {asp} (high positive sentiment, {score:.2f})")
    elif score < 0.35:
        print(f"Weakness → {asp} (low positive sentiment, {score:.2f})")
    else:
        print(f"Neutral → {asp} (mixed sentiment, {score:.2f})")

# ==========================================
# STEP 4: Multi-label Aspect Classification (Logistic Regression)
# ==========================================

# --- Group aspects per review (for multi-label learning) ---
grouped = df.groupby('text').agg({
    'aspect': lambda x: list(set(x)),      # unique aspects in review
    'sentiment': lambda x: list(x)[0],     # take first sentiment
    'clean_review': 'first'
}).reset_index()

# --- Binary encode aspects ---
mlb = MultiLabelBinarizer()
y_aspect = mlb.fit_transform(grouped['aspect'])

# --- TF-IDF vectorization ---
X_aspect = grouped['clean_review']
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
X_aspect_vect = tfidf.fit_transform(X_aspect)

# --- Train-test split ---
Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    X_aspect_vect, y_aspect, test_size=0.2, random_state=RANDOM_SEED
)

# --- Train Logistic Regression classifier ---
clf_aspect = OneVsRestClassifier(LogisticRegression(max_iter=500, random_state=RANDOM_SEED))
clf_aspect.fit(Xa_train, ya_train)
ya_pred = clf_aspect.predict(Xa_test)

# --- Classification Report ---
aspect_report = classification_report(
    ya_test, ya_pred, target_names=mlb.classes_, output_dict=True
)
aspect_report_df = pd.DataFrame(aspect_report).transpose()
aspect_report_df.to_csv("outputs/tables/aspect_classification_report.csv")

print("\nAspect Classification Report (sample shown):")
print(aspect_report_df.head())

# --- 5-Fold Cross-Validation (micro-F1) ---
print("\nCross-Validation (5-Fold, F1-micro):")
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(clf_aspect, X_aspect_vect, y_aspect, cv=kf, scoring='f1_micro')

print(f"Scores: {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==========================================
# Extended Evaluation Metrics (for robustness)
# ==========================================
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# --- Macro and Micro F1 ---
f1_macro = f1_score(ya_test, ya_pred, average='macro', zero_division=0)
f1_micro = f1_score(ya_test, ya_pred, average='micro', zero_division=0)

print(f"\nMacro-F1: {f1_macro:.4f}")
print(f"Micro-F1: {f1_micro:.4f}")

# --- Hamming Loss ---
hloss = hamming_loss(ya_test, ya_pred)
print(f"Hamming Loss: {hloss:.4f}")

# --- Jaccard Index (sample-based) ---
jaccard = jaccard_score(ya_test, ya_pred, average='samples', zero_division=0)
print(f"Jaccard Index (sample-level): {jaccard:.4f}")

# --- Precision-Recall (PR-AUC) curves for top aspects ---
# Get predicted probabilities for PR curves
ya_pred_prob = np.array([estimator.predict_proba(Xa_test)[:, 1] for estimator in clf_aspect.estimators_]).T

aspect_names = mlb.classes_
support_counts = ya_test.sum(axis=0)  # frequency of each aspect in test set
top_aspects = np.argsort(support_counts)[::-1][:5]  # pick top 5 frequent aspects

plt.figure(figsize=(12, 8))
for i in top_aspects:
    precision, recall, _ = precision_recall_curve(ya_test[:, i], ya_pred_prob[:, i])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"{aspect_names[i]} (PR-AUC={pr_auc:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for Top-5 Frequent Aspects")
plt.legend()
plt.grid()
plt.savefig("outputs/plots/aspect_pr_curves.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================
# STEP 5: Binary Sentiment Classification with Classical Models
# ==========================================

# Prepare data for binary classification
Xs = df['clean_review']
ys = df['sentiment']

# Train-test split
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    Xs, ys, test_size=0.2, random_state=RANDOM_SEED, stratify=ys
)

# Vectorize text
vec_sent = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
Xs_train_vect = vec_sent.fit_transform(Xs_train)
Xs_test_vect = vec_sent.transform(Xs_test)

# Evaluation function with threshold tuning
def eval_classical(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]

    # Threshold tuning
    precs, recs, ths = precision_recall_curve(y_test, probs)
    f1s = 2 * (precs * recs) / (precs + recs + 1e-6)
    best_idx = np.argmax(f1s)
    best_thr = ths[best_idx] if best_idx < len(ths) else 0.5
    preds = (probs >= best_thr).astype(int)

    return {
        'name': model.__class__.__name__,
        'f1': f1_score(y_test, preds),
        'auc': roc_auc_score(y_test, probs),
        'kappa': cohen_kappa_score(y_test, preds),
        'thr': best_thr,
        'probs': probs,
        'preds': preds,
        'model': model
    }

# Models to evaluate (5 classical)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = [
    MultinomialNB(),
    SVC(probability=True, kernel='linear', random_state=RANDOM_SEED),
    LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
]

# Evaluate and collect results
all_results = []
for m in models:
    res = eval_classical(m, Xs_train_vect, ys_train, Xs_test_vect, ys_test)
    all_results.append(res)

# Print results
print("\nThreshold-tuned Evaluation on Hold-Out Test Set:")
for r in all_results:
    print(f"{r['name']}: F1={r['f1']:.4f}, AUC={r['auc']:.4f}, "
          f"Kappa={r['kappa']:.4f}, Best Thr={r['thr']:.2f}")

# 5-Fold Cross-Validation
print("\n5-Fold Cross-Validation (Macro F1 Score):")
for m in models:
    scores = cross_val_score(m, vec_sent.transform(Xs), ys, cv=5, scoring='f1_macro')
    print(f"{m.__class__.__name__}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

# ==========================================
# STEP 6: RoBERTa Fine-Tuning and Evaluation
# ==========================================

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=ROBERTA_MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Dataset and loaders
train_ds = ReviewDataset(Xs_train.tolist(), ys_train.tolist(), tokenizer)
test_ds = ReviewDataset(Xs_test.tolist(), ys_test.tolist(), tokenizer)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_TEST)

# Device and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
roberta.to(device)
optim = AdamW(roberta.parameters(), lr=2e-5)

# Training loop
roberta.train()
for ep in range(ROBERTA_EPOCHS):
    for batch in train_loader:
        optim.zero_grad()
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        out = roberta(ids, attention_mask=mask, labels=labels)
        loss = out.loss
        loss.backward()
        optim.step()

# Evaluation
roberta.eval()
all_probs, all_true = [], []
with torch.no_grad():
    for batch in test_loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()
        out = roberta(ids, attention_mask=mask)
        logits = out.logits
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        all_probs.extend(probs)
        all_true.extend(labels)

# Threshold tuning
all_probs = np.array(all_probs)
all_true = np.array(all_true)
ths = np.linspace(0.1, 0.9, 81)
f1s = [f1_score(all_true, (all_probs >= t).astype(int)) for t in ths]
best_idx = np.argmax(f1s)
best_thr_roberta = ths[best_idx]
final_preds = (all_probs >= best_thr_roberta).astype(int)

# Metrics
f1_r = f1_score(all_true, final_preds)
auc_r = roc_auc_score(all_true, all_probs)
kappa_r = cohen_kappa_score(all_true, final_preds)

# Append
all_results.append({
    'name': 'RoBERTa',
    'f1': f1_r,
    'auc': auc_r,
    'kappa': kappa_r,
    'thr': best_thr_roberta,
    'probs': all_probs,
    'preds': final_preds,
    'model': roberta
})

print("\nRoBERTa Evaluation:")
print(f"F1={f1_r:.4f}, AUC={auc_r:.4f}, Kappa={kappa_r:.4f}, Best Thr={best_thr_roberta:.2f}")

# ==========================================
# STEP 7: Statistical Significance Testing
# ==========================================

print("\nPaired t-test: RoBERTa vs Classical Models")

# Find RoBERTa results
res_roberta = [r for r in all_results if r['name']=='RoBERTa'][0]
roberta_probs = res_roberta['probs']
roberta_f1 = res_roberta['f1']

for r in all_results:
    if r['name']=='RoBERTa':
        continue
    t_stat, p_val = ttest_rel(roberta_probs, r['probs'])
    print(f"\nRoBERTa vs {r['name']}")
    print(f" - t-test → t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("   → Significant (p<0.05)")
    else:
        print("   → Not significant (p≥0.05)")
    print(f" - F1 Scores → RoBERTa={roberta_f1:.4f}, {r['name']}={r['f1']:.4f}")

# ==========================================
# STEP 8: Final Model Comparison & Visualization
# ==========================================

# ------------------------------
# 8.1 Aspect Classification (Logistic Regression, Multi-label)
# ------------------------------

# Collect evaluation metrics from Step 4
aspect_results = {
    "Macro-F1": f1_score(ya_test, ya_pred, average='macro'),
    "Micro-F1": f1_score(ya_test, ya_pred, average='micro'),
    "Hamming Loss": hamming_loss(ya_test, ya_pred),
    "Jaccard Score": jaccard_score(ya_test, ya_pred, average='samples')
}

# Save results to table
aspect_df = pd.DataFrame([aspect_results])
aspect_df.to_csv("outputs/tables/aspect_summary.csv", index=False)

print("\nAspect Classification Performance Summary (Multi-label Logistic Regression):")
print(aspect_df.to_string(index=False))

# Plot bar chart for aspect metrics
plt.figure(figsize=(7,5))
sns.barplot(data=aspect_df.melt(var_name="Metric", value_name="Score"),
            x="Metric", y="Score", palette="viridis")
plt.title("Aspect Classification Metrics")
plt.ylabel("Score")
plt.xlabel("Metric")
plt.tight_layout()
plt.savefig("outputs/plots/aspect_classification_metrics.png")
plt.show()

# Precision-Recall Curves for frequent aspects
common_aspects = ['food','service']
for asp in common_aspects:
    if asp in mlb.classes_:
        idx = list(mlb.classes_).index(asp)
        precision, recall, _ = precision_recall_curve(ya_test[:, idx], ya_pred[:, idx])

        plt.figure(figsize=(6,5))
        plt.plot(recall, precision, marker='.')
        plt.title(f"PR Curve for Aspect: {asp}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(f"outputs/plots/pr_curve_{asp}.png")
        plt.show()

# 8.2 Binary Sentiment Classification (NB, SVC, LR, RF, XGB) + RoBERTa

# Create summary DataFrame for sentiment models
summary_df = pd.DataFrame(all_results)
summary_df = summary_df[['name','f1','auc','kappa','thr']]
summary_df.columns = ['Model','F1 Score','AUC-ROC',"Cohen's Kappa",'Best Threshold']
summary_df.to_csv("outputs/tables/model_summary.csv", index=False)

print("\nFinal Model Performance Summary (Binary Sentiment Classification):")
print(summary_df.to_string(index=False))

# Bar plots for sentiment models
def plot_metric_bar(data, metric, ylabel):
    plt.figure(figsize=(8,5))
    sns.barplot(data=data, x='Model', y=metric, palette='Set2')
    plt.title(f"{metric} Comparison")
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{metric.lower().replace(' ','_')}_comparison.png")
    plt.show()

plot_metric_bar(summary_df, 'F1 Score', 'F1 Score')
plot_metric_bar(summary_df, 'AUC-ROC', 'Area Under ROC')
plot_metric_bar(summary_df, "Cohen's Kappa", "Cohen's Kappa")

# Confusion matrices for sentiment models
for res in all_results:
    true_labels = ys_test if res['name']!='RoBERTa' else all_true
    cm = confusion_matrix(true_labels, res['preds'])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative','Positive'],
                yticklabels=['Negative','Positive'])
    plt.title(f"Confusion Matrix - {res['name']}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"outputs/plots/cm_{res['name'].lower()}.png")
    plt.show()

# ==========================================
# STEP 9: Error Analysis – Misclassified Samples by RoBERTa
# ==========================================

# Extract RoBERTa results explicitly
res_roberta = [r for r in all_results if r['name']=='RoBERTa'][0]

# Build DataFrame for test set results
df_test_sent = pd.DataFrame({
    'review': Xs_test.reset_index(drop=True),
    'true_label': ys_test.reset_index(drop=True),
    'predicted_label': res_roberta['preds'],
    'probability': res_roberta['probs']
})

# Filter misclassified samples
misclassified = df_test_sent[df_test_sent['true_label'] != df_test_sent['predicted_label']]
misclassified['confidence_error'] = np.abs(misclassified['probability'] - 0.5)

# Top 10 hardest misclassifications
top_misclassified = misclassified.sort_values(by='confidence_error', ascending=False).head(10)

# Save to CSV
top_misclassified.to_csv("outputs/tables/top_misclassified_roberta.csv", index=False)

print("\nTop 10 Misclassified Reviews (RoBERTa):")
print(top_misclassified[['review','true_label','predicted_label']])

# ==========================================
# STEP 10: Model Explainability with SHAP and LIME
# ==========================================

print("\nSHAP and LIME Explanation for RoBERTa")

# Prediction function for SHAP/LIME
def roberta_predict_proba(texts):
    roberta.eval()
    dataset = ReviewDataset(texts, [0]*len(texts), tokenizer)
    loader = DataLoader(dataset, batch_size=4)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out = roberta(ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)

# --- SHAP Explanation ---
explainer = shap.Explainer(roberta_predict_proba, tokenizer)
shap_values = explainer(Xs_test[:5].tolist())  # take first 5 samples

# Visualize SHAP for first sample
shap.plots.text(shap_values[0])

# --- LIME Explanation ---
print("\nLIME Explanation for one misclassified review")

# Pick one misclassified sample
sample_review = top_misclassified.iloc[0]['review']
class_names = ['Negative','Positive']

lime_explainer = LimeTextExplainer(class_names=class_names)
exp = lime_explainer.explain_instance(sample_review, roberta_predict_proba, num_features=10)

# Save explanation to HTML
exp.save_to_file("outputs/tables/lime_explanation.html")
print("LIME explanation saved as: outputs/tables/lime_explanation.html")






















