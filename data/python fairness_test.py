import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# AIF360 imports
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# ------------------------------------------------------------
# 1. Load COMPAS dataset
# ------------------------------------------------------------
df = pd.read_csv("compas.csv")

# ------------------------------------------------------------
# 2. Prepare dataset
# ------------------------------------------------------------
# Create binary protected attribute: 1 = Black, 0 = Non-Black
df["race_black"] = (df["race"] == "African-American").astype(int)

# Create binary label: 1 = high recidivism, 0 = low
df["label"] = (df["two_year_recid"] == 1).astype(int)

# ------------------------------------------------------------
# 3. Convert to AIF360 dataset format
# ------------------------------------------------------------
bld = BinaryLabelDataset(
    df=df,
    label_names=['label'],
    protected_attribute_names=['race_black'],
    favourable_label=0,
    unfavourable_label=1
)

# Split into train-test
train, test = bld.split([0.7], shuffle=True)

# ------------------------------------------------------------
# 4. Baseline fairness metrics
# ------------------------------------------------------------
metric_test = BinaryLabelDatasetMetric(
    test,
    privileged_groups=[{'race_black': 0}],
    unprivileged_groups=[{'race_black': 1}]
)

print("=== Baseline Fairness Metrics ===")
print("Disparate impact:", metric_test.disparate_impact())
print("Selection rate privileged:", metric_test.selection_rate(privileged=True))
print("Selection rate unprivileged:", metric_test.selection_rate(privileged=False))
print("\n")

# ------------------------------------------------------------
# 5. Train Logistic Regression model
# ------------------------------------------------------------
features = ['priors_count', 'decile_score', 'age']  # adjust if different in your file
X = df[features].values
y = df['label'].values

Xtr, Xte, ytr, yte, df_tr, df_te = train_test_split(
    X, y, df, test_size=0.3, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, ytr)

# Predict
y_pred = clf.predict(Xte)

# ------------------------------------------------------------
# 6. Wrap predictions in AIF360 format
# ------------------------------------------------------------
test_pred = test.copy()
test_pred.labels = y_pred.reshape(-1, 1)

# ------------------------------------------------------------
# 7. Compute fairness of model predictions
# ------------------------------------------------------------
cm = ClassificationMetric(
    test,
    test_pred,
    privileged_groups=[{'race_black': 0}],
    unprivileged_groups=[{'race_black': 1}]
)

print("=== Prediction Fairness Metrics ===")
print("False Positive Rate difference:", cm.false_positive_rate_difference())
print("True Positive Rate difference:", cm.true_positive_rate_difference())
print("Accuracy difference:", cm.accuracy_difference())
