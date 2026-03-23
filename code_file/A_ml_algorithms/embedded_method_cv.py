import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load data
df = pd.read_csv(r'E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\csv_data\hist_features_all_disease.csv')

# Prepare features and labels
X = df.drop(columns=['Name'])
y = df['Name']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Số lượng mẫu: {X.shape[0]}")
print(f"Số lượng features: {X.shape[1]}")
print(f"Các lớp: {le.classes_}")
print()

# Feature selection using Random Forest on all data
print("=== FEATURE SELECTION ===")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y_encoded)

importances_rf = rf_model.feature_importances_
indices_dt = np.argsort(importances_rf)[::-1]
features_dt = X.columns[indices_dt]

# Print feature importance statistics
num_total = len(importances_rf)
num_nonzero = (importances_rf > 0).sum()
num_zero = (importances_rf == 0).sum()

print(f"Tổng số feature: {num_total}")
print(f"Số feature có điểm > 0: {num_nonzero}")
print(f"Số feature có điểm = 0: {num_zero}")
print()

# Select top features
top_ranking_rf = 200
top_features_rf = features_dt[:top_ranking_rf]
X_selected = X[top_features_rf]

print(f"Sử dụng top {top_ranking_rf} features")
print()

# Define classifiers
classifiers = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

# 5-Fold Cross-Validation
print("=== TRAINING VỚI 5-FOLD CROSS-VALIDATION ===")
print()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, clf in classifiers.items():
    print(f"Training {name}...")
    accuracies = []
    f1_scores = []
    
    fold_num = 1
    for train_idx, test_idx in tqdm(skf.split(X_selected, y_encoded), total=5, desc=f"{name}"):
        # Split data according to fold
        X_train_fold = X_selected.iloc[train_idx]
        X_test_fold = X_selected.iloc[test_idx]
        y_train_fold = y_encoded[train_idx]
        y_test_fold = y_encoded[test_idx]
        
        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)
        
        # Training
        clf.fit(X_train_scaled, y_train_fold)
        y_pred = clf.predict(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test_fold, y_pred)
        f1 = f1_score(y_test_fold, y_pred, average='weighted')
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        fold_num += 1
    
    # Calculate average results from 5 folds
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)
    avg_accuracy = np.mean(accuracies)
    avg_f1_score = np.mean(f1_scores)
    std_accuracy = np.std(accuracies)
    std_f1_score = np.std(f1_scores)
    
    results[name] = {
        'min_accuracy': min_accuracy,
        'max_accuracy': max_accuracy,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'avg_f1_score': avg_f1_score,
        'std_f1_score': std_f1_score
    }
    
    print(f'{name}:')
    print(f'  Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f} (min: {min_accuracy:.4f}, max: {max_accuracy:.4f})')
    print(f'  F1-Score: {avg_f1_score:.4f} ± {std_f1_score:.4f}')
    print()

# Display final results
print("=== KẾT QUẢ CUỐI CÙNG ===")
print()
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('avg_accuracy', ascending=False)
print(results_df.to_string())
print()

# Find best model
best_model = results_df.index[0]
print(f"Model tốt nhất: {best_model}")
print(f"  Accuracy: {results_df.loc[best_model, 'avg_accuracy']:.4f} ± {results_df.loc[best_model, 'std_accuracy']:.4f}")
print(f"  F1-Score: {results_df.loc[best_model, 'avg_f1_score']:.4f} ± {results_df.loc[best_model,'std_f1_score']:.4f}")
