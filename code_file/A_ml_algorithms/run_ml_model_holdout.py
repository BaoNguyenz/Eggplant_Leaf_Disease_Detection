import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

try:
    import xgboost as xgb
    import catboost as cb
    import lightgbm as lgb
    BOOSTING_AVAILABLE = True
except ImportError as e:
    print(f"Cảnh báo: Thiếu thư viện boosting ({e}). Script vẫn sẽ chạy với các model cơ bản.")
    BOOSTING_AVAILABLE = False


def run_ml_experiment_holdout(file_path, output_csv='machine_learning_results_holdout.csv', scale_before_split=True):
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    print(f"Classes found: {class_names}")
    
    if scale_before_split:
        print("Đang áp dụng StandardScaler trên toàn bộ dữ liệu X trước khi chia...")
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
    else:
        print("Bỏ qua StandardScaler trước khi chia.")
        X_processed = X.values
        
    print("Tiến hành chia dữ liệu lần 1: 80% Train cơ bản, 20% Phần dư (Temporary)...")
    X_train_base, X_temp, y_train_base, y_temp = train_test_split(
        X_processed, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    print("Tiến hành chia dữ liệu lần 2: Tách 20% Phần dư thành 10% Validation và 10% Test độc lập...")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print("Đang gộp 80% Train và 10% Val lại thành tập huấn luyện tổng (90%)...")
    X_train_combined = np.vstack((X_train_base, X_val))
    y_train_combined = np.concatenate((y_train_base, y_val))
    
    print(f"Kích thước tập huấn luyện gộp (90%): X={X_train_combined.shape}, y={y_train_combined.shape}")
    print(f"Kích thước tập kiểm định (Test - 10%): X={X_test.shape}, y={y_test.shape}")
    
    models = {
        'KNN': KNeighborsClassifier(weights='distance'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Extra Trees': ExtraTreesClassifier(random_state=42, class_weight='balanced')
    }
    
    if BOOSTING_AVAILABLE:
        models.update({
            'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, device="cuda"),
            'CatBoost': cb.CatBoostClassifier(verbose=0, random_state=42, task_type="GPU", devices="0"),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
        })
        
    results = []
    
    cm_dir = 'confusion_matrices_holdout'
    if not os.path.exists(cm_dir):
        os.makedirs(cm_dir)

    print("\nBắt đầu huấn luyện và đánh giá trên tập Test 10%...")
    
    desired_order = ['KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'XGBoost', 'CatBoost', 'Extra Trees', 'LightGBM']
    
    for name in desired_order:
        if name not in models:
            continue
            
        model = models[name]
        print(f" -> Đang chạy: {name}")
        
        try:
            model.fit(X_train_combined, y_train_combined)
            
            y_pred = model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'{cm_dir}/cm_{name.replace(" ", "_")}.png')
            plt.close()
            
            acc = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            
            results.append({
                'Model': name,
                'Accuracy': round(acc, 2),
                'Precision': round(prec, 2),
                'Recall': round(rec, 2),
                'F1 Score': round(f1, 2)
            })
            
        except Exception as e:
            print(f"Lỗi khi chạy {name}: {e}")
            
    results_df = pd.DataFrame(results)
    print("\n=== KẾT QUẢ TỔNG HỢP (Hold-out TEST 10%) ===")
    print(results_df.to_string(index=False))
    
    results_df.to_csv(output_csv, index=False)
    print(f"\nĐã lưu file CSV kết quả tại: {output_csv}")
    print(f"Đã lưu các hình ảnh Confusion Matrix trong thư mục: '{cm_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Models with Hold-out (Train 90%, Test 10%)")
    parser.add_argument('--data_dir', type=str, 
                        default=r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\ml_algorithms\csv_deeplearning_features\resnet34_features.csv",
                        help="Đường dẫn đến file CSV chứa đặc trưng (features) của bạn")
    
    parser.add_argument('--scale_before_split', action='store_true', default=True,
                        help="Áp dụng StandardScaler trước khi chia (Mặc định: True)")
    parser.add_argument('--no_scale_before_split', dest='scale_before_split', action='store_false',
                        help="Tắt áp dụng StandardScaler trước khi chia dữ liệu")
                        
    args = parser.parse_args()
    
    run_ml_experiment_holdout(file_path=args.data_dir, scale_before_split=args.scale_before_split)
