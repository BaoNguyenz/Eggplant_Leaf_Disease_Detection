import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import seaborn as sns
import argparse
import os

# Cố gắng import các thư viện boosting
try:
    import xgboost as xgb
    import catboost as cb
    import lightgbm as lgb
    BOOSTING_AVAILABLE = True
except ImportError as e:
    print(f"Cảnh báo: Thiếu thư viện boosting ({e}). Script vẫn sẽ chạy với các model cơ bản.")
    BOOSTING_AVAILABLE = False

def run_ml_experiment_full(file_path, output_csv='machine_learning_results.csv'):
    # 1. Load và Tiền xử lý dữ liệu
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    X = df.drop('class_name', axis=1)
    y = df['class_name']
    
    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    print(f"Classes found: {class_names}")
    
    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Định nghĩa các Model
    models = {
        'KNN': KNeighborsClassifier(weights='distance'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Extra Trees': ExtraTreesClassifier(random_state=42, class_weight='balanced')
    }
    
    if BOOSTING_AVAILABLE:
        models.update({
            'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
            'CatBoost': cb.CatBoostClassifier(verbose=0, random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
        })
    
    # 3. Cấu hình Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }
    
    results = []
    
    # Tạo thư mục chứa ảnh nếu chưa có
    if not os.path.exists('confusion_matrices'):
        os.makedirs('confusion_matrices')

    print("\nBắt đầu huấn luyện và đánh giá...")
    
    # Thứ tự mong muốn trong CSV
    desired_order = ['KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'XGBoost', 'CatBoost', 'Extra Trees', 'LightGBM']
    
    # Lọc ra các model có sẵn
    available_models = [name for name in desired_order if name in models]
    
    # Sử dụng tqdm để theo dõi tiến trình
    for name in tqdm(available_models, desc="🔄 Training Models", unit="model", 
                     bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
        model = models[name]
        tqdm.write(f"  📊 Đang huấn luyện: {name}")
        
        try:
            # Tính Metrics
            cv_res = cross_validate(model, X_scaled, y_encoded, cv=cv, scoring=scoring)
            
            # Tính Confusion Matrix (dựa trên cross_val_predict để gộp kết quả 5 fold)
            y_pred = cross_val_predict(model, X_scaled, y_encoded, cv=cv)
            cm = confusion_matrix(y_encoded, y_pred)
            
            # Vẽ Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'confusion_matrices/cm_{name.replace(" ", "_")}.png')
            plt.close()
            
            # Lưu kết quả
            results.append({
                'Model': name,
                'Accuracy': round(np.mean(cv_res['test_accuracy']) * 100, 2),
                'Precision': round(np.mean(cv_res['test_precision']) * 100, 2),
                'Recall': round(np.mean(cv_res['test_recall']) * 100, 2),
                'F1 Score': round(np.mean(cv_res['test_f1']) * 100, 2)
            })
            
        except Exception as e:
            import traceback
            tqdm.write(f"  ❌ Lỗi khi chạy {name}: {e}")
            tqdm.write(traceback.format_exc())
            
    # 4. Xuất kết quả ra CSV
    results_df = pd.DataFrame(results)
    print("\n=== KẾT QUẢ TỔNG HỢP ===")
    print(results_df.to_string(index=False))
    
    results_df.to_csv(output_csv, index=False)
    print(f"\nĐã lưu file CSV tại: {output_csv}")
    print("Đã lưu các hình ảnh Confusion Matrix trong thư mục 'confusion_matrices'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='🌿 Huấn luyện và đánh giá các model Machine Learning trên dữ liệu features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python run_ml_models.py --data_dir vgg16_features.csv
  python run_ml_models.py --data_dir features.csv --output results.csv
        """
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True,
        help='Đường dẫn đến file CSV chứa features (bắt buộc)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='machine_learning_results.csv',
        help='Đường dẫn file CSV kết quả (mặc định: machine_learning_results.csv)'
    )
    
    args = parser.parse_args()
    
    # Chạy experiment
    run_ml_experiment_full(args.data_dir, args.output)