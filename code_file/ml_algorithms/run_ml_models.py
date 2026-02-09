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
import argparse
import os
import sys

# Cố gắng import các thư viện boosting
try:
    import xgboost as xgb
    import catboost as cb
    import lightgbm as lgb
    BOOSTING_AVAILABLE = True
except ImportError as e:
    print(f"❌ Lỗi: Thiếu thư viện boosting ({e}).")
    print("Vui lòng cài đặt: pip install xgboost catboost lightgbm")
    BOOSTING_AVAILABLE = False


def validate_all_models(models, X_sample, y_sample):
    """
    Kiểm tra tất cả models có hoạt động không trước khi training.
    Trả về True nếu tất cả OK, False nếu có lỗi.
    """
    print("\n🔍 Kiểm tra tất cả models trước khi training...")
    failed_models = []
    
    for name, model in tqdm(models.items(), desc="✅ Validating Models", unit="model"):
        try:
            # Clone model để không ảnh hưởng đến model gốc
            from sklearn.base import clone
            test_model = clone(model)
            # Fit với một phần nhỏ dữ liệu
            test_model.fit(X_sample, y_sample)
            # Thử predict
            _ = test_model.predict(X_sample)
            tqdm.write(f"  ✅ {name}: OK")
        except Exception as e:
            tqdm.write(f"  ❌ {name}: FAILED - {e}")
            failed_models.append((name, str(e)))
    
    if failed_models:
        print(f"\n❌ Có {len(failed_models)} model bị lỗi:")
        for name, error in failed_models:
            print(f"   - {name}: {error}")
        return False
    
    print(f"\n✅ Tất cả {len(models)} models đều hoạt động tốt!")
    return True


def run_ml_experiment_full(file_path, output_csv='machine_learning_results.csv'):
    # 1. Load và Tiền xử lý dữ liệu
    print(f"📂 Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    X = df.drop('class_name', axis=1)
    y = df['class_name']
    
    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    print(f"📊 Classes found: {class_names}")
    print(f"📊 Data shape: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Định nghĩa các Model (BẮT BUỘC 8 models)
    if not BOOSTING_AVAILABLE:
        print("❌ Không thể tiếp tục vì thiếu thư viện boosting!")
        print("Vui lòng cài đặt: pip install xgboost catboost lightgbm")
        sys.exit(1)
    
    models = {
        'KNN': KNeighborsClassifier(weights='distance'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, verbosity=0),
        'CatBoost': cb.CatBoostClassifier(
            verbose=0, 
            random_state=42, 
            task_type='CPU',  # Đảm bảo dùng CPU để tương thích
            allow_writing_files=False  # Tắt log files
        ),
        'Extra Trees': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    # Thứ tự mong muốn trong CSV
    desired_order = ['KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'XGBoost', 'CatBoost', 'Extra Trees', 'LightGBM']
    
    # Kiểm tra đủ 8 models
    if len(models) != 8:
        print(f"❌ Lỗi: Chỉ có {len(models)}/8 models được định nghĩa!")
        sys.exit(1)
    
    # 3. VALIDATION: Kiểm tra tất cả models trước khi training
    # Sử dụng stratified sampling để đảm bảo mẫu test chứa tất cả các classes
    from sklearn.model_selection import train_test_split
    sample_size = min(200, len(X_scaled))
    # Lấy mẫu stratified
    X_sample, _, y_sample, _ = train_test_split(
        X_scaled, y_encoded, 
        train_size=sample_size, 
        stratify=y_encoded, 
        random_state=42
    )
    
    if not validate_all_models(models, X_sample, y_sample):
        print("\n❌ Không thể tiếp tục vì có model bị lỗi!")
        print("Vui lòng sửa lỗi và chạy lại.")
        sys.exit(1)
    
    # 4. Cấu hình Cross Validation
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

    print("\n🚀 Bắt đầu huấn luyện và đánh giá...")
    
    # Sử dụng tqdm để theo dõi tiến trình
    for name in tqdm(desired_order, desc="🔄 Training Models", unit="model", 
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
            tqdm.write(f"  ✅ {name}: Hoàn thành!")
            
        except Exception as e:
            import traceback
            tqdm.write(f"  ❌ Lỗi khi chạy {name}: {e}")
            tqdm.write(traceback.format_exc())
            # Thêm kết quả lỗi để đảm bảo có 8 dòng trong CSV
            results.append({
                'Model': name,
                'Accuracy': 'ERROR',
                'Precision': 'ERROR',
                'Recall': 'ERROR',
                'F1 Score': 'ERROR'
            })
            
    # 5. Xuất kết quả ra CSV
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("=== KẾT QUẢ TỔNG HỢP ===")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Kiểm tra có đủ 8 kết quả không
    success_count = len([r for r in results if r['Accuracy'] != 'ERROR'])
    print(f"\n📊 Thành công: {success_count}/8 models")
    
    results_df.to_csv(output_csv, index=False)
    print(f"\n💾 Đã lưu file CSV tại: {output_csv}")
    print("🖼️  Đã lưu các hình ảnh Confusion Matrix trong thư mục 'confusion_matrices'")

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