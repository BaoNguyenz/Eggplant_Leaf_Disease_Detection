"""
Example: Sử Dụng Features Đã Trích Xuất
========================================
Demo script minh họa cách load và sử dụng features từ CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_features(csv_path: str) -> tuple:
    """
    Load features từ CSV file.
    
    Args:
        csv_path: Đường dẫn tới file CSV
    
    Returns:
        Tuple (features, labels, image_paths)
    """
    print(f"Loading features from: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Extract features (bỏ 2 cột đầu: image_path, class_name)
    features = df.iloc[:, 2:].values
    
    # Extract labels
    labels = df['class_name'].values
    
    # Extract image paths
    image_paths = df['image_path'].values
    
    print(f"✓ Loaded {len(df)} samples")
    print(f"  - Feature shape: {features.shape}")
    print(f"  - Unique classes: {np.unique(labels)}")
    
    return features, labels, image_paths


def train_svm_classifier(X_train, X_test, y_train, y_test, class_names):
    """
    Train SVM classifier với RBF kernel.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Labels (encoded)
        class_names: List tên classes
    
    Returns:
        Trained SVM model
    """
    print("\n" + "="*50)
    print("Training SVM Classifier...")
    print("="*50)
    
    # Train SVM
    svm_clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_clf.fit(X_train, y_train)
    
    # Evaluate
    train_score = svm_clf.score(X_train, y_train)
    test_score = svm_clf.score(X_test, y_test)
    
    print(f"✓ SVM Training completed!")
    print(f"  - Train Accuracy: {train_score:.4f}")
    print(f"  - Test Accuracy : {test_score:.4f}")
    
    # Classification report
    y_pred = svm_clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return svm_clf


def train_random_forest(X_train, X_test, y_train, y_test, class_names):
    """
    Train Random Forest classifier.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Labels (encoded)
        class_names: List tên classes
    
    Returns:
        Trained RF model
    """
    print("\n" + "="*50)
    print("Training Random Forest Classifier...")
    print("="*50)
    
    # Train RF
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    
    # Evaluate
    train_score = rf_clf.score(X_train, y_train)
    test_score = rf_clf.score(X_test, y_test)
    
    print(f"✓ Random Forest Training completed!")
    print(f"  - Train Accuracy: {train_score:.4f}")
    print(f"  - Test Accuracy : {test_score:.4f}")
    
    # Classification report
    y_pred = rf_clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return rf_clf


def plot_confusion_matrix(y_test, y_pred, class_names, save_path=None):
    """
    Vẽ confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        class_names: List tên classes
        save_path: Optional path để lưu hình
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    else:
        plt.show()


def main():
    """Main execution."""
    
    # 1. Load features
    csv_path = r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction\efficientnet_b0_features.csv"
    
    features, labels, image_paths = load_features(csv_path)
    
    # 2. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_
    
    print(f"\n✓ Label Encoding:")
    for idx, class_name in enumerate(class_names):
        print(f"  {idx} -> {class_name}")
    
    # 3. Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\n✓ Data Split:")
    print(f"  - Train samples: {len(X_train)}")
    print(f"  - Test samples : {len(X_test)}")
    
    # 4. Train SVM
    svm_model = train_svm_classifier(X_train, X_test, y_train, y_test, class_names)
    
    # 5. Train Random Forest
    rf_model = train_random_forest(X_train, X_test, y_train, y_test, class_names)
    
    # 6. Plot confusion matrix (SVM)
    y_pred_svm = svm_model.predict(X_test)
    output_dir = Path(csv_path).parent
    plot_confusion_matrix(
        y_test,
        y_pred_svm,
        class_names,
        save_path=output_dir / "confusion_matrix_svm.png"
    )
    
    print("\n" + "="*50)
    print("✓ TRAINING COMPLETED!")
    print("="*50)


if __name__ == "__main__":
    # Kiểm tra dependencies
    try:
        import sklearn
        import matplotlib
        import seaborn
        print("✓ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install scikit-learn matplotlib seaborn")
        exit(1)
    
    main()
