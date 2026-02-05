import pandas as pd
import os

# Đường dẫn folder chứa các file CSV
csv_folder = r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file"

# List các file CSV cần sửa
csv_files = [
    # "basic_features_Wilt_Disease.csv",
    # "gist_features_Wilt_Disease.csv",
    "glcm_features_wilt_disease_trans.csv",
    # "lbp_features_Wilt_Disease.csv"
]

# Sửa từng file
for csv_file in csv_files:
    file_path = os.path.join(csv_folder, csv_file)
    
    # Đọc file CSV
    df = pd.read_csv(file_path)

    df['Name'] = 'wilt_disease'
    
    # Lưu lại file CSV
    df.to_csv(file_path, index=False)
    
    print(f"✓ Đã sửa: {csv_file}")

print(f"\n✓ Hoàn thành! Tất cả nhãn đã được thay đổi thành {df['Name'][0]}")
