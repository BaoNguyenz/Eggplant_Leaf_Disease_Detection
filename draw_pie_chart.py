import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import os

# Đường dẫn thư mục lưu file
save_dir = r"e:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset"

# Dữ liệu từ ảnh của bạn
labels = ['Healthy', 'Mosaic Virus', 'Leaf Spot', 'Insect Pest', 'Wilt', 'White Mold']
counts = [1451, 1362, 602, 546, 65, 63]

# Tạo DataFrame cho Plotly
df = pd.DataFrame({'Class': labels, 'Count': counts})

# =====================================================================
# 1. VẼ BẰNG SEABORN & MATPLOTLIB (Lưu thành file Ảnh PNG)
# =====================================================================
# Đặt style cho chữ nhật, gọn gàng
sns.set_theme(style="whitegrid")

# Lấy bảng màu từ seaborn (sử dụng palette 'viridis' giống tông màu biểu đồ cột gốc)
colors = sns.color_palette('viridis', len(labels))

# Đặt font chữ thành Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(10, 8))
# Vẽ biểu đồ tròn tròn (Pie Chart), dùng white edgecolor để chia tách các múi rõ ràng
wedges, texts, autotexts = plt.pie(
    counts, 
    labels=labels, 
    colors=colors, 
    autopct='%1.1f%%', 
    startangle=140, 
    pctdistance=0.85,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 14, 'fontname': 'Times New Roman'}
)

# Làm cho % số hiển thị in đậm và màu trắng/đen cho dễ nhìn
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')
    autotext.set_fontname('Times New Roman')
    autotext.set_fontsize(13)

# Nếu muốn thành Donut Chart (Hình vòng tròn rỗng ở giữa cho đẹp), bỏ comment 3 dòng dưới
# centre_circle = plt.Circle((0,0),0.70,fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)

# plt.title('Eggplant Leaf Disease Dataset Distribution', fontname='Times New Roman', fontsize=18, fontweight='bold', pad=20)
plt.axis('equal') # Đảm bảo biểu đồ tròn xoe chứ không bị méo

# Lưu ảnh
seaborn_out = os.path.join(save_dir, 'data_distribution_seaborn.pdf')
plt.tight_layout()
plt.savefig(seaborn_out, format='pdf', dpi=300)
print(f"Đã lưu biểu đồ Seaborn/Matplotlib (PDF) tại: {seaborn_out}")

# =====================================================================
# 2. VẼ BẰNG PLOTLY (Lưu thành file HTML tương tác động)
# =====================================================================
# Sử dụng Plotly Express để dễ dàng vẽ biểu đồ tương tác
fig = px.pie(
    df, 
    values='Count', 
    names='Class', 
    title='<b>Eggplant Leaf Disease Dataset Distribution</b>',
    color_discrete_sequence=px.colors.sequential.Viridis,
    hole=0.4 # hole > 0 sẽ biến Pie Chart thành Donut Chart (nhìn hiện đại hơn)
)

# Cập nhật hiển thị label và percent ngay trên múi
fig.update_traces(
    textposition='inside', 
    textinfo='percent+label',
    insidetextorientation='radial',
    marker=dict(line=dict(color='#ffffff', width=2))
)

# Lưu thành file web
plotly_out = os.path.join(save_dir, 'data_distribution_plotly.html')
fig.write_html(plotly_out)
print(f"Đã lưu biểu đồ web Plotly tại: {plotly_out}")
