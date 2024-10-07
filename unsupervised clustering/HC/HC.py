import time
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# 提供你的 CSV 文件的绝对路径
csv_file_path = r"D:\algorithm paper\data_cleaning\Datasets\flights\clean.csv"

start_time = time.time()

# 使用 pandas 读取 CSV 文件
print("加载数据集...")
df = pd.read_csv(csv_file_path)

# 排除列名中包含 'id' 的列
excluded_columns = [col for col in df.columns if 'id' in col.lower()]
print(f"排除包含 'id' 的列: {excluded_columns}")

# 选择不包含 'id' 的列作为目标列
remaining_columns = df.columns.difference(excluded_columns)
target_column = np.random.choice(remaining_columns)
print(f"随机选择的目标列是: {target_column}")

# 将目标列与特征列分开
y = df[target_column]
X = df.drop(columns=[target_column])

# 如果目标列是类别型数据，进行编码
if y.dtype == 'object' or y.dtype == 'category':
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"目标列 {target_column} 已进行编码处理")

# 对类别型数据进行频率编码
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
        X[col] = X[col].map(X[col].value_counts(normalize=True))

# 删除包含 NaN 的行
X = X.dropna()

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 设置簇的最大数量为行数的平方根
max_clusters = 2*math.isqrt(X.shape[0])

# 定义加权评分的 alpha 和 beta
alpha = 0.7
beta = 0.3

# 定义簇数量的惩罚因子，针对簇数量小于某阈值时应用惩罚
penalty_factor = 1.1
min_clusters_threshold = 10

# 定义加权评分函数
def weighted_score(silhouette, davies_bouldin, n_clusters):
    silhouette = 1 + silhouette
    score = alpha * np.exp(silhouette) + beta * np.exp(-davies_bouldin)
    if n_clusters < min_clusters_threshold:
        score += penalty_factor * (min_clusters_threshold - n_clusters)
    return score

# 遍历簇数量，从 3 到 max_clusters
best_weighted_score = float('-inf')
best_silhouette_score = float('-inf')
best_db_score = float('inf')
best_n_clusters = None
best_labels = None

print(f"开始遍历簇数量，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

for n_clusters in range(3, max_clusters + 1):
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='complete')
    labels = hc.fit_predict(X_scaled)

    silhouette_avg = silhouette_score(X_scaled, labels, metric='cosine')
    db_score = davies_bouldin_score(X_scaled, labels)
    current_weighted_score = weighted_score(silhouette_avg, db_score, n_clusters)

    print(f"簇数量: {n_clusters}, Silhouette Score: {silhouette_avg}, Davies-Bouldin Score: {db_score}, Weighted Score: {current_weighted_score}")

    if silhouette_avg > best_silhouette_score and db_score < best_db_score:
        best_weighted_score = current_weighted_score
        best_silhouette_score = silhouette_avg
        best_db_score = db_score
        best_n_clusters = n_clusters
        best_labels = labels

print(f"\n最佳簇数量: {best_n_clusters}")
print(f"最优的 Weighted Score: {best_weighted_score}")

# 计算并输出最终的评估指标
final_silhouette_avg = silhouette_score(X_scaled, best_labels, metric='cosine')
final_db_score = davies_bouldin_score(X_scaled, best_labels)

print(f"最终 Silhouette Score: {final_silhouette_avg}")
print(f"最终 Davies-Bouldin Score: {final_db_score}")

# 数据可视化部分
print("数据可视化...")

# 使用 PCA 将数据降维到 3D
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 创建 3D 散点图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=best_labels, cmap='Set1', alpha=0.7)
plt.colorbar(sc, ax=ax, label='Cluster Label')
ax.set_title(f'Agglomerative Clustering with {best_n_clusters} Clusters (3D)')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
plt.show()

end_time = time.time()
print(f"程序执行结束，总耗时: {end_time - start_time} 秒")
