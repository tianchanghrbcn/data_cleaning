import time
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 提供你的 CSV 文件的绝对路径
csv_file_path = r"D:\algorithm paper\data_cleaning\Datasets\flights\clean.csv"

start_time = time.time()

# 定义加权评分的 alpha 和 beta
alpha = 0.7
beta = 0.3

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

# 计算余弦距离
X_cosine = cosine_distances(X_scaled)

# 参数范围调整
min_samples_range = [5, 10, 20, 30]
xi_range = [0.01, 0.05, 0.1]
min_cluster_size_range = [0.01, 0.02, 0.05]

# 定义最小簇数阈值和惩罚系数
min_clusters_threshold = 10
penalty_factor = 1.1

best_combined_score = float('-inf')
best_min_samples = None
best_xi = None
best_min_cluster_size = None
best_labels = None

print(f"开始遍历 OPTICS 参数，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

# 遍历 min_samples, xi 和 min_cluster_size 参数
for min_samples in min_samples_range:
    for xi in xi_range:
        for min_cluster_size in min_cluster_size_range:
            # 使用 OPTICS 进行聚类
            optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, metric='precomputed')
            optics.fit(X_cosine)
            labels = optics.labels_
            n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)

            if n_clusters > 1:
                silhouette_avg = silhouette_score(X_cosine, labels, metric='precomputed')
                db_score = davies_bouldin_score(X_scaled, labels)
                combined_score = alpha * np.exp(1 + silhouette_avg) + beta * np.exp(-db_score)

                # 小簇惩罚：当簇数量小于阈值时，加入惩罚
                if n_clusters < min_clusters_threshold:
                    combined_score -= penalty_factor * (min_clusters_threshold - n_clusters)

                print(
                    f"簇数量: {n_clusters}, min_samples: {min_samples}, xi: {xi}, min_cluster_size: {min_cluster_size}, "
                    f"Silhouette Score: {silhouette_avg}, Davies-Bouldin Score: {db_score}, Combined Score: {combined_score}")

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_min_samples = min_samples
                    best_xi = xi
                    best_min_cluster_size = min_cluster_size
                    best_labels = labels

# 输出最优参数结果
if best_min_samples and best_xi and best_min_cluster_size:
    print(f"\n最佳参数: min_samples={best_min_samples}, xi={best_xi}, min_cluster_size={best_min_cluster_size}")
    print(f"最大 Combined Score: {best_combined_score}")
    n_clusters_final = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)
    print(f"最终的簇数量: {n_clusters_final}")

    # 使用 PCA 将数据降维到 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # 3D 散点图可视化
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=best_labels, cmap='Set1', alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Cluster Label')
    ax.set_title(f'OPTICS Clustering with {n_clusters_final} Clusters')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.show()

end_time = time.time()
print(f"程序执行结束，总耗时: {end_time - start_time} 秒")
