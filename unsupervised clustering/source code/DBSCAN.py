import time
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 提供你的 CSV 文件的绝对路径
csv_file_path = r"D:\algorithm paper\data_cleaning\Datasets\flights\clean.csv"

start_time = time.time()

# 使用 pandas 读取 CSV 文件
print("加载数据集...")
df = pd.read_csv(csv_file_path)

# 打印数据集的列名
print("数据集的列名:", df.columns)

# 排除列名中包含 'id' 的列
excluded_columns = [col for col in df.columns if 'id' in col.lower()]
print(f"排除包含 'id' 的列: {excluded_columns}")

# 选择不包含 'id' 的列作为目标列
remaining_columns = df.columns.difference(excluded_columns)

# 随机选择一列作为分类任务的目标列
target_column = np.random.choice(remaining_columns)
print(f"随机选择的目标列是: {target_column}")

# 将目标列与特征列分开
y = df[target_column]  # 目标列
X = df.drop(columns=[target_column])  # 特征列

# 编码目标列（如有必要）
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

# DBSCAN 参数范围
eps_values = np.arange(0.1, 1.05, 0.05)
min_samples_values = range(5, 26, 5)

# 定义 e 指数加权评分的 alpha 和 beta
alpha = 0.7
beta = 0.3

# 初始化最佳分数
best_combined_score = float('-inf')
best_eps = None
best_min_samples = None
best_labels = None

print(f"开始遍历 DBSCAN 参数，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

# 遍历 eps 和 min_samples 参数
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)

        if n_clusters > 1:
            silhouette_avg = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            combined_score = alpha * (1+silhouette_avg) + beta * (-db_score)

            print(f"簇数量: {n_clusters}, eps: {eps}, min_samples: {min_samples}, "
                  f"Silhouette Score: {silhouette_avg}, Davies-Bouldin Score: {db_score}, Weighted Score: {combined_score}")

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_eps = eps
                best_min_samples = min_samples
                best_labels = labels
        else:
            print(f"簇数量: {n_clusters}, eps: {eps}, min_samples: {min_samples}, 只有一个簇，跳过")

# 输出最佳参数结果
if best_eps and best_min_samples:
    print(f"\n最佳参数: eps={best_eps}, min_samples={best_min_samples}")
    print(f"最高的 Weighted Score: {best_combined_score}")

    if best_labels is not None and len(np.unique(best_labels)) > 1:
        final_silhouette_avg = silhouette_score(X_scaled, best_labels)
        final_db_score = davies_bouldin_score(X_scaled, best_labels)

        n_clusters_final = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)
        print(f"最终的簇数量: {n_clusters_final}")

        print(
            f"最终 Silhouette Score: {final_silhouette_avg}, 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(
            f"最终 Davies-Bouldin Score: {final_db_score}, 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        # 使用 PCA 将数据降维到 3D
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        # 3D 散点图可视化
        fig = plt.figure(figsize=(16, 10), dpi=120)  # 全屏显示
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=best_labels, cmap="Set1", alpha=0.7, s=50)

        # 添加颜色图例
        colorbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        colorbar.set_label('Cluster Label')

        ax.set_title(
            f'DBSCAN Clustering with {n_clusters_final} Clusters (eps={best_eps}, min_samples={best_min_samples})',
            fontsize=15)
        ax.set_xlabel('PCA Component 1', fontsize=12)
        ax.set_ylabel('PCA Component 2', fontsize=12)
        ax.set_zlabel('PCA Component 3', fontsize=12)
        plt.show()
else:
    print("没有找到合适的参数组合。")

end_time = time.time()
print(f"程序执行结束，总耗时: {end_time - start_time} 秒")
