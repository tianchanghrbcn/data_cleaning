import time
import math
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed

# 提供你的 CSV 文件的绝对路径
csv_file_path = r"D:\algorithm paper\data_cleaning\Datasets\flights\clean.csv"

start_time = time.time()

# 定义加权评分的 alpha 和 beta
alpha = 0.7
beta = 0.3

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

# 设置簇的最大数量为行数的平方根
max_clusters = 2 * math.isqrt(X.shape[0])

# 定义并行化的评估函数
def evaluate_gmm(n_components, cov_type):
    gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=0)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    silhouette_avg = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)

    # 计算综合得分
    combined_score = alpha * np.exp(1 + silhouette_avg) + beta * np.exp(-db_score)
    print(f"簇数量: {n_components}, 协方差类型: {cov_type}, Silhouette Score: {silhouette_avg}, Davies-Bouldin Score: {db_score}, Combined Score: {combined_score}")

    return n_components, cov_type, combined_score, labels

# 并行化运算，遍历不同簇数量和协方差类型
results = Parallel(n_jobs=-1)(delayed(evaluate_gmm)(n, cov_type)
                              for n in range(2, max_clusters + 1)
                              for cov_type in ['full', 'tied', 'diag', 'spherical'])

# 找到最佳结果
best_result = max(results, key=lambda x: x[2])
best_n_components, best_cov_type, best_combined_score, best_labels = best_result

# 输出最佳结果
print(f"\n最佳簇数量: {best_n_components}")
print(f"最佳协方差类型: {best_cov_type}")
print(f"最大 Combined Score: {best_combined_score}")

# 使用 PCA 将数据降维到 3D
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 创建 3D 散点图
fig = plt.figure(figsize=(16, 10), dpi=120)  # 全屏显示
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=best_labels, cmap='Set1', alpha=0.7, s=50)
colorbar = plt.colorbar(sc, ax=ax, pad=0.1)
colorbar.set_label('Cluster Label')

ax.set_title(f'GMM Clustering with {best_n_components} Components ({best_cov_type} covariance) (3D)', fontsize=15)
ax.set_xlabel('PCA Component 1', fontsize=12)
ax.set_ylabel('PCA Component 2', fontsize=12)
ax.set_zlabel('PCA Component 3', fontsize=12)
plt.show()

# 打印最终评估指标
final_silhouette_avg = silhouette_score(X_scaled, best_labels)
final_db_score = davies_bouldin_score(X_scaled, best_labels)

print(f"最终 Silhouette Score: {final_silhouette_avg}, 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
print(f"最终 Davies-Bouldin Score: {final_db_score}, 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

end_time = time.time()
print(f"程序执行结束，总耗时: {end_time - start_time} 秒")
