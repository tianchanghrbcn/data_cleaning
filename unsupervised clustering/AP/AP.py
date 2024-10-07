import time
import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# 对特征数据进行标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 扩大参数范围
damping_values = np.linspace(0.5, 0.9, 9)
preference_values = np.arange(-500, -100, 50)
best_combined_score = float('-inf')
best_labels = None
best_damping = None
best_preference = None


# 定义计算 e 指数加权综合得分的函数
def calculate_combined_score(silhouette_avg, db_score):
    alpha = 0.7
    beta = 0.3
    combined_score = alpha * np.exp(1 + silhouette_avg) + beta * np.exp(-db_score)
    return combined_score

# 遍历参数
print(f"开始遍历 AP 参数，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
for damping in damping_values:
    for preference in preference_values:
        ap = AffinityPropagation(damping=damping, preference=preference, random_state=0)
        labels = ap.fit_predict(X_scaled)
        n_clusters = len(np.unique(labels))

        if n_clusters <= 1 or n_clusters >= len(X_scaled):
            print(f"簇数量: {n_clusters}, damping: {damping}, preference: {preference}, 无效的簇数量，跳过")
            continue

        silhouette_avg = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        combined_score = calculate_combined_score(silhouette_avg, db_score)

        print(
            f"簇数量: {n_clusters}, damping: {damping}, preference: {preference}, Silhouette Score: {silhouette_avg}, Davies-Bouldin Score: {db_score}, Combined Score: {combined_score}")

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_damping = damping
            best_preference = preference
            best_labels = labels
            best_n_clusters = n_clusters

# 输出最佳参数组合
if best_labels is not None:
    print(f"\n最佳参数组合: damping={best_damping}, preference={best_preference}")
    print(f"最优 e 指数加权综合得分: {best_combined_score}")
    print(f"最终簇数量: {best_n_clusters}")

    # 计算并输出最终的评估指标
    final_silhouette_avg = silhouette_score(X_scaled, best_labels)
    final_db_score = davies_bouldin_score(X_scaled, best_labels)
    print(
        f"最终 Silhouette Score: {final_silhouette_avg}, 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(
        f"最终 Davies-Bouldin Score: {final_db_score}, 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # 数据可视化部分 - 3D 散点图
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=best_labels, cmap="Set1", alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Cluster Label')
    ax.set_title(f'AffinityPropagation Clustering with {best_n_clusters} Clusters (3D)')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.show()

else:
    print("没有找到合适的参数组合。")

end_time = time.time()
print(f"程序执行结束，总耗时: {end_time - start_time} 秒")
