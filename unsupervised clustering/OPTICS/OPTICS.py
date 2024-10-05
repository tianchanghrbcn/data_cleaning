from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt

# 创建一个示例数据集
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0],
              [8, 2], [8, 4], [8, 0],
              [15, 2], [15, 4], [15, 0]])

# 创建 OPTICS 模型并拟合数据
optics = OPTICS(min_samples=2, xi=0.05, min_cluster_size=0.1)
optics.fit(X)

# 获取簇标签
labels = optics.labels_
print("Labels (样本的簇标签):", labels)

# 获取可达距离
reachability = optics.reachability_
print("Reachability distances (可达距离):", reachability)

# 使用 reachability 和 ordering 筛选核心样本
core_sample_indices = np.where(reachability[optics.ordering_] < np.inf)[0]
print("Core sample indices (核心样本的索引):", core_sample_indices)

# 绘制可达距离图
plt.figure(figsize=(10, 7))
plt.bar(range(len(reachability)), reachability[optics.ordering_], color='r')
plt.plot(range(len(reachability)), reachability[optics.ordering_], 'b-', marker='.')
plt.title('Reachability Plot')
plt.xlabel('Sample index')
plt.ylabel('Reachability distance')
plt.show()

# 绘制聚类结果
plt.figure(figsize=(10, 7))
colors = ['r', 'g', 'b', 'y', 'c']

for klass, color in zip(range(0, 5), colors):
    Xk = X[labels == klass]
    plt.plot(Xk[:, 0], Xk[:, 1], color + '.', alpha=0.5)
plt.plot(X[labels == -1, 0], X[labels == -1, 1], 'k+', alpha=0.1)
plt.title('OPTICS Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
