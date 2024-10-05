from sklearn.cluster import Birch
import numpy as np
import matplotlib.pyplot as plt

# 创建一个示例数据集
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0],
              [8, 2], [8, 4], [8, 0],
              [15, 2], [15, 4], [15, 0]])

# 创建 BIRCH 模型并拟合数据
birch_model = Birch(n_clusters=3)
labels = birch_model.fit_predict(X)

print("Labels (样本的簇标签):", labels)

# 绘制聚类结果
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.title('BIRCH Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
