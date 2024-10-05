from sklearn.cluster import AffinityPropagation
import numpy as np

# 创建一个示例数据集
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# 创建 Affinity Propagation 模型并拟合数据
ap = AffinityPropagation(random_state=0).fit(X)

# 打印聚类中心
print("Cluster Centers (聚类中心):", ap.cluster_centers_)

# 打印预测的簇标签
labels = ap.labels_
print("Labels (样本的簇标签):", labels)
