from sklearn.cluster import KMeans
import numpy as np

# 创建一个简单的数据集
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# 进行K-Means聚类
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# 打印聚类结果
print("Cluster Centers:", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)
