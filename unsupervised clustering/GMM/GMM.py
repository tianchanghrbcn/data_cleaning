from sklearn.mixture import GaussianMixture
import numpy as np

# 创建一个示例数据集
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# 创建 GMM 模型并拟合数据
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)

# 打印拟合后的参数
print("Means (簇的均值):", gmm.means_)
print("Covariances (协方差矩阵):", gmm.covariances_)

# 预测每个样本的簇标签
labels = gmm.predict(X)
print("Labels (样本的簇标签):", labels)

# 计算每个样本属于每个簇的概率
probs = gmm.predict_proba(X)
print("Probabilities (样本属于每个簇的概率):", probs)
