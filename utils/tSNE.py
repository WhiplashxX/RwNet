import matplotlib.pyplot as plt
from shap.plots._labels import labels
from sklearn.manifold import TSNE
from data.dataset import get_iter
import pandas as pd
import torch



# 加载数据
dataloader = get_iter(filename='train.pkl', batch_size=500)
data = pd.read_pickle("D:\\MINET\\data\\test.pkl")
data.fillna(0, inplace=True)
data = torch.tensor(data[['LoE_DI', 'age_DI', 'primary_reason', 'learner_type', 'expected_hours_week',
                       'discipline']].values, dtype=torch.float32)

# 创建 t-SNE 模型
model = TSNE(n_components=2, random_state=0)

# 将高维数据降到二维
transformed_data = model.fit_transform(data)

# 将结果可视化
plt.figure(figsize=(12, 8))
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis', edgecolor='k')
plt.colorbar()
plt.show()
