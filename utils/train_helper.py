import torch
from data.dataset import get_iter
import numpy as np
import random

from utils.model_helper import IPM_loss, wassIPM_loss
from utils import data_helper


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def rwt_regression_loss(w, y, y_pre):
    y_pre, w = y_pre.to('cpu'), w.to('cpu')

    return ((y_pre.squeeze() - y.squeeze()) ** 2 * w.squeeze()).mean()


def train(model, data, args):
    model.train()
    epochs = args.n_epochs
    optimizer = torch.optim.Adam(
        [
            {'params': model.rwt.parameters(), 'weight_decay': 0},
            {'params': model.hidden_features.parameters()},
            {'params': model.out.parameters(), 'weight_decay': 0},
        ],
        lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay,
        amsgrad=False
    )
    # data = data_helper.load_train(args)
    dataloader = get_iter(filename='train.pkl', batch_size=500)

    for epoch in range(epochs):
        total_loss = []
        mmds = []
        for (x, t, y) in dataloader:
            if args.scale:
                y = args.scaler.transform(y.reshape(-1, 1))
                y = torch.from_numpy(y)

            optimizer.zero_grad()
            y_pre, w, _ = model(x, t)
            # print("y_pre shape", y_pre.shape, "w shape", w.shape)
            loss = rwt_regression_loss(w, y, y_pre)
            # print("loSS", loss)
            total_loss.append(loss.data)

            mmd = wassIPM_loss(x, t, w, k=5)
            mmds.append(mmd.data)
            loss = loss + mmd

            loss.backward()
            optimizer.step()

        total_loss = np.mean(total_loss)

        yield epoch + 1, model, total_loss


# for epoch in range(epochs):：开始训练循环，迭代指定的训练轮数（epochs）。
#
# total_loss = []：初始化一个空列表，用于存储每个批次的损失值。
#
# mmds = []：初始化一个空列表，用于存储每个批次的最大均值差（Maximum Mean Discrepancy，MMD）值。
#
# for (x, t, y) in dataloader:：迭代数据加载器，每次迭代从数据加载器中获取一个批次的数据。
#
# if args.scale:：如果设置了 args.scale（进行数据缩放）。
#
# y = args.scaler.transform(y.reshape(-1, 1))：将目标变量 y 应用缩放器进行数据缩放。
#
# y = torch.from_numpy(y)：将 NumPy 数组转换为 PyTorch 张量。
#
# x, t = x.to(args.device), t.to(args.device)：将特征变量 x 和时间变量 t 移动到指定的设备（通常是 GPU）上。
#
# optimizer.zero_grad()：将优化器的梯度归零，准备进行反向传播。
#
# y_pre, w, _ = model(x, t)：使用模型对特征变量 x 和时间变量 t 进行前向传播，得到预测值 y_pre 和其他输出。
#
# loss = rwt_regression_loss(w, y, y_pre)：计算 RWT 回归损失，衡量预测值与实际值的差异。
#
# total_loss.append(loss.data)：将当前批次的损失值添加到总损失列表中。
#
# mmd = IPM_loss(x, t, w, k=5)：计算最大均值差（MMD）损失，用于衡量特征分布与时间分布之间的差异。
#
# mmds.append(mmd.data)：将当前批次的 MMD 值添加到 MMD 列表中。
#
# loss = loss + mmd：将 RWT 回归损失和 MMD 损失相加，得到最终的总损失。
#
# loss.backward()：执行反向传播，计算梯度。
#
# optimizer.step()：根据计算的梯度更新模型的参数。
#
# total_loss = np.mean(total_loss)：计算所有批次的平均损失。
#
# yield epoch + 1, model, total_loss：使用生成器（generator）返回当前轮次的信息，包括轮次编号、模型和平均损失。生成器允许您逐步获取训练结果，而不是一次性计算完毕。




