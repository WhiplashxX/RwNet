import pandas as pd
import torch
import os
import numpy as np
from scipy.stats import wasserstein_distance


def pdist2sq(A, B):
    # return pairwise euclidean difference matrix
    D = torch.sum((torch.unsqueeze(A, 1) - torch.unsqueeze(B, 0)) ** 2, 2)
    return D


def rbf_kernel(A, B, rbf_sigma=1):
    rbf_sigma = torch.tensor(rbf_sigma)
    return torch.exp(-pdist2sq(A, B) / torch.square(rbf_sigma) * .5)


def calculate_mmd(A, B, rbf_sigma=1):
    Kaa = rbf_kernel(A, A, rbf_sigma)
    Kab = rbf_kernel(A, B, rbf_sigma)
    Kbb = rbf_kernel(B, B, rbf_sigma)
    mmd = Kaa.mean() - 2 * Kab.mean() + Kbb.mean()
    return mmd


def IPM_loss(x, t, w, k=5, rbf_sigma=1):
    _, idx = torch.sort(t)
    xw = x * w
    sorted_x = x[idx]
    sorted_xw = xw[idx]
    split_x = torch.tensor_split(sorted_x, k)
    split_xw = torch.tensor_split(sorted_xw, k)
    loss = torch.zeros(k)
    for i in range(k):
        A = split_xw[i]
        tmp_loss = torch.zeros(k - 1)
        idx = 0
        for j in range(k):
            if i == j:
                continue
            B = split_x[j]
            partial_loss = calculate_mmd(A, B, rbf_sigma)
            tmp_loss[idx] = partial_loss
            idx += 1
        loss[i] = tmp_loss.max()

    return loss.mean()


def calculate_wassIPM(A, B):
    A_np = A.detach().numpy()
    A = A_np.reshape(-1)
    B_np = B.detach().numpy()
    B = B_np.reshape(-1)
    wass_distance = wasserstein_distance(A, B)
    return wass_distance


def wassIPM_loss(x, t, w, k=5):
    _, idx = torch.sort(t)
    xw = x * w
    sorted_x = x[idx]
    sorted_xw = xw[idx]
    split_x = torch.tensor_split(sorted_x, k)
    split_xw = torch.tensor_split(sorted_xw, k)
    loss = torch.zeros(k)

    for i in range(k):
        A = split_xw[i]
        tmp_loss = torch.zeros(k - 1)
        idx = 0

        for j in range(k):
            if i == j:
                continue
            B = split_x[j]
            # A = np.array(A).flatten()
            # B = np.array(B).flatten()
            partial_loss = calculate_wassIPM(A, B)
            tmp_loss[idx] = partial_loss
            idx += 1

        loss[i] = tmp_loss.max()

    return loss.mean()
























# ----------------mit---------------------
def mutual_info(joint, marginal, prop, **kwargs):
    sample_wise_max = marginal.max(dim=0)[0]
    mi = joint.mean() - (
            torch.log(torch.exp(marginal - sample_wise_max.unsqueeze(0)).mean(dim=0)) + sample_wise_max) @ prop
    return torch.sqrt(2 * mi + 1) - 1


# --------------- cmi -----------------
def conditional_mutual_info(joint, marginal_x, **kwargs):
    # joint: 同时分布的概率
    # marginal_x: x的边缘分布概率，marginal_y: y的边缘分布概率. prop: 权重向量
    sample_wise_max_x = marginal_x.max(dim=0)[0]  # []

    # 上述计算与之前一样，只是针对 x 的边缘分布进行计算
    mi = joint.mean() - (torch.log(torch.exp(
        marginal_x - sample_wise_max_x.unsqueeze(0)).mean(dim=0)) + sample_wise_max_x)

    return torch.sqrt(2 * mi + 1) - 1
    # 返回计算结果

#
# # 假设 t_list 是包含三个 t 变量的列表
# # 假设 x 是包含 x 变量的张量，y 是包含 y 变量的张量
# data = pd.read_pickle("D:\\MINET\\data\\test.pkl")
# data.fillna(0, inplace=True)
# columns_to_convert = ['nevents', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'ndays_act']
# for col in columns_to_convert:
#     data[col] = pd.to_numeric(data[col], errors='coerce')
#
# t = torch.tensor(data[['grade_reqs', 'ndays_act', 'nforum_posts']].values, dtype=torch.float32)
# x = torch.tensor(data[['LoE_DI', 'age_DI', 'primary_reason', 'learner_type', 'expected_hours_week',
#                        'discipline']].values, dtype=torch.float32)
#
# y = torch.tensor(data[['grade']].values, dtype=torch.float32)
#
# t_transposed = t.transpose(0, 1)  # so we need transpose
# cmi_values = []
#
# for ti in t_transposed:
#     # for ti in t: 将 t 视为一个矩阵，每次迭代将从 t 的每一行中取出一行作为 ti。
#     # 因此，在每次迭代中，ti 是一个表示一个样本的一维张量，其长度为 3，表示你的t 张量的列数。
#     joint = torch.cat((ti, y), dim=1)
#     marginal_x = torch.cat((x, y), dim=1)
#     marginal_y = torch.cat((ti, x), dim=1)
#
#     cmi = conditional_mutual_info(joint, marginal_x, marginal_y)
#     cmi_values.append(cmi)
#
# # 假设 cmi_values 是包含三个 cmi 值的列表
# # 假设权重为 w，其中 w 是一个包含三个权重的向量
# w = torch.tensor(cmi_values)
# w = torch.softmax(w, dim=0)
