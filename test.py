# import numpy as np
# import torch
# import torch.nn.functional as F
# from scipy.stats import wasserstein_distance
#
#
# def pdist2sq(A, B):
#     # return pairwise euclidean difference matrix
#     D = torch.sum((torch.unsqueeze(A, 1) - torch.unsqueeze(B, 0)) ** 2, 2)
#     return D
#
#
# def rbf_kernel(A, B, rbf_sigma=1):
#     rbf_sigma = torch.tensor(rbf_sigma)
#     return torch.exp(-pdist2sq(A, B) / torch.square(rbf_sigma) * .5)
#
#
# def calculate_mmd(A, B, rbf_sigma=1):
#     Kaa = rbf_kernel(A, A, rbf_sigma)
#     Kab = rbf_kernel(A, B, rbf_sigma)
#     Kbb = rbf_kernel(B, B, rbf_sigma)
#     mmd = Kaa.mean() - 2 * Kab.mean() + Kbb.mean()
#     return mmd
#
#
# def calculate_wassIPM(A, B):
#     A_np = A.detach().numpy()
#     A = A_np.reshape(-1)
#     B_np = B.detach().numpy()
#     B = B_np.reshape(-1)
#     wass_distance = wasserstein_distance(A, B)
#     return wass_distance
#
#
# def wassIPM_loss(x, t, w, k=5):
#     _, idx = torch.sort(t)
#     xw = x * w
#     sorted_x = x[idx]
#     sorted_xw = xw[idx]
#     split_x = torch.tensor_split(sorted_x, k)
#     split_xw = torch.tensor_split(sorted_xw, k)
#     loss = torch.zeros(k)
#
#     for i in range(k):
#         A = split_xw[i]
#         tmp_loss = torch.zeros(k - 1)
#         idx = 0
#
#         for j in range(k):
#             if i == j:
#                 continue
#             B = split_x[j]
#             # A = np.array(A).flatten()
#             # B = np.array(B).flatten()
#             partial_loss = calculate_wassIPM(A, B)
#             tmp_loss[idx] = partial_loss
#             idx += 1
#
#         loss[i] = tmp_loss.max()
#
#     return loss.mean()
#
#
# def IPM_loss_mmd(x, t, w, k=5, rbf_sigma=1):
#     _, idx = torch.sort(t)
#     xw = x * w
#     sorted_x = x[idx]
#     sorted_xw = xw[idx]
#     split_x = torch.tensor_split(sorted_x, k)
#     split_xw = torch.tensor_split(sorted_xw, k)
#     loss = torch.zeros(k)
#     for i in range(k):
#         A = split_xw[i]
#         tmp_loss = torch.zeros(k - 1)
#         idx = 0
#         for j in range(k):
#             if i == j:
#                 continue
#             B = split_x[j]
#             partial_loss = calculate_mmd(A, B, rbf_sigma)
#             tmp_loss[idx] = partial_loss
#             idx += 1
#         loss[i] = tmp_loss.max()
#
#     return loss.mean()
#
#
# # Example usage
# x = torch.rand(10000, 50)
# t = torch.rand(10000, 1)
# w = torch.rand(10000, 1)
# loss1 = IPM_loss_mmd(x, t, w)
# loss2 = wassIPM_loss(x, t, w)
# print("mmd IPM Loss:", loss1)
# print("wassIPM Loss:", loss2)
