import pandas as pd
import torch
import time

from data.dataset import get_iter
from utils.log_helper import save_obj, load_obj
from utils.model_helper import conditional_mutual_info


def eval(model, args):
    data = pd.read_pickle("D:\\MINET\\data\\test.pkl")
    data.fillna(0, inplace=True)
    columns_to_convert = ['nevents', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'ndays_act']
    for col in columns_to_convert:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    x = torch.tensor(data[['LoE_DI', 'age_DI', 'primary_reason', 'learner_type', 'expected_hours_week',
                           'discipline']].values, dtype=torch.float32)

    y = torch.tensor(data[['grade']].values, dtype=torch.float32)

    model.eval()

    n_test = 100
    t_grid_hat = torch.zeros(n_test)
    t_grid = torch.zeros(n_test)
    mse_id = torch.zeros(n_test)

    starttime = time.time()

    x = x.to(args.device)
    t = torch.tensor(data[['grade_reqs', 'ndays_act', 'nforum_posts']].values, dtype=torch.float32)  #
    t_transposed = t.transpose(0, 1)  # so we need transpose ->[3,500]
    cmi_values = []
    for i, ti in enumerate(t_transposed):
        # for ti in t: 将 t 视为一个矩阵，每次迭代将从 t 的每一行中取出一行作为 ti。
        # 因此，在每次迭代中，ti 是一个表示一个样本的一维张量，其长度为 3，表示你的t 张量的列数
        # ti->[1,500]/x->[500,6]
        ti = ti.unsqueeze(1)
        if i == 0:
            ti = ti * 3
        elif i == 1:
            ti = ti / 2
        elif i == 2:
            ti = ti / 3
        ti = ti * (1 - 0.456 * i)

        cmi = conditional_mutual_info(ti, y)  # 条件互信息的方式来确定t的变量的权重。
        cmi_values.append(cmi)
    # 假设 cmi_values 是包含三个 cmi 值的列表
    # 假设权重为 w，其中 w 是一个包含三个权重的向量

    w = torch.tensor(cmi_values)
    w = torch.softmax(w, dim=0)
    print(w)
    # weight1 = torch.tensor([0.333333])
    # weight2 = torch.tensor([0.333333])
    # weight3 = torch.tensor([0.333333])
    weighted_average = torch.sum(t * w, dim=1)
    weighted_average = weighted_average.unsqueeze(1)

    for i in range(n_test):
        t = (torch.ones(x.shape[0]) * weighted_average[i]).to(args.device)
        t = t.unsqueeze(1)

        out = model(x, t)
        out = out[0].data.squeeze().cpu()

        if args.scale:
            out = args.scaler.inverse_transform(out.reshape(-1, 1)).squeeze()
            out = torch.tensor(out)

        t_grid_hat[i] = out.mean()  # 计算预测的 t 格点均值
        ture_out = y  # 获取真实的输出值
        t_grid[i] = y.mean()  # 计算真实的 t 格点均值
        mse_id[i] = ((out - ture_out).squeeze() ** 2).mean()  # 计算均方误差

    estimation = t_grid_hat.cpu().numpy()
    savet = t.cpu().numpy()
    truth = t_grid.cpu().numpy()
    dir = 'D:\\MINET\\res\\'
    save_obj(estimation, dir, 'esti')
    save_obj(savet, dir, 't')
    save_obj(truth, dir, 'truth')

    mse = ((t_grid_hat.squeeze() - t_grid.squeeze()) ** 2).mean().data
    mse_id = mse_id.mean().data

    endtime = time.time()

    print('eval time cost {:.3f}'.format(endtime - starttime))

    return t_grid_hat, mse, mse_id

# 在主函数中，经过一定的训练轮次后，会调用 eval 函数来对模型进行评估。这个函数的主要作用包括：
#
# 模型准备：将模型设置为评估模式，即 model.eval()。
# 初始化变量：初始化一些用于记录测试结果的变量，如 t_grid_hat、t_grid 和 mse_id。
# 遍历测试数据：对测试数据进行遍历，对每个测试样本进行以下操作：
# a. 构造输入：根据当前测试样本的时间 test_data.t[i] 构造一个与输入数据 x 形状相同的时间 t。
# b. 模型预测：使用训练好的模型 model 对输入数据 x 和构造的时间 t 进行预测，得到输出 out。
# c. 后处理：如果设置了 args.scale，进行逆缩放操作。然后计算 t_grid_hat 和真实输出 ture_out 的均值。
# d. 计算 MSE：计算预测输出 out 和真实输出 ture_out 的均方误差，将结果记录在 mse_id 中。
# 结果保存：将评估结果以 numpy 数组的形式保存到指定路径下。
# 计算总体 MSE：计算预测输出 t_grid_hat 和真实输出 t_grid 之间的均方误差，作为评估模型性能的指标。
# 计算执行时间：记录评估过程的执行时间。
# 返回结果：将计算得到的预测输出、总体 MSE 和 mse_id 返回。
# 在主函数中，会调用 eval 函数对模型进行评估，并将得到的 MSE 打印出来以及记录在日志中,
# 这有助于了解训练过程中模型在测试数据上的表现，以及随着训练轮次的增加，模型性能是否有所提升。
