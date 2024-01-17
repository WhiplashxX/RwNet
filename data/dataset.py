import sys

import torch
from torch.utils.data import Dataset, DataLoader
# from data.simdata import *
import numpy as np
from utils.data_helper import *
from scipy.stats import norm
from scipy import interpolate
import pandas as pd


class basedata(DataLoader):
    def __init__(self, filename, n_feature=6) -> None:
        self.data = self.load_data(filename)  # 调用加载数据函数
        self.filename = filename
        self.data.x = self.load_data_x(filename)
        self.data.t = self.load_data_t(filename)
        self.data.y = self.load_data_y(filename)
        self.n_feature = n_feature

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, t, y = self.data.x[idx], self.data.t[idx], self.data.y[idx],
        return (x, t, y)

    def __len__(self):
        return len(self.data)  # 你的数据的长度

    def __iter__(self):
        self.current_idx = 0  # 初始化索引
        return self

    def __next__(self):
        if self.current_idx >= len(self):
            raise StopIteration
        else:
            sample = self[self.current_idx]
            self.current_idx += 1
            return sample

    def load_data(self, filename):
        # Load your preprocessed data from CSV
        base_dir = 'D:\\MINET\\data\\'
        if filename is None:
            path = 'D:\\MINET\\data\\processed_data.pkl'
        else:
            path = os.path.join(base_dir, filename)
        data = pd.read_pickle(path)
        data.fillna(0, inplace=True)
        columns_to_convert = ['nevents', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'ndays_act']
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # print(data.head())#debugging
        tensor_data = torch.tensor(data.values, dtype=torch.float32)
        return tensor_data

    def load_data_x(self, filename):
        # Load your preprocessed data from CSV
        base_dir = 'D:\\MINET\\data\\'
        if filename is None:
            path = 'D:\\MINET\\data\\processed_data.pkl'
        else:
            path = os.path.join(base_dir, filename)
        data = pd.read_pickle(path)
        data.fillna(0, inplace=True)
        columns_to_convert = ['nevents', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'ndays_act']
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        x = torch.tensor(data[['LoE_DI', 'age_DI', 'primary_reason', 'learner_type', 'expected_hours_week',
                               'discipline']].values, dtype=torch.float32)  # 'course_length'
        return x

    def load_data_t(self, filename):
        # Load your preprocessed data from CSV
        base_dir = 'D:\\MINET\\data\\'
        if filename is None:
            path = 'D:\\MINET\\data\\processed_data.pkl'
        else:
            path = os.path.join(base_dir, filename)
        data = pd.read_pickle(path)
        data.fillna(0, inplace=True)
        columns_to_convert = ['nevents', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'ndays_act']
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        t = torch.tensor(data[['grade_reqs', 'ndays_act', 'nforum_posts']].values, dtype=torch.float32)  #
        weight1 = torch.tensor([0.4185])
        weight2 = torch.tensor([0.4722])
        weight3 = torch.tensor([0.1093])
        weighted_average = torch.sum(t * torch.cat((weight1, weight2, weight3), dim=0), dim=1)
        # torch.Size([33195])

        weighted_average = weighted_average.unsqueeze(1)

        return weighted_average

    def load_data_y(self, filename):
        # Load your preprocessed data from CSV
        base_dir = 'D:\\MINET\\data\\'
        if filename is None:
            path = 'D:\\MINET\\data\\processed_data.pkl'
        else:
            path = os.path.join(base_dir, filename)
        data = pd.read_pickle(path)
        data.fillna(0, inplace=True)
        columns_to_convert = ['nevents', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'ndays_act']
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        y = torch.tensor(data[['grade']].values, dtype=torch.float32)  # , 'explored', 'nevents', 'completed_%'
        return y

    def get_outcome(self, filename):
        """ used in eval_helper """
        return self.load_data_y(filename)


def get_iter(filename, batch_size, shuffle=True, rw=False):
    dataset = basedata(filename=filename)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return iterator


# dataloader = get_iter(filename='train.pkl', batch_size=500)
# for batch in dataloader:
#     print("Batch:")
#     print(batch)
#     x_batch, t_batch, y_batch = batch
#     print("x_batch shape:", x_batch.shape)
#     print("t_batch shape:", t_batch.shape)
#     print("y_batch shape:", y_batch.shape)
# print("Number of batches:", len(dataloader))


# ---------------------------------------------------------
# datax = basedata(filename='train.pkl').load_data_t(filename='train.pkl')
# print(datax.shape)


# data = pd.read_pickle('processed_data.pkl')
# data.fillna(0, inplace=True)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# num_rows, num_columns = data.shape
# print("Number of rows:", num_rows)
# print("Number of columns:", num_columns)
# print(data.head(47422))
# print(data[-1000:])  # 显示最后1000行数据

# 创建一个文件并写入内容
# with open('data_info.txt', 'w') as f:
#     original_stdout = sys.stdout  # 保存原始的标准输出对象
#     sys.stdout = f  # 将标准输出重定向到文件
#
#     # 打印所有数据
#     print(data)
#
#     sys.stdout = original_stdout  # 恢复原始的标准输出对象
#
# print("Data information saved to data_info.txt")

#
# columns_to_convert = ['nevents', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'ndays_act']
# for col in columns_to_convert:
#     data[col] = pd.to_numeric(data[col], errors='coerce')
# print(data.dtypes)

# x = torch.tensor(data[['LoE_DI', 'age_DI', 'primary_reason', 'learner_type', 'expected_hours_week',
#                       'discipline', 'course_length']].values, dtype=torch.float32)
# t = torch.tensor(data[['grade_reqs', 'nforum_posts', 'ndays_act']].values, dtype=torch.float32)
# y = torch.tensor(data[['grade', 'explored', 'nevents', 'completed_%']].values, dtype=torch.float32)
#
# print("Loaded data:")
# print("x shape:", x.shape)
# print("t shape:", t.shape)
# print("y shape:", y.shape)

# LoE_DI'-1.0, 'age_DI'-1.0, 'primary_reason'3.0, 'learner_type'2.0, 'expected_hours_week2.0

# 打印数据+2 = 原始csv

# data_helper.py: 这个文件包含了一些数据处理的辅助函数，例如加载数据、保存数据、数据预处理等。
# 它定义了一些用于处理数据的函数，例如load_data、save_data、load_train、load_test等。
# 这些函数用于管理数据的读取和保存，以及对数据的预处理操作。data_helper.py 里的函数和逻辑有助于更好地组织和管理数据。
#
# dataset.py: 这是一个数据集定义的文件，可能在代码中没有给出，但在代码的其他部分可能会使用它。
# 在伪代码中，有一个 Dataset_from_simdata 类，它继承自 PyTorch 中的 Dataset 类。
# 这个类的作用是将原始的数据（在伪代码中的 data）转化为适合训练的数据集对象，以便在训练时可以方便地迭代和获取数据样本。
# 在实际代码中，这个文件可能会定义如何将原始数据转化为数据集对象，并实现 __len__ 和 __getitem__ 方法。
#
# 所以，data_helper.py 提供了对数据的处理和管理功能，而 dataset.py 则定义了数据集对象的构建方式，
# 两者共同协作以提供训练和测试数据的有效管理和使用。
