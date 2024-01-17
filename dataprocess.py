import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


# 读取数据
data = pd.read_csv('D:\\dataset1.csv', low_memory=False)


# (1) 缺失值处理
# data.fillna(0, inplace=True)
data['completed_%'].fillna(0, inplace=True)
data['primary_reason'].fillna('Missing', inplace=True)
data['learner_type'].fillna('Missing', inplace=True)
data['expected_hours_week'].fillna('Missing', inplace=True)
data['LoE_DI'].fillna('Missing', inplace=True)
data['nevents'].fillna(0, inplace=True)
data['nforum_posts'].fillna(0, inplace=True)
data['ndays_act'].fillna(0, inplace=True)


# (2) 去重
data.drop_duplicates(inplace=True)

# (3) 确定变量和转换
# 根据需求选择自变量、处理变量和因变量
selected_columns = [
    'grade', 'explored', 'nevents', 'completed_%',
    'grade_reqs', 'nforum_posts', 'course_length',
    'LoE_DI', 'age_DI', 'primary_reason', 'learner_type', 'expected_hours_week',
    'discipline', 'ndays_act'
]
discipline_mapping = {
    'Mathematics & Statistics': 1,
    'Professions and Applied Sciences': 2,
    'Medical Pre-Medical': 3,
    'Education': 4,
    'Interdisciplinary and Other': 5,
    'Social Sciences': 6,
    'Business and Management': 7,
    'Humanities': 8,
    'Computer Science': 9,
    'Physical Sciences': 10
}
primary_reason_mapping = {
    '': 0,
    'Missing': 0,
    'I enjoy being part of a community of learners': 1,
    'I am curious about MOOCs': 2,
    'I hope to gain skills for a new career': 3,
    'I am preparing to go back to school': 4,
    'I hope to gain skills for a promotion at work': 5,
    'I enjoy learning about topics that interest me': 6,
    'I hope to gain skills to use at work': 7,
    'I like the format (online)': 8,
    'I want to try Canvas Network': 9,
    'I am preparing for college for the first time': 10
}
learner_type_mapping = {
    '': 0,
    'Missing': 0,
    'Passive': 1,
    'Passive participant': 2,
    'Active participant': 3,
    'Active': 4,
    'Drop-in': 5,
    'Observer': 6
}
expected_hours_week_type = {
    '': 0,
    'Missing': 0,
    'Less than 1 hour': 1,
    'Between 1 and 2 hours': 2,
    'Between 2 and 4 hours': 3,
    'Between 4 and 6 hours': 4,
    'Between 6 and 8 hours': 5,
    'More than 8 hours per week': 6
}
LoE_DI_type = {
    '': 0,
    'Missing': 0,
    'Completed 2-year college degree': 1,
    'Completed 4-year college degree': 2,
    "Master's Degree (or equivalent)": 3,
    'None of these': 4,
    'Ph.D., J.D., or M.D. (or equivalent)': 5,
    'High School or College Preparatory School': 6,
    'Some college, but have not finished a degree': 7,
    'Some graduate school': 8
}
age_DI_type = {
    '': 0,
    '{}': 0,
    '{19-34}': 1,
    '{34-54}': 2,
    '{55 or older}': 3
}
# data = data[selected_columns]
# 分离因变量和特征
x = data[['LoE_DI', 'age_DI', 'primary_reason', 'learner_type', 'expected_hours_week', 'discipline', 'course_length']]
y = data[['grade', 'explored', 'nevents', 'completed_%']]
T = data[['grade_reqs', 'nforum_posts', 'ndays_act']]

# 离散值和连续值处理
discrete_columns = ['discipline', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'LoE_DI', 'age_DI', 'primary_reason', 'learner_type', 'expected_hours_week', 'ndays_act', 'nevents']  # 根据变量取值数量决定
continuous_columns = ['grade', 'completed_%']  # 根据变量取值数量决定

# One-Hot 编码离散变量
for column in discrete_columns:
    data[column] = data[column].astype(str)  # 将数值型特征转换为字符串类型
# encoder = LabelEncoder(sparse=False)
# discrete_data = encoder.fit_transform(data[discrete_columns])
# discrete_df = pd.DataFrame(discrete_data, columns=encoder.get_feature_names_out(discrete_columns))
data['discipline'] = data['discipline'].map(discipline_mapping)
data['primary_reason'] = data['primary_reason'].map(primary_reason_mapping)
data['learner_type'] = data['learner_type'].map(learner_type_mapping)
data['expected_hours_week'] = data['expected_hours_week'].map(expected_hours_week_type)
data['LoE_DI'] = data['LoE_DI'].map(LoE_DI_type)
data['age_DI'] = data['age_DI'].map(age_DI_type)
mapping_columns = ['discipline', 'primary_reason', 'learner_type', 'expected_hours_week', 'LoE_DI', 'age_DI', 'nevents', 'explored', 'grade_reqs',
                   'nforum_posts', 'course_length', 'ndays_act']
mapping_df = pd.DataFrame(data[mapping_columns], columns=mapping_columns)


# 归一化连续变量
# scaler = StandardScaler()
# continuous_data = scaler.fit_transform(data[continuous_columns])
# continuous_df = pd.DataFrame(continuous_data, columns=continuous_columns)
# Min-Max 归一化连续变量
# scaler = MinMaxScaler()
# continuous_data = scaler.fit_transform(data[continuous_columns.drop('grade')])
# continuous_df = pd.DataFrame(continuous_data, columns=continuous_columns.drop('grade'))
continuous_columns_without_grade = [col for col in continuous_columns if col != 'grade']
scaler = MinMaxScaler()
continuous_data = scaler.fit_transform(data[continuous_columns_without_grade])
continuous_df = pd.DataFrame(continuous_data, columns=continuous_columns_without_grade)

# 合并处理后的数据
processed_data = pd.concat([mapping_df, continuous_df, data['grade']], axis=1)

# (5) 筛选有效数据
processed_data_data = processed_data[processed_data['grade'] > 0]
# processed_data_data.reset_index(drop=True, inplace=True)  # 重置索引并且丢弃原来的索引列
# processed_data_data['index'] = processed_data_data.index + 1  # 添加新的列作为顺序标号
processed_data_data.to_pickle('processed_data.pkl')
# 划分数据集为训练集、验证集和测试集
train_data, temp_data = train_test_split(processed_data_data, test_size=0.3, random_state=42)
eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 保存划分后的数据集为CSV文件
train_data.to_csv('train_dataset.csv', index=False)
eval_data.to_csv('val_dataset.csv', index=False)
test_data.to_csv('test_dataset.csv', index=False)
# 保存划分后的数据集为pkl文件
train_data.to_pickle('train_dataset.pkl')
eval_data.to_pickle('val_dataset.pkl')
test_data.to_pickle('test_dataset.pkl')



#
# matplotlib.use('TkAgg')  # 设置绘图后端为TkAgg
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
# plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
# plt.hist(filtered_data['grade'], bins=20, color='orange')
# plt.title("成绩分布")
# plt.xlabel("成绩")
# plt.ylabel("频数")
#
# # 调整dpi参数以增加分辨率
# dpi = 1000  # 设置更高的dpi值
# plt.savefig("D:\\high_resolution_plot2.png", dpi=dpi, bbox_inches="tight")
#
# plt.show()
# print(filtered_data['grade'])
