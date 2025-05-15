# 在 Colab 中安装 H2O
!pip
install - f
httph2o - release.s3.amazonaws.comh2olatest_stable_Py.html
h2o

import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 初始化 H2O
h2o.init()

# 假设你已有 `df_selected` DataFrame
from google.colab import files
import pandas as pd

# 上传并读取数据
uploaded = files.upload()
df = pd.read_csv(модуль
3 - датасет - практика.csv)

# 选择特征列
cols = ['Count_subj', 'rr_interval', 'p_end', 'qrs_onset', 'qrs_end', 'p_axis', 'qrs_axis', 't_axis', 'Healthy_Status']
df = df[cols].apply(pd.to_numeric, errors='coerce').dropna()

# 转为 H2OFrame
hf = h2o.H2OFrame(df)
hf['Healthy_Status'] = hf['Healthy_Status'].asfactor()

# 划分训练和测试
train, test = hf.split_frame(ratios=[0.8], seed=42)
features = cols[-1]
target = 'Healthy_Status'

# 启动 AutoML
aml = H2OAutoML(max_runtime_secs=30, seed=1)
aml.train(x=features, y=target, training_frame=train)

# 预测
pred = aml.leader.predict(test).as_data_frame()['predict']
true = test[target].as_data_frame()[target]

# 转换为整数并评估
y_true = true.astype(int)
y_pred = pred.astype(int)
cm = confusion_matrix(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# 显示混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title(fH2O
AutoML
Confusion
Matrix(F1 - score = {f1
.2
f}))
plt.xlabel(Predicted)
plt.ylabel(True)
plt.show()


# 正确的版本组合
!pip install numpy==1.26.4 pandas==2.2.2 lightautoml -U --quiet

# 重启运行时（Colab 运行）
import os
os.kill(os.getpid(), 9)

# 安装 LightAutoML
!pip install -U lightautoml

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

# LightAutoML 导入
from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML

# 读取数据（你可上传文件后用 pd.read_csv 读取）
df = pd.read_csv("модуль 3 - датасет - практика.csv")

# 选取字段并清洗
cols = ['Count_subj', 'rr_interval', 'p_end', 'qrs_onset', 'qrs_end',
        'p_axis', 'qrs_axis', 't_axis', 'Healthy_Status']
df = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
df['Healthy_Status'] = df['Healthy_Status'].astype(int)

# 划分训练测试集
train, test = train_test_split(df, test_size=0.2, random_state=42)
train = train.rename(columns={'Healthy_Status': 'target'})
test = test.rename(columns={'Healthy_Status': 'target'})

# 定义任务和模型
task = Task('binary')
automl = TabularAutoML(task=task, timeout=60)

# 模型训练与预测
oof = automl.fit_predict(train, roles={'target': 'target'})
preds = automl.predict(test)

# 评估
y_pred = (preds.data[:, 0] > 0.5).astype(int)
y_true = test['target'].values
cm = confusion_matrix(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# 可视化
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["Unhealthy", "Healthy"], yticklabels=["Unhealthy", "Healthy"])
plt.title(f"LightAutoML Confusion Matrix (F1-score = {f1:.2f})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
