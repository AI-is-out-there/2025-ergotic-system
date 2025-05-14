import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 启动 H2O 引擎
h2o.init()

# 加载数据
url = "https://raw.githubusercontent.com/AI-is-out-there/data2lab/main/module%203%20train.csv"
df = pd.read_csv(url, sep=';', nrows=5000)

# 选择字段并清洗
columns = ['Count_subj', 'rr_interval', 'p_end', 'qrs_onset', 'qrs_end',
           'p_axis', 'qrs_axis', 't_axis', 'Healthy_Status']
df = df[columns]
df.dropna(inplace=True)
df = df[df['rr_interval'] > 0]

# 分割训练与测试集
X = df.drop('Healthy_Status', axis=1)
y = df['Healthy_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 转换为 H2OFrame
train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test_h2o = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
train_h2o['Healthy_Status'] = train_h2o['Healthy_Status'].asfactor()
test_h2o['Healthy_Status'] = test_h2o['Healthy_Status'].asfactor()

# AutoML 训练
aml = H2OAutoML(max_runtime_secs=120, seed=42, nfolds=5)
aml.train(x=X.columns.tolist(), y='Healthy_Status', training_frame=train_h2o)

# 性能评估
perf = aml.leader.model_performance(test_h2o)
threshold = perf.F1().as_data_frame().iloc[0, 0]

# 使用自定义阈值生成预测结果
pred_proba = aml.leader.predict(h2o.H2OFrame(X_test)).as_data_frame()['p1']
y_pred = (pred_proba > threshold).astype(float)
y_true = y_test.astype(float)

# 标签映射与指定（更复杂逻辑）
labels_names = [0.1]        # 实际上表示“非健康者”，值为 0.1（根据你的精确复刻）
target_names = ['True', 'False']  # 自定义分类报告标签

# 分类报告输出
print(classification_report(y_true, y_pred, labels=labels_names, target_names=target_names))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred, labels=labels_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues, values_format='g')
plt.title("Confusion Matrix with labels_names = [0.1] and Threshold = {:.3f}".format(threshold))
plt.show()
