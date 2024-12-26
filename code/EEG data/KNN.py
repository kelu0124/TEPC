import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import sys

# 读取命令行参数
couple = sys.argv[1]  # 获取提交任务编号（couple）

# 定义文件路径
feature_file = f"/mnt/ufs18/home-192/jiangj33/chenlong666/desktop/my_desk1/eeg_classify/result/npy_combine/eeg_feature_combined_couple_{couple}.npy"
label_file = "/mnt/ufs18/home-192/jiangj33/chenlong666/desktop/my_desk1/eeg_classify/eeg_data/eeg_label.xlsx"
output_file = f"/mnt/ufs18/home-192/jiangj33/chenlong666/desktop/my_desk1/eeg_classify/result/machine_out/KNN_{couple}_tuned.txt"

# 加载特征和标签
features = np.load(feature_file)  # (300, 30)
labels = pd.read_excel(label_file)["label"].values  # (300,)

# 初始化十折交叉验证
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 定义KNN模型
model = KNeighborsClassifier()

# 定义超参数网格
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# 使用GridSearchCV进行超参数优化
grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)

# 进行网格搜索
grid_search.fit(features, labels)

# 输出最佳超参数
print("最佳超参数组合:", grid_search.best_params_)

# 使用最佳超参数训练最终模型
best_model = grid_search.best_estimator_

# 定义保存指标的列表
accuracy_list = []
ppv_list = []
sensitivity_list = []
specificity_list = []

# 十折交叉验证
for train_index, test_index in kf.split(features, labels):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # 训练KNN模型
    best_model.fit(X_train, y_train)
    
    # 预测
    y_pred = best_model.predict(X_test)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    tp = np.diag(cm)  # 真正例
    fp = cm.sum(axis=0) - tp  # 假正例
    fn = cm.sum(axis=1) - tp  # 假反例
    tn = cm.sum() - (fp + fn + tp)  # 真负例

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    ppv = np.nanmean(tp / (tp + fp))  # 平均PPV
    sensitivity = np.nanmean(tp / (tp + fn))  # 平均Sensitivity
    specificity = np.nanmean(tn / (tn + fp))  # 平均Specificity
    
    # 保存指标
    accuracy_list.append(accuracy)
    ppv_list.append(ppv)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)

# 计算平均指标
final_accuracy = np.mean(accuracy_list)
final_ppv = np.mean(ppv_list)
final_sensitivity = np.mean(sensitivity_list)
final_specificity = np.mean(specificity_list)

# 保存结果到文件
with open(output_file, "w") as f:
    f.write(f"Accuracy: {final_accuracy:.4f}\n")
    f.write(f"PPV: {final_ppv:.4f}\n")
    f.write(f"Sensitivity: {final_sensitivity:.4f}\n")
    f.write(f"Specificity: {final_specificity:.4f}\n")

print("结果已保存到:", output_file)
