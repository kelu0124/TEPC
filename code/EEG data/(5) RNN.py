import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import random
import os

# 设置随机数种子，确保每次运行结果一致
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# 加载数据
all_data = np.load('all.npy')  # 形状为(300, 300, 3)
labels = pd.read_excel('eeg_label.xlsx')['label'].values  # 标签数据，形状为(300,)

# 将标签转换为 one-hot 编码（适用于分类任务）
labels_one_hot = to_categorical(labels, num_classes=3)

# 数据预处理：标准化和归一化
def preprocess_data(all_data):
    # 1. 标准化数据
    scaler = StandardScaler()
    all_data_reshaped = all_data.reshape(-1, 3)  # 将数据展平为 (300*300, 3)
    all_data_standardized = scaler.fit_transform(all_data_reshaped)
    all_data_standardized = all_data_standardized.reshape(300, 300, 3)  # 恢复为原来的形状

    # 2. 归一化数据
    min_max_scaler = MinMaxScaler()
    all_data_reshaped = all_data.reshape(-1, 3)
    all_data_normalized = min_max_scaler.fit_transform(all_data_reshaped)
    all_data_normalized = all_data_normalized.reshape(300, 300, 3)

    return all_data_standardized, all_data_normalized

# 对数据进行标准化和归一化
all_data_standardized, all_data_normalized = preprocess_data(all_data)

# 选择一种预处理后的数据：这里我们使用标准化后的数据
data_to_use = all_data_standardized

# 准备十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 用于保存每一折的评估指标
accuracies = []
ppvs = []
sensitivities = []
specificities = []

# RNN 模型构建函数
def create_model():
    model = Sequential()

    # 使用堆叠 RNN 层
    model.add(SimpleRNN(128, activation='tanh', input_shape=(300, 3), return_sequences=True))  # 第一层 RNN
    model.add(Dropout(0.3))  # Dropout 防止过拟合
    model.add(SimpleRNN(64, activation='tanh', return_sequences=False))  # 第二层 RNN
    model.add(Dropout(0.3))  # Dropout 防止过拟合

    # 添加一个全连接层
    model.add(Dense(64, activation='relu'))  # 隐藏层
    model.add(Dropout(0.3))  # 再次 Dropout 避免过拟合

    # 输出层
    model.add(Dense(3, activation='softmax'))  # 三分类任务

    # 使用 Adam 优化器，并设置较小的学习率
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# 学习率调度函数：根据训练进度逐步衰减学习率
def lr_scheduler(epoch, lr):
    if epoch > 20:
        return lr * 0.8  # 每 20 个 epoch 衰减学习率
    return lr

# 进行十折交叉验证
for train_index, test_index in kf.split(data_to_use):
    X_train, X_test = data_to_use[train_index], data_to_use[test_index]
    y_train, y_test = labels_one_hot[train_index], labels_one_hot[test_index]
    
    # 创建模型
    model = create_model()
    
    # 早停机制
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # 学习率调度
    lr_schedule = LearningRateScheduler(lr_scheduler)
    
    # 训练模型
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stop, lr_schedule])
    
    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # 预测类别
    
    # 获取真实标签
    y_true = np.argmax(y_test, axis=1)  # 获取真实类别
    
    # 计算评估指标
    acc = accuracy_score(y_true, y_pred_classes)
    ppv = precision_score(y_true, y_pred_classes, average='macro')  # PPV (Precision)
    sensitivity = recall_score(y_true, y_pred_classes, average='macro')  # Sensitivity (Recall)
    cm = confusion_matrix(y_true, y_pred_classes)
    specificity = cm[1,1] / (cm[1,1] + cm[0,1]) if cm[1,1] + cm[0,1] > 0 else 0  # Specificity
    
    # 保存每一折的评估指标
    accuracies.append(acc)
    ppvs.append(ppv)
    sensitivities.append(sensitivity)
    specificities.append(specificity)

# 计算十折交叉验证的平均值
avg_accuracy = np.mean(accuracies)
avg_ppv = np.mean(ppvs)
avg_sensitivity = np.mean(sensitivities)
avg_specificity = np.mean(specificities)

# 输出评估指标
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average PPV: {avg_ppv:.4f}")
print(f"Average Sensitivity: {avg_sensitivity:.4f}")
print(f"Average Specificity: {avg_specificity:.4f}")
