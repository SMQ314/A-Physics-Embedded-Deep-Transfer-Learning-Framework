import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import time

# ===============================================
# ========== 0. 全局开关 ==========
TRAIN_FLAG = True  # 第一次 True，之后 False
MODEL_PATH = 'cnn_multi_output_model.h5'  # 模型保存路径
X_SCALER_PATH = 'x_scaler_cnn_multi.pkl'
Y_SCALER_PATH = 'y_scaler_cnn_multi.pkl'
# ===============================================

# --------------------------------------------------
# 1. 读数据 - 修改为前9列特征，后4列目标
# --------------------------------------------------
df = pd.read_excel('D20-60 H40-120 s0.25-0.5.xlsx')

# 假设数据列结构：前9列是特征，后4列是目标
# 获取列名
all_columns = df.columns.tolist()

# 确定特征和目标列
if len(all_columns) >= 13:  # 至少13列（9特征+4目标）
    feature_columns = all_columns[:9]  # 前9列为特征
    target_columns = all_columns[9:13]  # 后4列为目标
else:
    # 或者根据实际情况指定列名
    # 这里假设后4列名为: 'Target1', 'Target2', 'Target3', 'Target4'
    feature_columns = all_columns[:9]
    target_columns = [col for col in all_columns if col.startswith('Target')]

    if len(target_columns) != 4:
        raise ValueError(f"请确保有4个目标列，当前找到 {len(target_columns)} 个目标列")

print(f"特征列: {feature_columns}")
print(f"目标列: {target_columns}")

X = df[feature_columns].values.astype('float32')
y = df[target_columns].values.astype('float32')

# --------------------------------------------------
# 2. 随机 10 % - 保持不变
# --------------------------------------------------
'''rng = np.random.RandomState(42)
n_keep = int(len(X) * 0.1)
idx = rng.choice(len(X), size=n_keep, replace=False)
X, y = X[idx], y[idx]
print(f'原始数据 {len(df)} 条 → 随机保留 10 % 共 {len(X)} 条')
'''
# --------------------------------------------------
# 3. 训练 / 加载 分支 - 修改为多输出
# --------------------------------------------------
if TRAIN_FLAG:
    print(f'【首次运行】开始训练 1-D CNN 多输出模型...')
    print(f'输入形状: {X.shape}, 输出形状: {y.shape}')

    # 3.1 标准化
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # 3.2 拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True)

    print(f'训练集: {X_train.shape}, 测试集: {X_test.shape}')
    print(f'目标维度: {y_train.shape[1]}')


    # 3.3 建 1-D CNN 多输出模型
    def build_cnn_multi_output(feat_dim, output_dim):
        model = models.Sequential([
            layers.Input(shape=(feat_dim,)),
            layers.Reshape((feat_dim, 1)),  # ← 关键一步

            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPool1D(2),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_dim)  # 多输出，输出维度为目标数
        ])
        return model


    output_dim = y_train.shape[1]  # 目标数量
    model = build_cnn_multi_output(X_train.shape[1], output_dim)

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='mse',  # 对于多输出回归，通常使用MSE
                  metrics=['mae'])

    print(model.summary())

    # 3.4 训练
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=200,
                        batch_size=64,
                        callbacks=[early_stop],
                        verbose=1)

    # 3.5 保存
    model.save(MODEL_PATH)
    joblib.dump(x_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)
    joblib.dump(history.history, 'train_history_cnn_multi.pkl')
    print('CNN 多输出模型与 scaler 已保存，下次把 TRAIN_FLAG 设为 False 即可直接加载。')

else:
    print('【加载模式】跳过训练，直接加载 CNN 多输出模型与 scaler...')
    model = models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='mse',
                  metrics=['mae'])

    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    X_scaled = x_scaler.transform(X)
    y_scaled = y_scaler.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True)

    output_dim = y_scaled.shape[1]

# --------------------------------------------------
# 4. 预测、评估、出图
# --------------------------------------------------
train_pred = y_scaler.inverse_transform(model.predict(X_train, verbose=0))
test_pred = y_scaler.inverse_transform(model.predict(X_test, verbose=0))
train_true = y_scaler.inverse_transform(y_train)
test_true = y_scaler.inverse_transform(y_test)


def calc_metrics(y_true, y_pred, suffix=''):
    """计算多个评估指标，针对多输出"""
    metrics_dict = {}

    # 整体指标（所有目标一起计算）
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f'{suffix:>6} 整体: R²={r2:.4f}  MSE={mse:.4e}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    # 每个目标的单独指标
    for i, target_name in enumerate(target_columns):
        target_mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        target_rmse = np.sqrt(target_mse)
        target_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        target_r2 = r2_score(y_true[:, i], y_pred[:, i])

        print(
            f'       {target_name}: R²={target_r2:.4f}  MSE={target_mse:.4e}  RMSE={target_rmse:.4f}  MAE={target_mae:.4f}')

        metrics_dict[f'{target_name}_R2'] = target_r2
        metrics_dict[f'{target_name}_MSE'] = target_mse
        metrics_dict[f'{target_name}_RMSE'] = target_rmse
        metrics_dict[f'{target_name}_MAE'] = target_mae

    # 整体指标
    metrics_dict['Overall_R2'] = r2
    metrics_dict['Overall_MSE'] = mse
    metrics_dict['Overall_RMSE'] = rmse
    metrics_dict['Overall_MAE'] = mae

    return metrics_dict


print('\n' + '=' * 50)
train_metrics = calc_metrics(train_true, train_pred, 'Train')
print('-' * 50)
test_metrics = calc_metrics(test_true, test_pred, 'Test')
print('=' * 50)

# loss 曲线
hist = history.history if TRAIN_FLAG else joblib.load('train_history_cnn_multi.pkl')
plt.figure(figsize=(6, 4))
plt.plot(hist['loss'], label='Train loss', color='#1f77b4', lw=2)
plt.plot(hist['val_loss'], label='Val loss', color='#ff7f0e', lw=2)
plt.xlabel('Epoch', fontdict={'family': 'Times New Roman', 'size': 22})
plt.ylabel('MSE Loss', fontdict={'family': 'Times New Roman', 'size': 22})
plt.tick_params(labelsize=18)
plt.legend(prop={'family': 'Times New Roman', 'size': 18})
plt.tight_layout()
plt.savefig('loss_curve_multi.png', dpi=600)
#plt.show()

# 为每个目标绘制散点图
for i, target_name in enumerate(target_columns):
    plt.figure(figsize=(6, 6))

    # 准备数据
    train_actual = train_true[:, i]
    train_pred_i = train_pred[:, i]
    test_actual = test_true[:, i]
    test_pred_i = test_pred[:, i]

    data2 = pd.DataFrame({
        'Actual': np.concatenate([train_actual, test_actual]),
        'Predicted': np.concatenate([train_pred_i, test_pred_i]),
        'Dataset': ['Train'] * len(train_actual) + ['Test'] * len(test_actual)
    })

    sns.scatterplot(data=data2, x='Actual', y='Predicted', hue='Dataset',
                    palette={'Train': '#1f77b4', 'Test': '#ff7f0e'},
                    s=25, alpha=0.6)

    min_val = data2[['Actual', 'Predicted']].min().min()
    max_val = data2[['Actual', 'Predicted']].max().max()
    plt.plot([min_val, max_val], [min_val, max_val],
             color='yellow', linestyle='--', linewidth=2, label='Perfect Fit')

    plt.xlabel(f'Actual {target_name}', fontdict={'family': 'Times New Roman', 'size': 22})
    plt.ylabel(f'Predicted {target_name}', fontdict={'family': 'Times New Roman', 'size': 22})
    plt.tick_params(labelsize=22)
    plt.legend(prop={'family': 'Times New Roman', 'size': 23})
    plt.tight_layout()
    plt.savefig(f'scatter_{target_name}.png', dpi=600)
    #plt.show()

# --------------------------------------------------
# 5. 置换重要度 (Permutation Importance) - 修改为多输出
# --------------------------------------------------
print('\n【计算置换重要性...】')


# 包装模型，使其兼容 sklearn 的 permutation_importance
class KerasModelMultiOutputWrapper:
    def __init__(self, keras_model, y_scaler):
        self.model = keras_model
        self.y_scaler = y_scaler

    def fit(self, X, y):
        # 伪fit方法，满足sklearn接口要求
        return self

    def predict(self, X):
        # 输入 X 是二维 (n_samples, n_features)，直接预测
        pred_scaled = self.model.predict(X, verbose=0)
        # 返回反标准化后的预测值
        return self.y_scaler.inverse_transform(pred_scaled)

    def score(self, X, y):
        # 使用负MSE作为评分，sklearn默认支持多输出
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        return -mse  # 返回负MSE，因为permutation_importance期望更高的分数更好


# 创建包装器实例
wrapped_model = KerasModelMultiOutputWrapper(model, y_scaler)

# 使用反标准化后的真实值
y_test_true = y_scaler.inverse_transform(y_test)

start_time = time.time()
print("开始计算置换重要性（这可能需要一些时间）...")

# 对每个目标分别计算重要性，然后取平均
all_importances = []
all_importances_std = []

for target_idx in range(output_dim):
    print(f"计算目标 {target_columns[target_idx]} 的特征重要性...")


    # 创建单目标包装器
    class SingleTargetWrapper:
        def __init__(self, keras_model, y_scaler, target_idx):
            self.model = keras_model
            self.y_scaler = y_scaler
            self.target_idx = target_idx

        def fit(self, X, y):
            return self

        def predict(self, X):
            pred_scaled = self.model.predict(X, verbose=0)
            pred = self.y_scaler.inverse_transform(pred_scaled)
            return pred[:, self.target_idx]


    single_wrapper = SingleTargetWrapper(model, y_scaler, target_idx)

    result = permutation_importance(
        single_wrapper, X_test, y_test_true[:, target_idx],
        n_repeats=5,  # 减少重复次数以加快速度
        random_state=42,
        scoring='neg_mean_squared_error',
        n_jobs=1
    )

    all_importances.append(result.importances_mean)
    all_importances_std.append(result.importances_std)

# 计算平均重要性
importance = np.mean(all_importances, axis=0)
importance_std = np.mean(all_importances_std, axis=0)

elapsed = time.time() - start_time
print(f'置换重要性计算完成，耗时: {elapsed:.2f} 秒')

# 排序索引（按重要性从高到低）
sorted_idx = np.argsort(importance)[::-1]

# ========== 绘制置换重要性 ==========
plt.figure(figsize=(8, max(4, len(feature_columns) * 0.3)), dpi=300)

# 创建颜色渐变
norm = plt.Normalize(importance.min(), importance.max())
colors = plt.cm.viridis(norm(importance))

# 绘制水平条形图
plt.barh(range(len(feature_columns)), importance[sorted_idx],
         xerr=importance_std[sorted_idx],
         color=colors[sorted_idx],
         edgecolor='black', linewidth=0.5)

plt.yticks(range(len(feature_columns)), np.array(feature_columns)[sorted_idx])
plt.xlabel('平均置换重要性 (MSE Increase)', fontsize=14, labelpad=10)
plt.title(f'特征重要性 (多目标预测)', fontsize=16, pad=15, fontweight='semibold')
plt.grid(axis='x', linestyle='--', alpha=0.5)

# 添加数值标签
for i, (imp, std) in enumerate(zip(importance[sorted_idx], importance_std[sorted_idx])):
    plt.text(imp + std + 0.001, i, f'{imp:.4f}',
             va='center', fontsize=9, color='black')

plt.tight_layout()
plt.savefig('CNN_permutation_importance_multi.png', bbox_inches='tight', dpi=300)
#plt.show()

# 打印重要性排名
print('\n【特征重要性排名】')
for i, idx in enumerate(sorted_idx):
    print(f'{i + 1:2d}. {feature_columns[idx]:20s} : {importance[idx]:.6f} ± {importance_std[idx]:.6f}')

# --------------------------------------------------
# 6. 保存结果到 Excel
# --------------------------------------------------
# 创建训练集结果
train_results = []
for i in range(len(train_true)):
    row = {'sample_id': i}
    for j, target_name in enumerate(target_columns):
        row[f'{target_name}_true'] = train_true[i, j]
        row[f'{target_name}_pred'] = train_pred[i, j]
    train_results.append(row)

train_out = pd.DataFrame(train_results)

# 创建测试集结果
test_results = []
for i in range(len(test_true)):
    row = {'sample_id': i}
    for j, target_name in enumerate(target_columns):
        row[f'{target_name}_true'] = test_true[i, j]
        row[f'{target_name}_pred'] = test_pred[i, j]
    test_results.append(row)

test_out = pd.DataFrame(test_results)

# 创建指标汇总
metrics_data = []
for target in target_columns:
    metrics_data.append({
        'Target': target,
        'Train_R2': train_metrics.get(f'{target}_R2', np.nan),
        'Train_MSE': train_metrics.get(f'{target}_MSE', np.nan),
        'Train_RMSE': train_metrics.get(f'{target}_RMSE', np.nan),
        'Train_MAE': train_metrics.get(f'{target}_MAE', np.nan),
        'Test_R2': test_metrics.get(f'{target}_R2', np.nan),
        'Test_MSE': test_metrics.get(f'{target}_MSE', np.nan),
        'Test_RMSE': test_metrics.get(f'{target}_RMSE', np.nan),
        'Test_MAE': test_metrics.get(f'{target}_MAE', np.nan)
    })

# 添加整体指标
metrics_data.append({
    'Target': 'OVERALL',
    'Train_R2': train_metrics.get('Overall_R2', np.nan),
    'Train_MSE': train_metrics.get('Overall_MSE', np.nan),
    'Train_RMSE': train_metrics.get('Overall_RMSE', np.nan),
    'Train_MAE': train_metrics.get('Overall_MAE', np.nan),
    'Test_R2': test_metrics.get('Overall_R2', np.nan),
    'Test_MSE': test_metrics.get('Overall_MSE', np.nan),
    'Test_RMSE': test_metrics.get('Overall_RMSE', np.nan),
    'Test_MAE': test_metrics.get('Overall_MAE', np.nan)
})

metrics_df = pd.DataFrame(metrics_data)

# 创建重要性DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance,
    'Std': importance_std
}).sort_values('Importance', ascending=False)

# 保存到Excel
with pd.ExcelWriter('train_test_results_cnn_multi.xlsx', engine='openpyxl') as w:
    train_out.to_excel(w, sheet_name='Train', index=False)
    test_out.to_excel(w, sheet_name='Test', index=False)
    metrics_df.to_excel(w, sheet_name='Metrics', index=False)
    importance_df.to_excel(w, sheet_name='Feature_Importance', index=False)

print('\n结果已写入 train_test_results_cnn_multi.xlsx')
print(f'模型已保存为: {MODEL_PATH}')