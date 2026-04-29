import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')

# ===============================================
# 全局字体配置（与第一个代码统一）
# ===============================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

GLOBAL_FONT_SIZE = 16
TITLE_FONT_SIZE = 18
SUBTITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
ANNOT_FONT_SIZE = 14

plt.rcParams['font.size'] = GLOBAL_FONT_SIZE
plt.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_FONT_SIZE
plt.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE
plt.rcParams['figure.titlesize'] = TITLE_FONT_SIZE

# ===============================================
# ========== 0. 全局开关 ==========
TRAIN_FLAG = False  # 第一次 True，之后 False
MODEL_PATH = 'cnn_multi_output_model.h5'
X_SCALER_PATH = 'x_scaler_cnn_multi.pkl'
Y_SCALER_PATH = 'y_scaler_cnn_multi.pkl'
# ===============================================

# --------------------------------------------------
# 1. 读数据
# --------------------------------------------------
df = pd.read_excel('D20-60 H40-120 s0.25-0.5.xlsx')

all_columns = df.columns.tolist()
if len(all_columns) >= 13:
    feature_columns = all_columns[:9]
    target_columns = all_columns[9:13]
else:
    # 如果列数不足，手动指定（根据实际文件调整）
    feature_columns = ['L', 'D', 'H', 'S', 'C', 'lamda', 'p', 'a', 'X']
    target_columns = ['Ce', 'Ci', 'Ei', 'Qe']
    if not all(col in df.columns for col in feature_columns + target_columns):
        raise ValueError("列名与预期不符，请检查数据文件列名")

print(f"特征列: {feature_columns}")
print(f"目标列: {target_columns}")

X = df[feature_columns].values.astype('float32')
y = df[target_columns].values.astype('float32')

# --------------------------------------------------
# 2. 训练 / 加载分支
# --------------------------------------------------
if TRAIN_FLAG:
    print(f'【首次运行】开始训练 1-D CNN 多输出模型...')
    print(f'输入形状: {X.shape}, 输出形状: {y.shape}')

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True)

    print(f'训练集: {X_train.shape}, 测试集: {X_test.shape}')
    print(f'目标维度: {y_train.shape[1]}')


    def build_cnn_multi_output(feat_dim, output_dim):
        model = models.Sequential([
            layers.Input(shape=(feat_dim,)),
            layers.Reshape((feat_dim, 1)),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPool1D(2),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_dim)
        ])
        return model


    output_dim = y_train.shape[1]
    model = build_cnn_multi_output(X_train.shape[1], output_dim)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='mse',
                  metrics=['mae'])

    print(model.summary())

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=200,
                        batch_size=64,
                        callbacks=[early_stop],
                        verbose=1)

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
# 3. 预测与评估
# --------------------------------------------------
train_pred = y_scaler.inverse_transform(model.predict(X_train, verbose=0))
test_pred = y_scaler.inverse_transform(model.predict(X_test, verbose=0))
train_true = y_scaler.inverse_transform(y_train)
test_true = y_scaler.inverse_transform(y_test)


def calc_metrics(y_true, y_pred, suffix=''):
    print(f'{suffix:>6} 整体: R²={r2_score(y_true, y_pred):.4f}  MSE={mean_squared_error(y_true, y_pred):.4e}  '
          f'RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.4f}  MAE={mean_absolute_error(y_true, y_pred):.4f}')
    for i, target_name in enumerate(target_columns):
        target_r2 = r2_score(y_true[:, i], y_pred[:, i])
        target_mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        target_rmse = np.sqrt(target_mse)
        target_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        print(f'       {target_name}: R²={target_r2:.4f}  MSE={target_mse:.4e}  RMSE={target_rmse:.4f}  MAE={target_mae:.4f}')
    return {'Overall_R2': r2_score(y_true, y_pred), 'Overall_MSE': mean_squared_error(y_true, y_pred),
            'Overall_RMSE': np.sqrt(mean_squared_error(y_true, y_pred)), 'Overall_MAE': mean_absolute_error(y_true, y_pred)}


print('\n' + '=' * 50)
train_metrics = calc_metrics(train_true, train_pred, 'Train')
print('-' * 50)
test_metrics = calc_metrics(test_true, test_pred, 'Test')
print('=' * 50)

# --------------------------------------------------
# 4. 物理一致性验证图（2×2布局）
# --------------------------------------------------
print('\n【生成物理一致性验证图（2×2布局）】')

# 计算测试集物理一致性
Ei_pred_test = test_pred[:, 2]
Qe_pred_test = test_pred[:, 3]
X_test_denorm = x_scaler.inverse_transform(X_test)
D_test = X_test_denorm[:, 1]
S_test = X_test_denorm[:, 3]
a_test = X_test_denorm[:, 7]
Qe_physics_from_pred = (Ei_pred_test / (np.square(D_test / (S_test + 1e-8)) + 1e-8)) * 1e18 / np.sin(np.radians(a_test))

# 计算训练集物理一致性
Ei_pred_train = train_pred[:, 2]
Qe_pred_train = train_pred[:, 3]
X_train_denorm = x_scaler.inverse_transform(X_train)
D_train = X_train_denorm[:, 1]
S_train = X_train_denorm[:, 3]
a_train = X_train_denorm[:, 7]
Qe_physics_from_pred_train = (Ei_pred_train / (np.square(D_train / (S_train + 1e-8)) + 1e-8)) * 1e18 / np.sin(np.radians(a_train))

abs_error_test = np.abs(Qe_pred_test - Qe_physics_from_pred)
abs_error_train = np.abs(Qe_pred_train - Qe_physics_from_pred_train)

# 创建2×2组图
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# 子图1: 测试集散点
ax = axes[0]
ax.scatter(Qe_physics_from_pred, Qe_pred_test, c='#2E86AB', alpha=0.6, s=50, edgecolors='none', label='Test samples')
min_val = min(Qe_physics_from_pred.min(), Qe_pred_test.min())
max_val = max(Qe_physics_from_pred.max(), Qe_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='Perfect Consistency')
ax.set_xlabel(r'$Q_e^{physics}$ (calculated from predicted $E_i$)', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax.set_ylabel(r'$Q_e^{predicted}$', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax.tick_params(labelsize=TICK_FONT_SIZE)
ax.set_title('Test Set: Physics Consistency Check', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
r2_physics_test = r2_score(Qe_physics_from_pred, Qe_pred_test)
mae_physics_test = mean_absolute_error(Qe_physics_from_pred, Qe_pred_test)
ax.text(0.05, 0.95, f'$R^2$ = {r2_physics_test:.4f}\nMAE = {mae_physics_test:.4e}',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'),
        fontsize=ANNOT_FONT_SIZE, verticalalignment='top', fontweight='bold')
ax.legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)
ax.grid(True, alpha=0.3)

# 子图2: 训练集散点
ax = axes[1]
ax.scatter(Qe_physics_from_pred_train, Qe_pred_train, c='#A23B72', alpha=0.5, s=40, edgecolors='none', label='Train samples')
min_val_train = min(Qe_physics_from_pred_train.min(), Qe_pred_train.min())
max_val_train = max(Qe_physics_from_pred_train.max(), Qe_pred_train.max())
ax.plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'r--', linewidth=3, label='Perfect Consistency')
ax.set_xlabel(r'$Q_e^{physics}$', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax.set_ylabel(r'$Q_e^{predicted}$', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax.tick_params(labelsize=TICK_FONT_SIZE)
ax.set_title('Train Set: Physics Consistency Check', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
r2_physics_train = r2_score(Qe_physics_from_pred_train, Qe_pred_train)
mae_physics_train = mean_absolute_error(Qe_physics_from_pred_train, Qe_pred_train)
ax.text(0.05, 0.95, f'$R^2$ = {r2_physics_train:.4f}\nMAE = {mae_physics_train:.4e}',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'),
        fontsize=ANNOT_FONT_SIZE, verticalalignment='top', fontweight='bold')
ax.legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)
ax.grid(True, alpha=0.3)

# 子图3: 绝对误差分布
ax = axes[2]
bins = np.linspace(0, max(np.percentile(abs_error_test, 95), np.percentile(abs_error_train, 95)), 50)
ax.hist(abs_error_train, bins=bins, color='#A23B72', alpha=0.6, label='Train', edgecolor='black', linewidth=0.5)
ax.hist(abs_error_test, bins=bins, color='#2E86AB', alpha=0.6, label='Test', edgecolor='black', linewidth=0.5)
ax.axvline(x=np.mean(abs_error_train), color='#A23B72', linestyle='--', linewidth=2.5,
           label=f'Train Mean: {np.mean(abs_error_train):.2e}')
ax.axvline(x=np.mean(abs_error_test), color='#2E86AB', linestyle='--', linewidth=2.5,
           label=f'Test Mean: {np.mean(abs_error_test):.2e}')
ax.axvline(x=np.median(abs_error_test), color='navy', linestyle='-.', linewidth=2,
           label=f'Test Median: {np.median(abs_error_test):.2e}')
ax.set_xlabel('Absolute Error of Physics Consistency', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax.tick_params(labelsize=TICK_FONT_SIZE)
ax.set_title('Distribution of Physics Consistency\nAbsolute Error', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# 子图4: 残差 vs 物理计算值
ax = axes[3]
residuals_physics = Qe_pred_test - Qe_physics_from_pred
ax.scatter(Qe_physics_from_pred, residuals_physics, c='#F18F01', alpha=0.6, s=50, edgecolors='none')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2.5)
ax.set_xlabel(r'$Q_e^{physics}$', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax.set_ylabel(r'$Q_e^{predicted} - Q_e^{physics}$', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax.tick_params(labelsize=TICK_FONT_SIZE)
ax.set_title('Physics Residuals vs Fitted Values', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Physics Consistency Validation: Verifying PINN Learned Physical Laws\n(Absolute Error Metrics)',
             fontsize=TITLE_FONT_SIZE, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('PECNN_physics_consistency_validation.png', dpi=600, bbox_inches='tight')
plt.close()
print('Saved: PECNN_physics_consistency_validation.png')
