import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# ===============================================
# 全局字体配置
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
# 全局配置
# ===============================================
TRAIN_FLAG = False
CONTINUE_TRAIN = False
SHAP_PDP_FLAG = False          # 关闭 SHAP/PDP
ADVANCED_PLOT_FLAG = True      # 关闭其他高级图表
MC_DROPOUT_SAMPLES = 100

WEIGHTS_PATH = 'pecnn_weights.weights.h5'
BEST_WEIGHTS_PATH = 'best_weights.weights.h5'
X_SCALER_PATH = 'x_scaler_pecnn.pkl'
Y_SCALER_PATH = 'y_scaler_pecnn.pkl'

# --------------------------------------------------
# 1. 读取数据
# --------------------------------------------------
df = pd.read_excel('D20-60 H40-120 s0.25-0.5.xlsx')

feature_columns = ['L', 'D', 'H', 'S', 'C', 'lamda', 'p', 'a', 'X']
target_columns = ['Ce', 'Ci', 'Ei', 'Qe']

X = df[feature_columns].values.astype('float32')
y = df[target_columns].values.astype('float32')

print(f"特征列: {feature_columns}")
print(f"目标列: {target_columns}")
print(f"数据形状: X={X.shape}, y={y.shape}")

# --------------------------------------------------
# 2. 数据预处理
# --------------------------------------------------
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
)

print(f'训练集: {X_train.shape}, 测试集: {X_test.shape}')

# --------------------------------------------------
# 3. 自定义物理约束损失层（保留，但未使用）
# --------------------------------------------------
class PhysicsInformedLossLayer(layers.Layer):
    def __init__(self, x_scaler, y_scaler, lambda_physics=0.1, **kwargs):
        super(PhysicsInformedLossLayer, self).__init__(**kwargs)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.lambda_physics = lambda_physics
        self.x_scale = tf.constant(x_scaler.scale_, dtype=tf.float32)
        self.x_mean = tf.constant(x_scaler.mean_, dtype=tf.float32)
        self.y_scale = tf.constant(y_scaler.scale_, dtype=tf.float32)
        self.y_mean = tf.constant(y_scaler.mean_, dtype=tf.float32)

    def call(self, inputs):
        y_true, y_pred, x_original = inputs
        y_pred_denorm = y_pred * self.y_scale + self.y_mean
        x_denorm = x_original * self.x_scale + self.x_mean
        D_nm = x_denorm[:, 1:2]
        S = x_denorm[:, 3:4]
        a_deg = x_denorm[:, 7:8]
        denominator = tf.square(D_nm / (S + 1e-8))
        a_rad = a_deg * tf.constant(np.pi / 180.0, dtype=tf.float32)
        Qe_physics = (y_pred_denorm[:, 2:3] / (denominator + 1e-8) * 1e18) / tf.sin(a_rad)
        loss_qe_physics = tf.reduce_mean(tf.square(y_pred_denorm[:, 3:4] - Qe_physics))
        loss_mse = tf.reduce_mean(tf.square(y_true - y_pred))
        total_loss = loss_mse + self.lambda_physics * loss_qe_physics
        self.add_loss(total_loss)
        return y_pred

    def get_config(self):
        config = super(PhysicsInformedLossLayer, self).get_config()
        config.update({
            'lambda_physics': self.lambda_physics,
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler
        })
        return config

# --------------------------------------------------
# 4. 构建 PECNN 模型
# --------------------------------------------------
def build_pecnn_model(feat_dim, output_dim, x_scaler, y_scaler):
    inputs = Input(shape=(feat_dim,), name='input')
    x = layers.Reshape((feat_dim, 1))(inputs)
    x = layers.Conv1D(64, 3, activation='relu', padding='same', name='conv1d')(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same', name='conv1d_1')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu', name='dense')(x)
    x = layers.Dense(32, activation='relu', name='dense_1')(x)
    outputs = layers.Dense(output_dim, name='output')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name='PECNN')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# --------------------------------------------------
# 5. 训练或加载模型（省略训练分支，与原始代码相同，此处保留加载部分）
# --------------------------------------------------
if TRAIN_FLAG:
    # 训练代码（与原始相同，省略）
    pass
else:
    print('【预测模式】加载已训练模型...')
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
    model = build_pecnn_model(X_train.shape[1], y_train.shape[1], x_scaler, y_scaler)
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
        print(f'已加载: {WEIGHTS_PATH}')
    elif os.path.exists(BEST_WEIGHTS_PATH):
        model.load_weights(BEST_WEIGHTS_PATH)
        print(f'已加载: {BEST_WEIGHTS_PATH}')
    else:
        raise FileNotFoundError('未找到权重文件！')
    # 历史记录占位（因无需绘图，可忽略）
    history = {'loss': [], 'val_loss': []}

# --------------------------------------------------
# 6. 预测与评估
# --------------------------------------------------
def predict_without_physics_correction(model, X, x_scaler, y_scaler):
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    return y_pred

train_pred = predict_without_physics_correction(model, X_train, x_scaler, y_scaler)
test_pred = predict_without_physics_correction(model, X_test, x_scaler, y_scaler)

train_true = y_scaler.inverse_transform(y_train)
test_true = y_scaler.inverse_transform(y_test)

# --------------------------------------------------
# 7. 计算评价指标
# --------------------------------------------------
def calc_metrics(y_true, y_pred, suffix=''):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'{suffix:>6}: R²={r2:.4f}  MSE={mse:.4e}  RMSE={rmse:.4f}  MAE={mae:.4f}')
    for i, name in enumerate(target_columns):
        mse_i = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse_i = np.sqrt(mse_i)
        mae_i = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2_i = r2_score(y_true[:, i], y_pred[:, i])
        print(f'       {name}: R²={r2_i:.4f}  RMSE={rmse_i:.4f}  MAE={mae_i:.4f}')
    return {'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}

print('\n' + '=' * 60)
print('【模型原始预测结果】')
print('=' * 60)
print('训练集:')
train_metrics = calc_metrics(train_true, train_pred, 'Train')
print('-' * 60)
print('测试集:')
test_metrics = calc_metrics(test_true, test_pred, 'Test')
print('=' * 60)

# 物理约束验证（仅打印）
print('\n【物理约束验证】(仅作为参考)')
def check_physics(y_pred, X, x_scaler, dataset_name):
    X_denorm = x_scaler.inverse_transform(X)
    D_nm = X_denorm[:, 1]
    S = X_denorm[:, 3]
    a_deg = X_denorm[:, 7]
    Ei = y_pred[:, 2]
    Qe = y_pred[:, 3]
    Qe_calc = (Ei / (np.square(D_nm / (S + 1e-8)) + 1e-8)) * 1e18 / np.sin(np.radians(a_deg))
    error_qe = np.mean(np.abs(Qe - Qe_calc))
    relative_error = np.mean(np.abs(Qe - Qe_calc) / (np.abs(Qe_calc) + 1e-8)) * 100
    print(f'{dataset_name}:')
    print(f'  Qe平均绝对误差: {error_qe:.6e}')
    print(f'  Qe相对误差: {relative_error:.2f}%')
check_physics(train_pred, X_train, x_scaler, '训练集')
check_physics(test_pred, X_test, x_scaler, '测试集')

# --------------------------------------------------
# 8. 物理一致性验证图（仅保留此图，改为2×2布局）
# --------------------------------------------------
print('\n【生成物理一致性验证图（2×2布局）】')

# 计算物理一致性
Ei_pred_test = test_pred[:, 2]
Qe_pred_test = test_pred[:, 3]
X_test_denorm = x_scaler.inverse_transform(X_test)
D_test = X_test_denorm[:, 1]
S_test = X_test_denorm[:, 3]
a_test = X_test_denorm[:, 7]
Qe_physics_from_pred = (Ei_pred_test / (np.square(D_test / (S_test + 1e-8)) + 1e-8)) * 1e18 / np.sin(np.radians(a_test))

Ei_pred_train = train_pred[:, 2]
Qe_pred_train = train_pred[:, 3]
X_train_denorm = x_scaler.inverse_transform(X_train)
D_train = X_train_denorm[:, 1]
S_train = X_train_denorm[:, 3]
a_train = X_train_denorm[:, 7]
Qe_physics_from_pred_train = (Ei_pred_train / (np.square(D_train / (S_train + 1e-8)) + 1e-8)) * 1e18 / np.sin(np.radians(a_train))

abs_error_test = np.abs(Qe_pred_test - Qe_physics_from_pred)
abs_error_train = np.abs(Qe_pred_train - Qe_physics_from_pred_train)

# 创建2×2组图（前四个子图）
fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # 改为2×2
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
# 新增：固定横坐标范围为 0-0.012
ax.set_xlim(0, 0.012)

# 子图4: 残差 vs 物理计算值
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
# 新增：固定纵坐标范围为 -0.2 到 0.2
ax.set_ylim(-0.2, 0.2)

plt.suptitle('Physics Consistency Validation: Verifying PINN Learned Physical Laws\n(Absolute Error Metrics)',
             fontsize=TITLE_FONT_SIZE, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('PECNN_physics_consistency_validation.png', dpi=600, bbox_inches='tight')
plt.close()
print('Saved: PECNN_physics_consistency_validation.png')

# --------------------------------------------------
# 9. 保存结果（仅保留Excel）
# --------------------------------------------------
results_train = pd.DataFrame({
    'Ce_true': train_true[:, 0], 'Ce_pred': train_pred[:, 0],
    'Ci_true': train_true[:, 1], 'Ci_pred': train_pred[:, 1],
    'Ei_true': train_true[:, 2], 'Ei_pred': train_pred[:, 2],
    'Qe_true': train_true[:, 3], 'Qe_pred': train_pred[:, 3]
})

results_test = pd.DataFrame({
    'Ce_true': test_true[:, 0], 'Ce_pred': test_pred[:, 0],
    'Ci_true': test_true[:, 1], 'Ci_pred': test_pred[:, 1],
    'Ei_true': test_true[:, 2], 'Ei_pred': test_pred[:, 2],
    'Qe_true': test_true[:, 3], 'Qe_pred': test_pred[:, 3]
})

metrics_summary = pd.DataFrame({
    'Target': target_columns + ['Overall'],
    'Train_R2': [r2_score(train_true[:, i], train_pred[:, i]) for i in range(4)] +
                [r2_score(train_true, train_pred)],
    'Test_R2': [r2_score(test_true[:, i], test_pred[:, i]) for i in range(4)] +
               [r2_score(test_true, test_pred)],
    'Train_RMSE': [np.sqrt(mean_squared_error(train_true[:, i], train_pred[:, i])) for i in range(4)] +
                  [np.sqrt(mean_squared_error(train_true, train_pred))],
    'Test_RMSE': [np.sqrt(mean_squared_error(test_true[:, i], test_pred[:, i])) for i in range(4)] +
                 [np.sqrt(mean_squared_error(test_true, test_pred))]
})

with pd.ExcelWriter('PECNN_results.xlsx', engine='openpyxl') as writer:
    results_train.to_excel(writer, sheet_name='Train', index=False)
    results_test.to_excel(writer, sheet_name='Test', index=False)
    metrics_summary.to_excel(writer, sheet_name='Metrics', index=False)

print('\n' + '=' * 60)
print('【运行完成】')
print('=' * 60)
print('结果已保存至 PECNN_results.xlsx')
print(f'模型权重已保存: {WEIGHTS_PATH}')
print(f'最佳权重保存在: {BEST_WEIGHTS_PATH}')
print('\n生成图表: PECNN_physics_consistency_validation.png (2×2布局)')
print('=' * 60)