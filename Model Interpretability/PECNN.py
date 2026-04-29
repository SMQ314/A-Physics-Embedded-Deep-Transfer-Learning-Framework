import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from itertools import combinations
from scipy import stats

warnings.filterwarnings('ignore')

# ===============================================
# 全局字体配置 - 统一使用第一段的字体设置
# ===============================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 统一使用第一段的字体大小设置
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
SHAP_PDP_FLAG = True  # 是否生成 SHAP 和 PDP 图
ADVANCED_PLOT_FLAG = True  # 是否生成高级学术图表
MC_DROPOUT_SAMPLES = 100  # MC Dropout 采样次数

WEIGHTS_PATH = 'pecnn_weights.weights.h5'
BEST_WEIGHTS_PATH = 'best_weights.weights.h5'
X_SCALER_PATH = 'x_scaler_pecnn.pkl'
Y_SCALER_PATH = 'y_scaler_pecnn.pkl'
# ===============================================
'''
# 删除不兼容的旧权重文件
old_files_to_remove = [WEIGHTS_PATH, BEST_WEIGHTS_PATH, 'train_history_pecnn.pkl']
for file_path in old_files_to_remove:
    if os.path.exists(file_path):
        print(f'删除旧版本文件: {file_path}')
        os.remove(file_path)
'''
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
# 3. 自定义物理约束损失层（保留，未使用）
# --------------------------------------------------
class PhysicsInformedLossLayer(layers.Layer):
    """
    自定义层：计算物理约束损失
    输入: [y_true, y_pred, x_original]
    输出: 总损失 (MSE + λ * physics_loss)
    """

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
# 4. 构建 PECNN 模型 (Functional API)
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
# 5. 训练或加载模型（损失统一到原始空间）
# --------------------------------------------------
if TRAIN_FLAG:
    print('【训练模式】开始训练 PECNN 模型...')

    model = build_pecnn_model(X_train.shape[1], y_train.shape[1], x_scaler, y_scaler)
    model.summary()

    # 扩展历史记录，用于高级图表
    history = {
        'loss': [], 'mse_loss': [], 'physics_loss': [], 'physics_abs_error': [],
        'val_loss': [], 'val_mse_loss': [], 'val_physics_loss': [], 'val_physics_abs_error': []
    }

    # 在 TRAIN_FLAG 分支内，训练开始前添加 scaler 张量（与之前相同）
    y_scale_tf = tf.constant(y_scaler.scale_, dtype=tf.float32)
    y_mean_tf = tf.constant(y_scaler.mean_, dtype=tf.float32)
    x_scale_tf = tf.constant(x_scaler.scale_, dtype=tf.float32)
    x_mean_tf = tf.constant(x_scaler.mean_, dtype=tf.float32)
    pi_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)

    # 训练参数
    max_epochs = 200
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    lambda_physics = 0.5  # 标准化空间下，建议设为 1.0 或 0.5

    optimizer = tf.keras.optimizers.Adam(1e-3)

    # 扩展历史记录
    history = {
        'loss': [], 'mse_loss': [], 'physics_loss': [], 'physics_abs_error': [],
        'val_loss': [], 'val_mse_loss': [], 'val_physics_loss': [], 'val_physics_abs_error': []
    }

    for epoch in range(max_epochs):
        current_epoch = epoch + 1

        # 训练阶段
        epoch_losses = []
        epoch_mse = []
        epoch_physics = []
        epoch_abs_error = []

        for i in range(0, len(X_train), 64):
            batch_x = X_train[i:i + 64]
            batch_y = y_train[i:i + 64]

            with tf.GradientTape() as tape:
                y_pred = model(batch_x, training=True)

                # 1) 标准化空间的 MSE 损失
                loss_mse = tf.reduce_mean(tf.square(batch_y - y_pred))

                # 2) 物理约束损失（标准化空间）
                # 反标准化预测值到原始空间
                y_pred_denorm = y_pred * y_scale_tf + y_mean_tf
                Ei_pred = y_pred_denorm[:, 2:3]
                Qe_pred = y_pred_denorm[:, 3:4]

                # 反标准化输入
                x_denorm = batch_x * x_scale_tf + x_mean_tf
                D_nm = x_denorm[:, 1:2]
                S = x_denorm[:, 3:4]
                a_deg = x_denorm[:, 7:8]

                # 物理公式计算原始 Qe_physics
                denominator = tf.square(D_nm / (S + 1e-8))
                a_rad = a_deg * pi_180
                Qe_physics_raw = (Ei_pred / (denominator + 1e-8) * 1e18) / tf.sin(a_rad)

                # 将 Qe_physics_raw 标准化，得到 Qe_physics_scaled
                Qe_physics_scaled = (Qe_physics_raw - y_mean_tf[3]) / y_scale_tf[3]

                # 标准化空间的物理约束损失（比较 Qe_pred_scaled 和 Qe_physics_scaled）
                Qe_pred_scaled = y_pred[:, 3:4]  # 直接取标准化后的 Qe
                loss_physics = tf.reduce_mean(tf.square(Qe_pred_scaled - Qe_physics_scaled))

                # 计算绝对误差（标准化空间，可选）
                abs_error = tf.reduce_mean(tf.abs(Qe_pred_scaled - Qe_physics_scaled))

                total_loss = loss_mse + lambda_physics * loss_physics

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_losses.append(total_loss.numpy())
            epoch_mse.append(loss_mse.numpy())
            epoch_physics.append(loss_physics.numpy())
            epoch_abs_error.append(abs_error.numpy())

        # 验证阶段
        val_pred = model.predict(X_test, verbose=0)
        val_pred_tf = tf.constant(val_pred, dtype=tf.float32)
        y_test_tf = tf.constant(y_test, dtype=tf.float32)
        X_test_tf = tf.constant(X_test, dtype=tf.float32)

        # 验证 MSE（标准化空间）
        val_loss_mse = tf.reduce_mean(tf.square(y_test_tf - val_pred_tf))

        # 验证物理损失（标准化空间）
        val_pred_denorm = val_pred_tf * y_scale_tf + y_mean_tf
        val_Ei = val_pred_denorm[:, 2:3]
        val_Qe = val_pred_denorm[:, 3:4]

        val_x_denorm = X_test_tf * x_scale_tf + x_mean_tf
        val_D = val_x_denorm[:, 1:2]
        val_S = val_x_denorm[:, 3:4]
        val_a_deg = val_x_denorm[:, 7:8]

        val_denominator = tf.square(val_D / (val_S + 1e-8))
        val_a_rad = val_a_deg * pi_180
        val_Qe_physics_raw = (val_Ei / (val_denominator + 1e-8) * 1e18) / tf.sin(val_a_rad)
        val_Qe_physics_scaled = (val_Qe_physics_raw - y_mean_tf[3]) / y_scale_tf[3]

        val_Qe_pred_scaled = val_pred_tf[:, 3:4]
        val_loss_physics = tf.reduce_mean(tf.square(val_Qe_pred_scaled - val_Qe_physics_scaled))
        val_abs_error = tf.reduce_mean(tf.abs(val_Qe_pred_scaled - val_Qe_physics_scaled))

        val_total_loss = val_loss_mse + lambda_physics * val_loss_physics

        # 记录历史
        history['loss'].append(np.mean(epoch_losses))
        history['mse_loss'].append(np.mean(epoch_mse))
        history['physics_loss'].append(np.mean(epoch_physics))
        history['physics_abs_error'].append(np.mean(epoch_abs_error))
        history['val_loss'].append(val_total_loss.numpy())
        history['val_mse_loss'].append(val_loss_mse.numpy())
        history['val_physics_loss'].append(val_loss_physics.numpy())
        history['val_physics_abs_error'].append(val_abs_error.numpy())

        print(f'Epoch {current_epoch}/{max_epochs} - '
              f'loss: {history["loss"][-1]:.6f} - '
              f'val_loss: {history["val_loss"][-1]:.6f} - '
              f'phy_abs_error: {history["physics_abs_error"][-1]:.4e}')

        # Early stopping
        if val_total_loss < best_val_loss:
            improvement = best_val_loss - val_total_loss
            best_val_loss = val_total_loss
            patience_counter = 0
            model.save_weights(BEST_WEIGHTS_PATH)
            print(f'  ↳ 验证损失改善 {improvement:.6f}，已保存到 {BEST_WEIGHTS_PATH}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {current_epoch}')
                break

    # 加载最佳权重并保存最终模型
    if os.path.exists(BEST_WEIGHTS_PATH):
        print(f'\n训练完成，加载最佳权重: {BEST_WEIGHTS_PATH}')
        model.load_weights(BEST_WEIGHTS_PATH)

    model.save_weights(WEIGHTS_PATH)
    joblib.dump(x_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)
    joblib.dump(history, 'train_history_pecnn.pkl')
    print(f'最终权重已保存到: {WEIGHTS_PATH}')

else:
    print('【预测模式】加载已训练模型...')

    # 确保历史记录结构完整（兼容旧版本）
    default_history = {
        'loss': [], 'mse_loss': [], 'physics_loss': [], 'physics_abs_error': [],
        'val_loss': [], 'val_mse_loss': [], 'val_physics_loss': [], 'val_physics_abs_error': []
    }

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

    loaded_history = joblib.load('train_history_pecnn.pkl')
    for key in default_history:
        if key not in loaded_history:
            loaded_history[key] = default_history[key]
    history = loaded_history


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
print('【模型原始预测结果】(Qe不使用强制物理约束)')
print('=' * 60)
print('训练集:')
train_metrics = calc_metrics(train_true, train_pred, 'Train')
print('-' * 60)
print('测试集:')
test_metrics = calc_metrics(test_true, test_pred, 'Test')
print('=' * 60)

# 验证物理约束满足程度（绝对误差）
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
# 8. 基础图表（第一段代码原有）
# --------------------------------------------------

# 8.1 Loss曲线
plt.figure(figsize=(8, 5))
plt.plot(history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
plt.xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
plt.ylabel('Physics-Informed Loss', fontsize=LABEL_FONT_SIZE)
plt.title('Training Loss Curve (Physics-Informed)', fontsize=TITLE_FONT_SIZE, fontweight='bold')
plt.legend(fontsize=LEGEND_FONT_SIZE)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('PECNN_loss_curve.png', dpi=600)
print('Saved: PECNN_loss_curve.png')

# 8.2 各目标散点图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
colors = {'Train': '#1f77b4', 'Test': '#ff7f0e'}

for i, target_name in enumerate(target_columns):
    ax = axes[i]
    ax.scatter(train_true[:, i], train_pred[:, i],
               c=colors['Train'], label='Train', alpha=0.6, s=30, edgecolors='none')
    ax.scatter(test_true[:, i], test_pred[:, i],
               c=colors['Test'], label='Test', alpha=0.6, s=30, edgecolors='none')
    min_val = min(train_true[:, i].min(), test_true[:, i].min())
    max_val = max(train_true[:, i].max(), test_true[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Fit')
    r2_train = r2_score(train_true[:, i], train_pred[:, i])
    r2_test = r2_score(test_true[:, i], test_pred[:, i])
    ax.set_xlabel(f'Actual {target_name}', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(f'Predicted {target_name}', fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'{target_name}\nTrain R²={r2_train:.3f}, Test R²={r2_test:.3f}', fontsize=SUBTITLE_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=0.3)

plt.suptitle('PECNN Prediction Results', fontsize=TITLE_FONT_SIZE, fontweight='bold')
plt.tight_layout()
plt.savefig('PECNN_scatter_plots.png', dpi=600)
print('Saved: PECNN_scatter_plots.png')

# 8.3 残差图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, target_name in enumerate(target_columns):
    ax = axes[i]
    train_residuals = train_true[:, i] - train_pred[:, i]
    test_residuals = test_true[:, i] - test_pred[:, i]
    ax.scatter(train_pred[:, i], train_residuals,
               c=colors['Train'], alpha=0.6, s=30, label='Train')
    ax.scatter(test_pred[:, i], test_residuals,
               c=colors['Test'], alpha=0.6, s=30, label='Test')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
    ax.set_xlabel(f'Predicted {target_name}', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Residuals', fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'{target_name} Residuals', fontsize=SUBTITLE_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=0.3)

plt.suptitle('Residual Plots', fontsize=TITLE_FONT_SIZE, fontweight='bold')
plt.tight_layout()
plt.savefig('PECNN_residuals.png', dpi=600)
print('Saved: PECNN_residuals.png')

# --------------------------------------------------
# 9. 置换重要性分析（第一段代码详细版）
# --------------------------------------------------
print('\n【计算置换重要性...】')


class KerasModelMultiOutputWrapper:
    def __init__(self, keras_model, y_scaler):
        self.model = keras_model
        self.y_scaler = y_scaler

    def fit(self, X, y):
        return self

    def predict(self, X):
        pred_scaled = self.model.predict(X, verbose=0)
        return self.y_scaler.inverse_transform(pred_scaled)

    def score(self, X, y):
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        return -mse


wrapped_model = KerasModelMultiOutputWrapper(model, y_scaler)
y_test_true = y_scaler.inverse_transform(y_test)
output_dim = y_test.shape[1]

start_time = time.time()
print("开始计算置换重要性...")

all_importances = []
all_importances_std = []

for target_idx in range(output_dim):
    print(f"计算目标 {target_columns[target_idx]} 的特征重要性...")


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
        n_repeats=5,
        random_state=42,
        scoring='neg_mean_squared_error',
        n_jobs=1
    )
    all_importances.append(result.importances_mean)
    all_importances_std.append(result.importances_std)

importance = np.mean(all_importances, axis=0)
importance_std = np.mean(all_importances_std, axis=0)

elapsed = time.time() - start_time
print(f'置换重要性计算完成，耗时: {elapsed:.2f} 秒')

# 保存每个目标的特征重要度到Excel
importance_df = pd.DataFrame(index=feature_columns)
for target_idx, target_name in enumerate(target_columns):
    importance_df[target_name] = all_importances[target_idx]
    importance_df[f'{target_name}_std'] = all_importances_std[target_idx]
importance_df['Mean_Importance'] = np.mean(all_importances, axis=0)
importance_df['Mean_Std'] = np.mean(all_importances_std, axis=0)
importance_df = importance_df.sort_values('Mean_Importance', ascending=False)
importance_df.to_excel('PECNN_feature_importance_per_target.xlsx')
print('\n特征重要度（每个目标）已保存至: PECNN_feature_importance_per_target.xlsx')

# 打印每个目标的重要度排名
print('\n【各目标特征重要性排名】')
for target_idx, target_name in enumerate(target_columns):
    print(f'\n目标: {target_name}')
    target_imp = all_importances[target_idx]
    sorted_idx_target = np.argsort(target_imp)[::-1]
    for i, idx in enumerate(sorted_idx_target):
        print(
            f'  {i + 1:2d}. {feature_columns[idx]:20s} : {target_imp[idx]:.6f} ± {all_importances_std[target_idx][idx]:.6f}')

# 打印平均重要度排名
print('\n【平均特征重要性排名】')
mean_importance = np.mean(all_importances, axis=0)
mean_std = np.mean(all_importances_std, axis=0)
sorted_idx_mean = np.argsort(mean_importance)[::-1]
for i, idx in enumerate(sorted_idx_mean):
    print(f'{i + 1:2d}. {feature_columns[idx]:20s} : {mean_importance[idx]:.6f} ± {mean_std[idx]:.6f}')

# 绘制置换重要性条形图（使用平均重要度）
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(mean_importance)[::-1]
plt.barh(range(len(feature_columns)), mean_importance[sorted_idx],
         xerr=mean_std[sorted_idx],
         color='steelblue', alpha=0.8, capsize=3)
plt.yticks(range(len(feature_columns)), [feature_columns[i] for i in sorted_idx])
plt.xlabel('Permutation Importance (Mean across targets)', fontsize=LABEL_FONT_SIZE)
plt.title('Feature Importance (Permutation Importance)', fontsize=TITLE_FONT_SIZE, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('PECNN_permutation_importance.png', dpi=300)
plt.close()
print('Saved: PECNN_permutation_importance.png')

# --------------------------------------------------
# 10. SHAP 和 PDP 分析（第一段代码原有）
# --------------------------------------------------
if SHAP_PDP_FLAG:
    print("\n" + "=" * 50)
    print("开始 SHAP 和 PDP 分析...")
    print("=" * 50)

    n_background = min(1000, X_train.shape[0])
    n_samples_shap = 200

    all_features = feature_columns.copy()
    n_features_all = len(all_features)

    top_for_pairs = [feature_columns[i] for i in sorted_idx_mean[:3]]
    feature_pairs = [(top_for_pairs[i], top_for_pairs[j])
                     for i in range(len(top_for_pairs))
                     for j in range(i + 1, len(top_for_pairs))]

    # ---------- SHAP 分析 ----------
    try:
        import shap
    except ImportError:
        print("请先安装 shap 库：pip install shap")
        raise

    print("\n【SHAP 分析 - 使用 DeepExplainer】")

    background = X_train[:n_background]
    X_test_shap = X_test[:n_samples_shap]
    y_test_shap = y_test_true[:n_samples_shap]

    print(f"计算 SHAP 值（样本数: {n_samples_shap}, 背景数: {n_background}）...")

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test_shap, check_additivity=False)

    print(f"shap_values 类型: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"目标数: {len(shap_values)}")
        print(f"每个目标的 SHAP 值形状: {shap_values[0].shape}")
    else:
        print(f"shap_values 形状: {shap_values.shape}")

    # 为每个目标绘制 SHAP 图
    for output_index, target_name in enumerate(target_columns):
        print(f"\n绘制目标 {target_name} 的 SHAP 图...")

        if isinstance(shap_values, list):
            shap_values_output = shap_values[output_index]
        else:
            if len(shap_values.shape) == 3:
                shap_values_output = shap_values[:, :, output_index]
            else:
                shap_values_output = shap_values

        if len(shap_values_output.shape) > 2:
            shap_values_output = shap_values_output.squeeze()

        print(f"shap_values_output 形状: {shap_values_output.shape}")

        X_test_original = x_scaler.inverse_transform(X_test_shap)

        n_features = len(feature_columns)
        if shap_values_output.shape[1] != n_features:
            print(f"警告: SHAP 特征数 ({shap_values_output.shape[1]}) 与特征列数 ({n_features}) 不匹配")
            if shap_values_output.shape[1] > n_features:
                shap_values_output = shap_values_output[:, :n_features]

        # SHAP 散点图（3行3列）
        n_rows = 3
        n_cols = 3

        plt.figure(figsize=(5 * n_cols, 4 * n_rows), dpi=300)

        for j, feature in enumerate(feature_columns):
            ax = plt.subplot(n_rows, n_cols, j + 1)

            scatter = ax.scatter(X_test_original[:, j], shap_values_output[:, j],
                                 c=X_test_original[:, j], cmap='viridis',
                                 alpha=0.8, edgecolor='w', linewidth=0.5,
                                 s=50, zorder=2)

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.05)
            cbar.set_label('Feature Value', fontsize=TICK_FONT_SIZE, labelpad=0)
            cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

            sns.regplot(x=X_test_original[:, j], y=shap_values_output[:, j],
                        scatter=False, ci=95,
                        line_kws={'color': '#d62728', 'lw': 1.5, 'alpha': 0.9},
                        ax=ax, truncate=True)

            ax.set_xlabel(f'{feature}', fontsize=LABEL_FONT_SIZE, labelpad=0)
            ax.set_ylabel('SHAP Value', fontsize=LABEL_FONT_SIZE, labelpad=0)
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
            ax.grid(True, which='both', linestyle=':', alpha=0.3)

            corr = np.corrcoef(X_test_original[:, j], shap_values_output[:, j])[0, 1]
            ax.text(0.97, 0.10, f'ρ = {corr:.2f}',
                    transform=ax.transAxes, ha='right',
                    fontsize=ANNOT_FONT_SIZE,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none',
                              boxstyle='round,pad=0.2'))

        plt.suptitle(f'SHAP Value Relationships - {target_name}',
                     y=0.995, fontsize=TITLE_FONT_SIZE, fontweight='semibold')
        plt.tight_layout(pad=1.5, h_pad=0.8, w_pad=0.8)
        plt.subplots_adjust(bottom=0.08, top=0.94, left=0.06, right=0.98)

        save_path = f'PECNN_shap_scatterplots_{target_name}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        plt.close()

        # SHAP Beeswarm 图
        try:
            if hasattr(explainer, 'expected_value'):
                if isinstance(explainer.expected_value, list):
                    expected_value = explainer.expected_value[output_index]
                else:
                    expected_value = explainer.expected_value
                if hasattr(expected_value, 'numpy'):
                    expected_value = expected_value.numpy()
            else:
                expected_value = np.mean(y_test_shap[:, output_index])

            shap_exp = shap.Explanation(
                values=shap_values_output,
                base_values=expected_value,
                data=X_test_original,
                feature_names=feature_columns,
                output_names=[target_name]
            )

            plt.figure(figsize=(10, 8), dpi=300)
            shap.plots.beeswarm(
                shap_exp,
                max_display=len(feature_columns),
                color=plt.colormaps['viridis'],
                axis_color="#333333",
                alpha=0.8,
                show=False
            )

            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.grid(True, which='major', linestyle='--', alpha=0.3)

            plt.title(f"SHAP Feature Importance - {target_name}",
                      fontsize=TITLE_FONT_SIZE, pad=15, fontweight='semibold')
            plt.xlabel("SHAP Value Impact", fontsize=LABEL_FONT_SIZE, labelpad=10)
            plt.ylabel("Features", fontsize=LABEL_FONT_SIZE, labelpad=10)

            plt.tight_layout()
            save_path = f'PECNN_SHAP_beeswarm_{target_name}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
            print(f"Saved: {save_path}")
            plt.close()

        except Exception as e:
            print(f"Beeswarm 图绘制失败: {e}")
            print("跳过 beeswarm 图...")

    # ---------- 1D PDP ----------
    print("\n【1D 部分依赖图 - 手动计算】")


    def compute_pdp_1d(model, y_scaler, X_data, feature_idx, target_idx,
                       n_grid=50, n_samples=None):
        X_data = np.asarray(X_data)
        n_samples_total = X_data.shape[0]
        if n_samples is not None and n_samples < n_samples_total:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_samples_total, n_samples, replace=False)
            X_sample = X_data[idx]
        else:
            X_sample = X_data
        feature_values = X_data[:, feature_idx]
        grid_values = np.linspace(feature_values.min(), feature_values.max(), n_grid)
        pdp_values = []
        for val in grid_values:
            X_modified = X_sample.copy()
            X_modified[:, feature_idx] = val
            pred_scaled = model.predict(X_modified, verbose=0)
            pred = y_scaler.inverse_transform(pred_scaled)
            pdp_values.append(np.mean(pred[:, target_idx]))
        return grid_values, np.array(pdp_values)


    n_samples_pdp = min(500, X_test.shape[0])

    for target_idx, target_name in enumerate(target_columns):
        print(f"\n计算目标 {target_name} 的1D PDP...")

        n_cols = 3
        n_rows = int(np.ceil(n_features_all / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=300)
        if n_features_all == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_features_all > 1 else [axes]

        for idx, feat_name in enumerate(all_features):
            feat_idx = feature_columns.index(feat_name)
            grid_vals, pdp_vals = compute_pdp_1d(
                model, y_scaler, X_test, feat_idx, target_idx,
                n_grid=50, n_samples=n_samples_pdp
            )
            ax = axes[idx]
            ax.plot(grid_vals, pdp_vals, color='red', linewidth=2.5, label='Partial Dependence')
            ax.fill_between(grid_vals, pdp_vals, alpha=0.3, color='red')
            ax.set_xlabel(feat_name, fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel(f'Predicted {target_name}', fontsize=LABEL_FONT_SIZE)
            ax.set_title(f'{feat_name}', fontsize=SUBTITLE_FONT_SIZE)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=TICK_FONT_SIZE)

        for idx in range(n_features_all, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'1D Partial Dependence Plots - {target_name} (All Features)',
                     fontsize=TITLE_FONT_SIZE, fontweight='semibold', y=1.02)
        plt.tight_layout()
        save_path = f'PECNN_pdp_1d_all_{target_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    # ---------- 2D PDP ----------
    print("\n【2D 部分依赖图 - 手动计算】")


    def compute_pdp_2d(model, y_scaler, X_data, feature_idx1, feature_idx2,
                       target_idx, n_grid=20, n_samples=None):
        X_data = np.asarray(X_data)
        n_samples_total = X_data.shape[0]
        if n_samples is not None and n_samples < n_samples_total:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_samples_total, n_samples, replace=False)
            X_sample = X_data[idx]
        else:
            X_sample = X_data
        feat1_values = X_data[:, feature_idx1]
        feat2_values = X_data[:, feature_idx2]
        grid_x = np.linspace(feat1_values.min(), feat1_values.max(), n_grid)
        grid_y = np.linspace(feat2_values.min(), feat2_values.max(), n_grid)
        XX, YY = np.meshgrid(grid_x, grid_y)
        pdp_values = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                X_modified = X_sample.copy()
                X_modified[:, feature_idx1] = XX[i, j]
                X_modified[:, feature_idx2] = YY[i, j]
                pred_scaled = model.predict(X_modified, verbose=0)
                pred = y_scaler.inverse_transform(pred_scaled)
                pdp_values[i, j] = np.mean(pred[:, target_idx])
        return grid_x, grid_y, pdp_values


    n_samples_2d = min(300, X_test.shape[0])
    n_grid_2d = 15
    n_pairs = len(feature_pairs)

    for target_idx, target_name in enumerate(target_columns):
        print(f"\n计算目标 {target_name} 的2D PDP...")
        n_rows = int(np.ceil(n_pairs / 3))
        n_cols = min(3, n_pairs)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=300)
        if n_pairs == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_pairs > 1 else [axes]

        for idx, (feat1_name, feat2_name) in enumerate(feature_pairs):
            feat1_idx = feature_columns.index(feat1_name)
            feat2_idx = feature_columns.index(feat2_name)
            print(f"  计算 {feat1_name} vs {feat2_name}...")
            grid_x, grid_y, pdp_vals = compute_pdp_2d(
                model, y_scaler, X_test, feat1_idx, feat2_idx, target_idx,
                n_grid=n_grid_2d, n_samples=n_samples_2d
            )
            ax = axes[idx]
            contour = ax.contourf(grid_x, grid_y, pdp_vals, levels=20,
                                  cmap='RdYlBu_r', alpha=0.8)
            cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
            cbar.set_label(f'Predicted {target_name}', fontsize=TICK_FONT_SIZE)
            contours = ax.contour(grid_x, grid_y, pdp_vals, levels=10,
                                  colors='black', linewidths=0.5, alpha=0.5)
            ax.clabel(contours, inline=True, fontsize=TICK_FONT_SIZE)
            ax.set_xlabel(feat1_name, fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel(feat2_name, fontsize=LABEL_FONT_SIZE)
            ax.set_title(f'{feat1_name} vs {feat2_name}', fontsize=SUBTITLE_FONT_SIZE)
            ax.tick_params(labelsize=TICK_FONT_SIZE)

        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'2D Partial Dependence Plots - {target_name}',
                     fontsize=TITLE_FONT_SIZE, fontweight='semibold', y=1.02)
        plt.tight_layout()
        save_path = f'PECNN_pdp_2d_all_{target_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    print("\nSHAP 和 PDP 分析完成，图片已保存。")

# --------------------------------------------------
# 11. 高级学术图表（第二段代码）- 修改：统一字体并生成单图
# --------------------------------------------------
if ADVANCED_PLOT_FLAG:
    print("\n" + "=" * 60)
    print("【绘制高级学术分析图表】")
    print("=" * 60)

    # 11.1 物理约束收敛分析图 (Physics-Informed Loss Decomposition)
    print('\n【1/4 物理约束收敛分析图...】')

    # 生成组图
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 子图1: 总损失和分解损失（双轴）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    line1 = ax1.semilogy(history['mse_loss'], 'b-', linewidth=2.5, label='MSE Loss')[0]
    line2 = ax1_twin.semilogy(history['physics_loss'], 'r--', linewidth=2.5, label='Physics Loss')[0]
    ax1.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax1.set_ylabel('MSE Loss', color='b', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax1_twin.set_ylabel('Physics Loss (λ=0.1)', color='r', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b', labelsize=TICK_FONT_SIZE)
    ax1_twin.tick_params(axis='y', labelcolor='r', labelsize=TICK_FONT_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
    ax1.set_title('Loss Decomposition (Training)', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both', linestyle='--')
    ax1.legend([line1, line2], ['MSE Loss', 'Physics Loss'], loc='upper right', fontsize=LEGEND_FONT_SIZE)

    # 子图2: 验证集损失分解
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    line3 = ax2.semilogy(history['val_mse_loss'], 'b-', linewidth=2.5, label='Val MSE')[0]
    line4 = ax2_twin.semilogy(history['val_physics_loss'], 'r--', linewidth=2.5, label='Val Physics')[0]
    ax2.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax2.set_ylabel('Val MSE Loss', color='b', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax2_twin.set_ylabel('Val Physics Loss', color='r', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='b', labelsize=TICK_FONT_SIZE)
    ax2_twin.tick_params(axis='y', labelcolor='r', labelsize=TICK_FONT_SIZE)
    ax2.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
    ax2.set_title('Loss Decomposition (Validation)', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linestyle='--')
    ax2.legend([line3, line4], ['Val MSE', 'Val Physics'], loc='upper right', fontsize=LEGEND_FONT_SIZE)

    # 子图3: 物理约束绝对误差收敛
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(history['physics_abs_error'], 'g-', linewidth=2.5, label='Train', alpha=0.8)
    ax3.semilogy(history['val_physics_abs_error'], 'orange', linewidth=2.5, linestyle='--', label='Validation',
                 alpha=0.8)
    train_median = np.median(history['physics_abs_error']) if history['physics_abs_error'] else 1e-3
    ax3.axhline(y=train_median, color='r', linestyle=':', linewidth=2, label=f'Median: {train_median:.2e}')
    ax3.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax3.set_ylabel('Mean Absolute Error of Qe', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax3.tick_params(labelsize=TICK_FONT_SIZE)
    ax3.set_title('Physics Constraint Satisfaction\n(Absolute Error)', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
    ax3.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 子图4: 总损失对比 - 修改为与第一段代码相同的格式
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(history['loss'], 'b-', linewidth=2.5, label='Train Total', alpha=0.8)
    ax4.plot(history['val_loss'], 'orange', linewidth=2.5, linestyle='--', label='Val Total', alpha=0.8)
    best_epoch = np.argmin(history['val_loss'])
    ax4.axvline(x=best_epoch, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Best epoch: {best_epoch}')
    ax4.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax4.set_ylabel('Total Loss', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax4.tick_params(labelsize=TICK_FONT_SIZE)
    ax4.set_title('Total Physics-Informed Loss', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
    ax4.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
    ax4.grid(True, alpha=0.3, which='both', linestyle='--')

    # 子图5: 损失占比饼图（最终epoch）
    ax5 = fig.add_subplot(gs[1, 1])
    final_mse = history['mse_loss'][-1] if history['mse_loss'] else 0
    final_physics = history['physics_loss'][-1] * 0.1 if history['physics_loss'] else 0
    sizes = [final_mse, final_physics]
    colors_pie = ['#1f77b4', '#ff7f0e']
    explode = (0.05, 0.05)
    wedges, texts, autotexts = ax5.pie(sizes, explode=explode, colors=colors_pie, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': ANNOT_FONT_SIZE, 'fontweight': 'bold'})
    ax5.legend(wedges, ['MSE Component', 'Physics Component (weighted)'],
               title="Final Loss Composition", loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1),
               fontsize=LEGEND_FONT_SIZE)
    ax5.set_title('Final Epoch Loss Composition', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')

    # 子图6: 过拟合检测 - 泛化间隙
    ax6 = fig.add_subplot(gs[1, 2])
    if len(history['loss']) == len(history['val_loss']):
        gap_mse = np.array(history['val_mse_loss']) - np.array(history['mse_loss'])
        gap_physics = np.array(history['val_physics_loss']) - np.array(history['physics_loss'])
        ax6.plot(gap_mse, 'b-', linewidth=2, label='MSE Gap (Val-Train)', alpha=0.8)
        ax6.plot(gap_physics, 'r--', linewidth=2, label='Physics Gap (Val-Train)', alpha=0.8)
        ax6.axhline(y=0, color='k', linestyle='-', linewidth=1.5, alpha=0.5)
        ax6.fill_between(range(len(gap_mse)), gap_mse, 0, alpha=0.2, color='b')
        ax6.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax6.set_ylabel('Generalization Gap', fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax6.tick_params(labelsize=TICK_FONT_SIZE)
        ax6.set_title('Overfitting Detection', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
        ax6.legend(fontsize=LEGEND_FONT_SIZE)
        ax6.grid(True, alpha=0.3)

    plt.suptitle('Physics-Informed Neural Network Training Dynamics',
                 fontsize=TITLE_FONT_SIZE, fontweight='bold', y=0.98)
    plt.savefig('PECNN_physics_convergence_analysis.png', dpi=600, bbox_inches='tight')
    plt.close()
    print('Saved: PECNN_physics_convergence_analysis.png')

    # 生成单个小图（与第一段代码格式一致）
    # 单图1: Loss曲线（与第一段代码完全一致）
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Physics-Informed Loss', fontsize=LABEL_FONT_SIZE)
    plt.title('Training Loss Curve (Physics-Informed)', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('PECNN_loss_curve_advanced.png', dpi=600)
    print('Saved: PECNN_loss_curve_advanced.png')
    plt.close()

    # 单图2: MSE Loss分解
    plt.figure(figsize=(8, 5))
    plt.semilogy(history['mse_loss'], 'b-', linewidth=2.5, label='Train MSE')
    plt.semilogy(history['val_mse_loss'], 'orange', linewidth=2.5, linestyle='--', label='Val MSE')
    plt.xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('MSE Loss (log scale)', fontsize=LABEL_FONT_SIZE)
    plt.title('MSE Loss Decomposition', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig('PECNN_mse_loss_single.png', dpi=600)
    print('Saved: PECNN_mse_loss_single.png')
    plt.close()

    # 单图3: Physics Loss分解
    plt.figure(figsize=(8, 5))
    plt.semilogy(history['physics_loss'], 'r-', linewidth=2.5, label='Train Physics')
    plt.semilogy(history['val_physics_loss'], 'orange', linewidth=2.5, linestyle='--', label='Val Physics')
    plt.xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Physics Loss (log scale)', fontsize=LABEL_FONT_SIZE)
    plt.title('Physics Loss Decomposition', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig('PECNN_physics_loss_single.png', dpi=600)
    print('Saved: PECNN_physics_loss_single.png')
    plt.close()

    # 单图4: 物理约束绝对误差
    plt.figure(figsize=(8, 5))
    plt.semilogy(history['physics_abs_error'], 'g-', linewidth=2.5, label='Train', alpha=0.8)
    plt.semilogy(history['val_physics_abs_error'], 'orange', linewidth=2.5, linestyle='--', label='Validation',
                 alpha=0.8)
    train_median = np.median(history['physics_abs_error']) if history['physics_abs_error'] else 1e-3
    plt.axhline(y=train_median, color='r', linestyle=':', linewidth=2, label=f'Median: {train_median:.2e}')
    plt.xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Mean Absolute Error of Qe', fontsize=LABEL_FONT_SIZE)
    plt.title('Physics Constraint Satisfaction', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('PECNN_physics_abs_error_single.png', dpi=600)
    print('Saved: PECNN_physics_abs_error_single.png')
    plt.close()

    # 11.2 物理一致性验证图 (Physics Consistency Validation)
    print('\n【2/4 物理一致性验证图...】')

    # 计算物理一致性
    Ei_pred_test = test_pred[:, 2]
    Qe_pred_test = test_pred[:, 3]
    X_test_denorm = x_scaler.inverse_transform(X_test)
    D_test = X_test_denorm[:, 1]
    S_test = X_test_denorm[:, 3]
    a_test = X_test_denorm[:, 7]
    Qe_physics_from_pred = (Ei_pred_test / (np.square(D_test / (S_test + 1e-8)) + 1e-8)) * 1e18 / np.sin(
        np.radians(a_test))

    Ei_pred_train = train_pred[:, 2]
    Qe_pred_train = train_pred[:, 3]
    X_train_denorm = x_scaler.inverse_transform(X_train)
    D_train = X_train_denorm[:, 1]
    S_train = X_train_denorm[:, 3]
    a_train = X_train_denorm[:, 7]
    Qe_physics_from_pred_train = (Ei_pred_train / (np.square(D_train / (S_train + 1e-8)) + 1e-8)) * 1e18 / np.sin(
        np.radians(a_train))

    abs_error_test = np.abs(Qe_pred_test - Qe_physics_from_pred)
    abs_error_train = np.abs(Qe_pred_train - Qe_physics_from_pred_train)

    # 生成组图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 子图1: 测试集散点
    ax = axes[0, 0]
    ax.scatter(Qe_physics_from_pred, Qe_pred_test, c='#2E86AB', alpha=0.6, s=50, edgecolors='none',
               label='Test samples')
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
    ax = axes[0, 1]
    ax.scatter(Qe_physics_from_pred_train, Qe_pred_train, c='#A23B72', alpha=0.5, s=40, edgecolors='none',
               label='Train samples')
    min_val_train = min(Qe_physics_from_pred_train.min(), Qe_pred_train.min())
    max_val_train = max(Qe_physics_from_pred_train.max(), Qe_pred_train.max())
    ax.plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'r--', linewidth=3,
            label='Perfect Consistency')
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
    ax = axes[0, 2]
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
    ax = axes[1, 0]
    residuals_physics = Qe_pred_test - Qe_physics_from_pred
    ax.scatter(Qe_physics_from_pred, residuals_physics, c='#F18F01', alpha=0.6, s=50, edgecolors='none')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2.5)
    ax.set_xlabel(r'$Q_e^{physics}$', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel(r'$Q_e^{predicted} - Q_e^{physics}$', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_title('Physics Residuals vs Fitted Values', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 子图5: 不同物理参数区间的绝对误差分析
    ax = axes[1, 1]
    DS_ratio = D_test / S_test
    DS_bins = np.percentile(DS_ratio, [0, 25, 50, 75, 100])
    DS_labels = ['Q1 (low D/S)', 'Q2', 'Q3', 'Q4 (high D/S)']
    DS_groups = np.digitize(DS_ratio, DS_bins[1:-1]) + 1
    mean_abs_errors_by_group = [abs_error_test[DS_groups == i].mean() for i in range(1, 5)]
    std_abs_errors_by_group = [abs_error_test[DS_groups == i].std() for i in range(1, 5)]
    x_pos = np.arange(len(DS_labels))
    bars = ax.bar(x_pos, mean_abs_errors_by_group, yerr=std_abs_errors_by_group,
                  color=['#E63946', '#F4A261', '#2A9D8F', '#264653'],
                  capsize=5, edgecolor='black', linewidth=1.2, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(DS_labels, fontsize=TICK_FONT_SIZE)
    ax.set_ylabel('Mean Absolute Error', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_title('Physics Absolute Error by D/S Ratio Quartiles', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mean_abs_errors_by_group):
        height = bar.get_height()
        ax.annotate(f'{val:.2e}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=ANNOT_FONT_SIZE,
                    fontweight='bold')

    # 子图6: Q-Q图检验正态性
    ax = axes[1, 2]
    stats.probplot(residuals_physics, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Physics Residuals Normality', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.get_lines()[0].set_markerfacecolor('#2E86AB')
    ax.get_lines()[0].set_markeredgecolor('none')
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('r')
    ax.get_lines()[1].set_linewidth(2.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Physics Consistency Validation: Verifying PINN Learned Physical Laws\n(Absolute Error Metrics)',
                 fontsize=TITLE_FONT_SIZE, fontweight='bold', y=0.98)
    plt.savefig('PECNN_physics_consistency_validation.png', dpi=600, bbox_inches='tight')
    plt.close()
    print('Saved: PECNN_physics_consistency_validation.png')

    # 生成单个小图
    # 单图1: 测试集散点
    plt.figure(figsize=(8, 6))
    plt.scatter(Qe_physics_from_pred, Qe_pred_test, c='#2E86AB', alpha=0.6, s=50, edgecolors='none',
                label='Test samples')
    min_val = min(Qe_physics_from_pred.min(), Qe_pred_test.min())
    max_val = max(Qe_physics_from_pred.max(), Qe_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='Perfect Consistency')
    plt.xlabel(r'$Q_e^{physics}$ (calculated from predicted $E_i$)', fontsize=LABEL_FONT_SIZE)
    plt.ylabel(r'$Q_e^{predicted}$', fontsize=LABEL_FONT_SIZE)
    plt.title('Test Set: Physics Consistency Check', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.text(0.05, 0.95, f'$R^2$ = {r2_physics_test:.4f}\nMAE = {mae_physics_test:.4e}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'),
             fontsize=ANNOT_FONT_SIZE, verticalalignment='top', fontweight='bold')
    plt.legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('PECNN_physics_consistency_test_single.png', dpi=600)
    print('Saved: PECNN_physics_consistency_test_single.png')
    plt.close()

    # 单图2: 训练集散点
    plt.figure(figsize=(8, 6))
    plt.scatter(Qe_physics_from_pred_train, Qe_pred_train, c='#A23B72', alpha=0.5, s=40, edgecolors='none',
                label='Train samples')
    min_val_train = min(Qe_physics_from_pred_train.min(), Qe_pred_train.min())
    max_val_train = max(Qe_physics_from_pred_train.max(), Qe_pred_train.max())
    plt.plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'r--', linewidth=3,
             label='Perfect Consistency')
    plt.xlabel(r'$Q_e^{physics}$', fontsize=LABEL_FONT_SIZE)
    plt.ylabel(r'$Q_e^{predicted}$', fontsize=LABEL_FONT_SIZE)
    plt.title('Train Set: Physics Consistency Check', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.text(0.05, 0.95, f'$R^2$ = {r2_physics_train:.4f}\nMAE = {mae_physics_train:.4e}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'),
             fontsize=ANNOT_FONT_SIZE, verticalalignment='top', fontweight='bold')
    plt.legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('PECNN_physics_consistency_train_single.png', dpi=600)
    print('Saved: PECNN_physics_consistency_train_single.png')
    plt.close()

    # 单图3: 绝对误差分布
    plt.figure(figsize=(8, 6))
    bins = np.linspace(0, max(np.percentile(abs_error_test, 95), np.percentile(abs_error_train, 95)), 50)
    plt.hist(abs_error_train, bins=bins, color='#A23B72', alpha=0.6, label='Train', edgecolor='black', linewidth=0.5)
    plt.hist(abs_error_test, bins=bins, color='#2E86AB', alpha=0.6, label='Test', edgecolor='black', linewidth=0.5)
    plt.axvline(x=np.mean(abs_error_train), color='#A23B72', linestyle='--', linewidth=2.5,
                label=f'Train Mean: {np.mean(abs_error_train):.2e}')
    plt.axvline(x=np.mean(abs_error_test), color='#2E86AB', linestyle='--', linewidth=2.5,
                label=f'Test Mean: {np.mean(abs_error_test):.2e}')
    plt.axvline(x=np.median(abs_error_test), color='navy', linestyle='-.', linewidth=2,
                label=f'Test Median: {np.median(abs_error_test):.2e}')
    plt.xlabel('Absolute Error of Physics Consistency', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Frequency', fontsize=LABEL_FONT_SIZE)
    plt.title('Distribution of Physics Consistency Absolute Error', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('PECNN.png', dpi=600)
    print('Saved: PECNNsingle.png')
    plt.close()
    # 接续上文，补全物理一致性验证图未完成的单个小图部分
    # 单图3: 绝对误差分布（已在前面部分完成，此处无需重复）
    # 单图4: 残差 vs 物理计算值
    plt.figure(figsize=(8, 6))
    plt.scatter(Qe_physics_from_pred, residuals_physics, c='#F18F01', alpha=0.6, s=50, edgecolors='none')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2.5)
    plt.xlabel(r'$Q_e^{physics}$', fontsize=LABEL_FONT_SIZE)
    plt.ylabel(r'$Q_e^{predicted} - Q_e^{physics}$', fontsize=LABEL_FONT_SIZE)
    plt.title('Physics Residuals vs Fitted Values', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('PECNN_physics_residuals_single.png', dpi=600)
    print('Saved: PECNN_physics_residuals_single.png')
    plt.close()

    # 单图5: D/S比分组误差图
    plt.figure(figsize=(8, 6))
    x_pos = np.arange(len(DS_labels))
    bars = plt.bar(x_pos, mean_abs_errors_by_group, yerr=std_abs_errors_by_group,
                   color=['#E63946', '#F4A261', '#2A9D8F', '#264653'],
                   capsize=5, edgecolor='black', linewidth=1.2, alpha=0.8)
    plt.xticks(x_pos, DS_labels, fontsize=TICK_FONT_SIZE)
    plt.ylabel('Mean Absolute Error', fontsize=LABEL_FONT_SIZE)
    plt.title('Physics Absolute Error by D/S Ratio Quartiles', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mean_abs_errors_by_group):
        height = bar.get_height()
        plt.annotate(f'{val:.2e}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=ANNOT_FONT_SIZE,
                     fontweight='bold')
    plt.tight_layout()
    plt.savefig('PECNN_physics_ds_error_single.png', dpi=600)
    print('Saved: PECNN_physics_ds_error_single.png')
    plt.close()

    # 单图6: Q-Q图
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals_physics, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Physics Residuals Normality', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('PECNN_physics_qqplot_single.png', dpi=600)
    print('Saved: PECNN_physics_qqplot_single.png')
    plt.close()

    # 打印物理一致性指标
    print('\n【物理一致性验证结果（绝对误差）】')
    print(f'测试集 - R² (Predicted vs Physics): {r2_physics_test:.4f}')
    print(f'测试集 - Mean Absolute Error: {np.mean(abs_error_test):.4e}')
    print(f'测试集 - Median Absolute Error: {np.median(abs_error_test):.4e}')
    print(f'测试集 - Max Absolute Error: {np.max(abs_error_test):.4e}')
    print(f'训练集 - R² (Predicted vs Physics): {r2_physics_train:.4f}')
    print(f'训练集 - Mean Absolute Error: {np.mean(abs_error_train):.4e}')

    # 11.3 预测不确定性量化图 (MC Dropout)
    print('\n【3/4 预测不确定性量化图...】')


    def mc_dropout_predict(model, X, n_samples=MC_DROPOUT_SAMPLES):
        predictions = []
        for _ in range(n_samples):
            pred = model(X, training=True)
            predictions.append(y_scaler.inverse_transform(pred.numpy()))
        return np.array(predictions)


    n_samples_for_uncertainty = min(200, X_test.shape[0])
    X_test_uncertainty = X_test[:n_samples_for_uncertainty]
    y_test_uncertainty_true = test_true[:n_samples_for_uncertainty]

    print(f"执行MC Dropout ({MC_DROPOUT_SAMPLES}次采样) 用于{n_samples_for_uncertainty}个样本...")
    mc_predictions = mc_dropout_predict(model, X_test_uncertainty, MC_DROPOUT_SAMPLES)

    mc_mean = mc_predictions.mean(axis=0)
    mc_std = mc_predictions.std(axis=0)
    mc_ci_lower = np.percentile(mc_predictions, 2.5, axis=0)
    mc_ci_upper = np.percentile(mc_predictions, 97.5, axis=0)

    # 组图：4个目标的不确定性区间
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors_unc = ['#E63946', '#F4A261', '#2A9D8F', '#264653']

    for i, target in enumerate(target_columns):
        ax = axes[i]
        sort_idx = np.argsort(y_test_uncertainty_true[:, i])
        y_sorted = y_test_uncertainty_true[sort_idx, i]
        mean_sorted = mc_mean[sort_idx, i]
        std_sorted = mc_std[sort_idx, i]
        ci_lower_sorted = mc_ci_lower[sort_idx, i]
        ci_upper_sorted = mc_ci_upper[sort_idx, i]
        x_plot = np.arange(len(y_sorted))

        ax.fill_between(x_plot, ci_lower_sorted, ci_upper_sorted, alpha=0.3, color=colors_unc[i],
                        label='95% CI (MC Dropout)')
        ax.fill_between(x_plot, mean_sorted - std_sorted, mean_sorted + std_sorted, alpha=0.5, color=colors_unc[i],
                        label='±1 Std Dev')
        ax.plot(x_plot, mean_sorted, '-', color=colors_unc[i], linewidth=2, label='Predicted Mean', zorder=3)
        ax.scatter(x_plot, y_sorted, c='black', s=30, alpha=0.6, label='True Values', zorder=4, edgecolors='white',
                   linewidth=0.5)

        in_ci = ((y_sorted >= ci_lower_sorted) & (y_sorted <= ci_upper_sorted)).mean()
        ax.set_xlabel('Sample Index (sorted by true value)', fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylabel(target, fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax.set_title(f'{target} with Uncertainty Quantification\n(Coverage: {in_ci * 100:.1f}%)',
                     fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
        ax.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Prediction Uncertainty via Monte Carlo Dropout', fontsize=TITLE_FONT_SIZE, fontweight='bold', y=0.98)
    plt.savefig('PECNN_uncertainty_quantification.png', dpi=600, bbox_inches='tight')
    plt.close()
    print('Saved: PECNN_uncertainty_quantification.png')

    # 为每个目标单独保存不确定性图
    for i, target in enumerate(target_columns):
        plt.figure(figsize=(8, 6))
        sort_idx = np.argsort(y_test_uncertainty_true[:, i])
        y_sorted = y_test_uncertainty_true[sort_idx, i]
        mean_sorted = mc_mean[sort_idx, i]
        ci_lower_sorted = mc_ci_lower[sort_idx, i]
        ci_upper_sorted = mc_ci_upper[sort_idx, i]
        x_plot = np.arange(len(y_sorted))

        plt.fill_between(x_plot, ci_lower_sorted, ci_upper_sorted, alpha=0.3, color=colors_unc[i], label='95% CI')
        plt.plot(x_plot, mean_sorted, '-', color=colors_unc[i], linewidth=2, label='Predicted Mean')
        plt.scatter(x_plot, y_sorted, c='black', s=30, alpha=0.6, label='True Values', zorder=4)

        in_ci = ((y_sorted >= ci_lower_sorted) & (y_sorted <= ci_upper_sorted)).mean()
        plt.xlabel('Sample Index (sorted by true value)', fontsize=LABEL_FONT_SIZE)
        plt.ylabel(target, fontsize=LABEL_FONT_SIZE)
        plt.title(f'{target}: Uncertainty Quantification (Coverage {in_ci * 100:.1f}%)', fontsize=TITLE_FONT_SIZE,
                  fontweight='bold')
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'PECNN_uncertainty_{target}_single.png', dpi=600)
        print(f'Saved: PECNN_uncertainty_{target}_single.png')
        plt.close()

    # 不确定性校准图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, target in enumerate(target_columns):
        ax = axes[i]
        pred_error = np.abs(y_test_uncertainty_true[:, i] - mc_mean[:, i])
        pred_uncertainty = mc_std[:, i]
        ax.scatter(pred_uncertainty, pred_error, c=colors_unc[i], alpha=0.6, s=50, edgecolors='none')
        z = np.polyfit(pred_uncertainty, pred_error, 1)
        p = np.poly1d(z)
        x_line = np.linspace(pred_uncertainty.min(), pred_uncertainty.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=2.5, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        corr = np.corrcoef(pred_uncertainty, pred_error)[0, 1]
        ax.set_xlabel('Predictive Uncertainty (Std Dev)', fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylabel('Prediction Error |True - Pred|', fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax.set_title(f'{target}: Error vs Uncertainty (ρ={corr:.3f})', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Uncertainty Calibration: Prediction Error vs Model Uncertainty', fontsize=TITLE_FONT_SIZE,
                 fontweight='bold', y=0.98)
    plt.savefig('PECNN_uncertainty_calibration.png', dpi=600, bbox_inches='tight')
    plt.close()
    print('Saved: PECNN_uncertainty_calibration.png')

    # 为每个目标单独保存校准图
    for i, target in enumerate(target_columns):
        plt.figure(figsize=(8, 6))
        pred_error = np.abs(y_test_uncertainty_true[:, i] - mc_mean[:, i])
        pred_uncertainty = mc_std[:, i]
        plt.scatter(pred_uncertainty, pred_error, c=colors_unc[i], alpha=0.6, s=50, edgecolors='none')
        z = np.polyfit(pred_uncertainty, pred_error, 1)
        p = np.poly1d(z)
        x_line = np.linspace(pred_uncertainty.min(), pred_uncertainty.max(), 100)
        plt.plot(x_line, p(x_line), 'k--', linewidth=2.5, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        corr = np.corrcoef(pred_uncertainty, pred_error)[0, 1]
        plt.xlabel('Predictive Uncertainty (Std Dev)', fontsize=LABEL_FONT_SIZE)
        plt.ylabel('Prediction Error |True - Pred|', fontsize=LABEL_FONT_SIZE)
        plt.title(f'{target}: Error vs Uncertainty (ρ={corr:.3f})', fontsize=TITLE_FONT_SIZE, fontweight='bold')
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'PECNN_uncertainty_calibration_{target}_single.png', dpi=600)
        print(f'Saved: PECNN_uncertainty_calibration_{target}_single.png')
        plt.close()

    # 11.4 特征交互热力图 (H-statistic)
    print('\n【4/4 特征交互热力图...】')


    def compute_h_statistic_empirical(model, y_scaler, X, feature_i, feature_j, n_grid=20):
        X = np.asarray(X)
        n_samples = min(500, X.shape[0])
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], n_samples, replace=False)
        X_sample = X[idx]

        xi_vals = np.linspace(X[:, feature_i].min(), X[:, feature_i].max(), n_grid)
        xj_vals = np.linspace(X[:, feature_j].min(), X[:, feature_j].max(), n_grid)

        pdp_joint = np.zeros((n_grid, n_grid))
        for ii, vi in enumerate(xi_vals):
            for jj, vj in enumerate(xj_vals):
                X_temp = X_sample.copy()
                X_temp[:, feature_i] = vi
                X_temp[:, feature_j] = vj
                pred = model.predict(X_temp, verbose=0)
                pred_denorm = y_scaler.inverse_transform(pred)
                pdp_joint[ii, jj] = pred_denorm[:, 3].mean()  # 只针对 Qe

        pdp_i = pdp_joint.mean(axis=1)
        pdp_j = pdp_joint.mean(axis=0)
        pdp_product = np.outer(pdp_i, pdp_j)
        numerator = np.sum((pdp_joint - pdp_product) ** 2)
        denominator = np.sum(pdp_joint ** 2)
        if denominator < 1e-10:
            return 0.0
        return min(numerator / denominator, 1.0)


    n_features = len(feature_columns)
    interaction_matrix = np.zeros((n_features, n_features))

    print("计算特征交互矩阵 (这可能需要几分钟)...")
    for i, j in combinations(range(n_features), 2):
        print(f"  计算 {feature_columns[i]} vs {feature_columns[j]}...")
        h_val = compute_h_statistic_empirical(model, y_scaler, X_test, i, j, n_grid=15)
        interaction_matrix[i, j] = h_val
        interaction_matrix[j, i] = h_val

    # 组图：完整热力图 + 排序条形图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    mask = np.triu(np.ones_like(interaction_matrix, dtype=bool), k=1)
    sns.heatmap(interaction_matrix, mask=~mask, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=feature_columns, yticklabels=feature_columns,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "H-statistic"},
                ax=ax, vmin=0, vmax=1, annot_kws={"size": ANNOT_FONT_SIZE, "weight": "bold"})
    ax.set_title('Feature Interaction Strength (H-statistic)\nfor Qe Prediction', fontsize=SUBTITLE_FONT_SIZE,
                 fontweight='bold')

    ax = axes[1]
    interactions = []
    labels = []
    for i, j in combinations(range(n_features), 2):
        interactions.append(interaction_matrix[i, j])
        labels.append(f"{feature_columns[i]} × {feature_columns[j]}")
    sorted_idx = np.argsort(interactions)[::-1]
    top_n = min(10, len(interactions))
    colors_bar = plt.cm.YlOrRd(np.linspace(0.3, 0.9, top_n))
    bars = ax.barh(range(top_n), [interactions[i] for i in sorted_idx[:top_n]],
                   color=colors_bar, edgecolor='black', linewidth=1)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([labels[i] for i in sorted_idx[:top_n]], fontsize=TICK_FONT_SIZE)
    ax.set_xlabel('H-statistic (Interaction Strength)', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_title('Top Feature Interactions', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, [interactions[i] for i in sorted_idx[:top_n]])):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=ANNOT_FONT_SIZE, fontweight='bold')

    plt.suptitle('Feature Interaction Analysis: Revealing Physical Parameter Couplings',
                 fontsize=TITLE_FONT_SIZE, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('PECNN_feature_interaction_heatmap.png', dpi=600, bbox_inches='tight')
    plt.close()
    print('Saved: PECNN_feature_interaction_heatmap.png')

    # 单独保存热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=feature_columns, yticklabels=feature_columns,
                square=True, linewidths=0.5, cbar_kws={"label": "H-statistic"},
                vmin=0, vmax=1, annot_kws={"size": ANNOT_FONT_SIZE, "weight": "bold"})
    plt.title('Feature Interaction Strength (H-statistic) for Qe Prediction', fontsize=TITLE_FONT_SIZE,
              fontweight='bold')
    plt.tight_layout()
    plt.savefig('PECNN_interaction_heatmap_single.png', dpi=600)
    print('Saved: PECNN_interaction_heatmap_single.png')
    plt.close()

    # 单独保存排序条形图
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(interactions)[::-1]
    top_n = min(10, len(interactions))
    colors_bar = plt.cm.YlOrRd(np.linspace(0.3, 0.9, top_n))
    plt.barh(range(top_n), [interactions[i] for i in sorted_idx[:top_n]],
             color=colors_bar, edgecolor='black', linewidth=1)
    plt.yticks(range(top_n), [labels[i] for i in sorted_idx[:top_n]], fontsize=TICK_FONT_SIZE)
    plt.xlabel('H-statistic (Interaction Strength)', fontsize=LABEL_FONT_SIZE)
    plt.title('Top Feature Interactions', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    for i, val in enumerate([interactions[i] for i in sorted_idx[:top_n]]):
        plt.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=ANNOT_FONT_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.savefig('PECNN_interaction_top_single.png', dpi=600)
    print('Saved: PECNN_interaction_top_single.png')
    plt.close()

    # 打印 Top 5 交互
    print('\n【Top 5 特征交互】')
    for i in sorted_idx[:5]:
        print(f"{labels[i]}: H = {interactions[i]:.4f}")

    # 11.5 双特征PDP交互图 (2D PDP for Top Interactions)
    print('\n【附加: 2D PDP交互图...】')

    top_pairs = []
    for i in sorted_idx[:2]:
        pair = labels[i].replace(' × ', ' ').split()
        top_pairs.append((feature_columns.index(pair[0]), feature_columns.index(pair[1]),
                          pair[0], pair[1]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (fi, fj, fn_i, fn_j) in enumerate(top_pairs):
        n_grid_2d = 20
        X_plot = X_test[:min(300, X_test.shape[0])]

        xi_range = np.linspace(X_test[:, fi].min(), X_test[:, fi].max(), n_grid_2d)
        xj_range = np.linspace(X_test[:, fj].min(), X_test[:, fj].max(), n_grid_2d)
        XI, XJ = np.meshgrid(xi_range, xj_range)

        pdp_2d_all = []
        for target_idx in range(4):
            pdp_2d = np.zeros((n_grid_2d, n_grid_2d))
            for ii in range(n_grid_2d):
                for jj in range(n_grid_2d):
                    X_temp = X_plot.copy()
                    X_temp[:, fi] = XI[ii, jj]
                    X_temp[:, fj] = XJ[ii, jj]
                    pred = model.predict(X_temp, verbose=0)
                    pred_denorm = y_scaler.inverse_transform(pred)
                    pdp_2d[ii, jj] = pred_denorm[:, target_idx].mean()
            pdp_2d_all.append(pdp_2d)

        # Qe
        ax = axes[idx, 0]
        contour = ax.contourf(XI, XJ, pdp_2d_all[3], levels=20, cmap='RdYlBu_r', alpha=0.9)
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label(f'Predicted Qe', fontsize=TICK_FONT_SIZE)
        ax.set_xlabel(fn_i, fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylabel(fn_j, fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax.set_title(f'2D PDP: {fn_i} × {fn_j} → Qe', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
        CS = ax.contour(XI, XJ, pdp_2d_all[3], levels=8, colors='black', linewidths=0.8, alpha=0.6)
        ax.clabel(CS, inline=True, fontsize=TICK_FONT_SIZE, fmt='%1.2e')

        # Ei
        ax = axes[idx, 1]
        contour2 = ax.contourf(XI, XJ, pdp_2d_all[2], levels=20, cmap='PRGn', alpha=0.9)
        cbar2 = plt.colorbar(contour2, ax=ax, shrink=0.8)
        cbar2.set_label(f'Predicted Ei', fontsize=TICK_FONT_SIZE)
        ax.set_xlabel(fn_i, fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylabel(fn_j, fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax.set_title(f'2D PDP: {fn_i} × {fn_j} → Ei', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
        CS2 = ax.contour(XI, XJ, pdp_2d_all[2], levels=8, colors='black', linewidths=0.8, alpha=0.6)
        ax.clabel(CS2, inline=True, fontsize=TICK_FONT_SIZE, fmt='%1.2e')

    plt.suptitle('2D Partial Dependence: Top Feature Interactions', fontsize=TITLE_FONT_SIZE, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('PECNN_2D_PDP_interactions.png', dpi=600, bbox_inches='tight')
    plt.close()
    print('Saved: PECNN_2D_PDP_interactions.png')

    # 为每个交互对单独保存2D PDP图（Qe和Ei）
    for idx, (fi, fj, fn_i, fn_j) in enumerate(top_pairs):
        n_grid_2d = 20
        X_plot = X_test[:min(300, X_test.shape[0])]

        xi_range = np.linspace(X_test[:, fi].min(), X_test[:, fi].max(), n_grid_2d)
        xj_range = np.linspace(X_test[:, fj].min(), X_test[:, fj].max(), n_grid_2d)
        XI, XJ = np.meshgrid(xi_range, xj_range)

        pdp_2d_all = []
        for target_idx in range(4):
            pdp_2d = np.zeros((n_grid_2d, n_grid_2d))
            for ii in range(n_grid_2d):
                for jj in range(n_grid_2d):
                    X_temp = X_plot.copy()
                    X_temp[:, fi] = XI[ii, jj]
                    X_temp[:, fj] = XJ[ii, jj]
                    pred = model.predict(X_temp, verbose=0)
                    pred_denorm = y_scaler.inverse_transform(pred)
                    pdp_2d[ii, jj] = pred_denorm[:, target_idx].mean()
            pdp_2d_all.append(pdp_2d)

        # Qe单独
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(XI, XJ, pdp_2d_all[3], levels=20, cmap='RdYlBu_r', alpha=0.9)
        cbar = plt.colorbar(contour)
        cbar.set_label('Predicted Qe', fontsize=TICK_FONT_SIZE)
        plt.xlabel(fn_i, fontsize=LABEL_FONT_SIZE)
        plt.ylabel(fn_j, fontsize=LABEL_FONT_SIZE)
        plt.title(f'2D PDP: {fn_i} × {fn_j} → Qe', fontsize=TITLE_FONT_SIZE, fontweight='bold')
        CS = plt.contour(XI, XJ, pdp_2d_all[3], levels=8, colors='black', linewidths=0.8, alpha=0.6)
        plt.clabel(CS, inline=True, fontsize=TICK_FONT_SIZE, fmt='%1.2e')
        plt.tight_layout()
        plt.savefig(f'PECNN_2D_PDP_{fn_i}_{fn_j}_Qe_single.png', dpi=600)
        print(f'Saved: PECNN_2D_PDP_{fn_i}_{fn_j}_Qe_single.png')
        plt.close()

        # Ei单独
        plt.figure(figsize=(8, 6))
        contour2 = plt.contourf(XI, XJ, pdp_2d_all[2], levels=20, cmap='PRGn', alpha=0.9)
        cbar2 = plt.colorbar(contour2)
        cbar2.set_label('Predicted Ei', fontsize=TICK_FONT_SIZE)
        plt.xlabel(fn_i, fontsize=LABEL_FONT_SIZE)
        plt.ylabel(fn_j, fontsize=LABEL_FONT_SIZE)
        plt.title(f'2D PDP: {fn_i} × {fn_j} → Ei', fontsize=TITLE_FONT_SIZE, fontweight='bold')
        CS2 = plt.contour(XI, XJ, pdp_2d_all[2], levels=8, colors='black', linewidths=0.8, alpha=0.6)
        plt.clabel(CS2, inline=True, fontsize=TICK_FONT_SIZE, fmt='%1.2e')
        plt.tight_layout()
        plt.savefig(f'PECNN_2D_PDP_{fn_i}_{fn_j}_Ei_single.png', dpi=600)
        print(f'Saved: PECNN_2D_PDP_{fn_i}_{fn_j}_Ei_single.png')
        plt.close()

    print('\n' + '=' * 60)
    print('【所有高级图表生成完成】')
    print('=' * 60)

# --------------------------------------------------
# 12. 保存结果
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
if SHAP_PDP_FLAG:
    print('\nSHAP/PDP分析图已保存')
if ADVANCED_PLOT_FLAG:
    print('\n生成的高级学术图表:')
    print('  1. PECNN_physics_convergence_analysis.png - 物理损失分解与收敛（组图）')
    print(
        '     - 子图单独保存：PECNN_loss_curve_advanced.png, PECNN_mse_loss_single.png, PECNN_physics_loss_single.png, PECNN_physics_abs_error_single.png')
    print('  2. PECNN_physics_consistency_validation.png - 物理一致性验证（组图）')
    print(
        '     - 子图单独保存：PECNN_physics_consistency_test_single.png, PECNN_physics_consistency_train_single.png, PECNN_physics_abs_error_distribution_single.png, PECNN_physics_residuals_single.png, PECNN_physics_ds_error_single.png, PECNN_physics_qqplot_single.png')
    print('  3. PECNN_uncertainty_quantification.png - 预测不确定性量化（组图）')
    print('     - 每个目标单独保存：PECNN_uncertainty_<target>_single.png')
    print('  4. PECNN_uncertainty_calibration.png - 不确定性校准（组图）')
    print('     - 每个目标单独保存：PECNN_uncertainty_calibration_<target>_single.png')
    print('  5. PECNN_feature_interaction_heatmap.png - 特征交互热力图（组图）')
    print('     - 单独保存：PECNN_interaction_heatmap_single.png, PECNN_interaction_top_single.png')
    print('  6. PECNN_2D_PDP_interactions.png - 2D PDP交互效应（组图）')
    print('     - 每个交互对单独保存：PECNN_2D_PDP_<fn_i>_<fn_j>_<target>_single.png')
print('=' * 60)
print('\n预测说明:')
print('  1. 所有目标(Ce, Ci, Ei, Qe)均使用神经网络原始预测值')
print('  2. 训练时使用了Qe的物理约束损失作为正则化，且 MSE 与物理损失均在原始空间计算，量纲一致')
print('  3. 预测时不强制应用物理约束，保留模型学习的映射关系')
print('  4. 物理约束验证使用绝对误差指标')