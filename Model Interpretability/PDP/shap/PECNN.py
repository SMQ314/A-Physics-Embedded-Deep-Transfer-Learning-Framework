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
LABEL_FONT_SIZE = 25
TICK_FONT_SIZE = 25
LEGEND_FONT_SIZE = 14
ANNOT_FONT_SIZE = 20

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

    print("\nSHAP 和 PDP 分析完成，图片已保存。")
