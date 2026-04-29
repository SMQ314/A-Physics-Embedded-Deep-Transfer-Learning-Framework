import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import warnings
import matplotlib.ticker as ticker
warnings.filterwarnings('ignore')

# ===============================================
# 全局字体配置
# ===============================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


# ===============================================
# 全局配置
# ===============================================
TRAIN_FLAG = False
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
# 2. 数据预处理（加载保存的scaler）
# --------------------------------------------------
x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)
X_scaled = x_scaler.transform(X)
y_scaled = y_scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
)
print(f'训练集: {X_train.shape}, 测试集: {X_test.shape}')

X_train_original = x_scaler.inverse_transform(X_train)


# --------------------------------------------------
# 3. 构建模型
# --------------------------------------------------
def build_pecnn_model(feat_dim, output_dim):
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
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model


model = build_pecnn_model(X_train.shape[1], y_train.shape[1])
if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print(f'已加载权重: {WEIGHTS_PATH}')
elif os.path.exists(BEST_WEIGHTS_PATH):
    model.load_weights(BEST_WEIGHTS_PATH)
    print(f'已加载权重: {BEST_WEIGHTS_PATH}')
else:
    raise FileNotFoundError('未找到权重文件！')


# ===============================================
# 4. 二维部分依赖图（2D PDP）分析
# ===============================================
def partial_dependence_2d(model, scaler, feature_indices, X_train_original, output_index, n_grid=50):
    """计算二维部分依赖"""
    fi, fj = feature_indices
    n_features = X_train_original.shape[1]

    grid_i = np.linspace(X_train_original[:, fi].min(), X_train_original[:, fi].max(), n_grid)
    grid_j = np.linspace(X_train_original[:, fj].min(), X_train_original[:, fj].max(), n_grid)

    gi_scaled = (grid_i - scaler.mean_[fi]) / scaler.scale_[fi]
    gj_scaled = (grid_j - scaler.mean_[fj]) / scaler.scale_[fj]

    xx, yy = np.meshgrid(gi_scaled, gj_scaled)

    samples = np.zeros((n_grid * n_grid, n_features))
    for k in range(n_features):
        if k == fi:
            samples[:, k] = xx.ravel()
        elif k == fj:
            samples[:, k] = yy.ravel()
        else:
            samples[:, k] = np.mean(X_train[:, k])

    preds = model.predict(samples, batch_size=256, verbose=0)

    dummy_y = np.zeros((n_grid * n_grid, y_scaler.mean_.shape[0]))
    dummy_y[:, output_index] = preds[:, output_index]
    pdp_raw = y_scaler.inverse_transform(dummy_y)[:, output_index]
    pdp = pdp_raw.reshape(n_grid, n_grid)

    return grid_i, grid_j, pdp


# 定义要分析的特征对
top_features = [5, 8, 4]

# 为每个目标变量生成2D PDP图
for t_idx, target_name in enumerate(target_columns):
    feature_pairs = list(combinations(top_features, 2))
    n_pairs = len(feature_pairs)

    fig = plt.figure(figsize=(20, 12), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.4)

    for i, (feat1, feat2) in enumerate(feature_pairs, 1):
        ax = plt.subplot(2, 3, i)

        grid_x, grid_y, pdp_2d = partial_dependence_2d(
            model, x_scaler, [feat1, feat2], X_train_original, t_idx, n_grid=50
        )

        XX, YY = np.meshgrid(grid_x, grid_y)

        contour = ax.contourf(XX, YY, pdp_2d, levels=15, cmap="viridis")

        vmin, vmax = pdp_2d.min(), pdp_2d.max()
        cbar_ticks = [vmin, (vmin + vmax) / 2, vmax]
        cbar = plt.colorbar(contour, ax=ax, label=f'{target_name}', shrink=0.9,
                            ticks=cbar_ticks)
        # 设置colorbar刻度格式为保留2位小数
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(target_name, size=14)

        ax.set_xlabel(feature_columns[feat1], fontsize=16, labelpad=5)
        ax.set_ylabel(feature_columns[feat2], fontsize=16, labelpad=5)
        ax.tick_params(axis='both', labelsize=20)

        # 修复：使用 np.ptp() 替代 pdp_2d.ptp()
        stats_text = (
            f"Max: {pdp_2d.max():.3f}\n"
            f"Min: {pdp_2d.min():.3f}\n"
            f"Range: {np.ptp(pdp_2d):.3f}"
        )
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=24,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_title(f"{feature_columns[feat1]} vs {feature_columns[feat2]}",
                     fontsize=16, pad=10)

    plt.suptitle(f"2D Partial Dependence Plots for {target_name} (Top 4 Features)",
                 fontsize=20, y=0.98, fontweight='semibold')

    plt.savefig(f'2D_PDP_{target_name}mean22.png', bbox_inches='tight', dpi=300)

    print(f"{target_name} 的2D PDP图已保存为 2D_PDP_{target_name}.png")

print("2D PDP分析完成。")