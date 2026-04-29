import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import warnings
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
TRAIN_FLAG = False          # 仅预测模式，不训练
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

# 划分训练/测试集（仅用于PDP计算时的背景数据）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
)
print(f'训练集: {X_train.shape}, 测试集: {X_test.shape}')

# --------------------------------------------------
# 3. 构建模型（与训练时架构一致）
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
# 4. 部分依赖图（PDP）分析（修正版）
# ===============================================
def compute_pdp_for_feature(model, X_background, feature_idx, grid_points_scaled,
                            y_scaler, target_idx):
    """
    计算单个特征的部分依赖曲线（针对单个目标输出）
    参数:
        model: 训练好的模型
        X_background: 背景数据（标准化后的特征矩阵，shape=(n_samples, n_features)）
        feature_idx: 要分析的列索引
        grid_points_scaled: 网格点的标准化值（一维数组）
        y_scaler: 目标标准化器
        target_idx: 目标变量的索引（0~3）
    返回:
        pdp_raw: 网格点对应的目标预测值（原始尺度）
    """
    n_points = len(grid_points_scaled)
    n_samples = X_background.shape[0]
    # 存储每个网格点的平均预测（标准化尺度）
    y_avg_scaled = np.zeros(n_points)
    # 对每个网格点计算平均预测
    for i, val_scaled in enumerate(grid_points_scaled):
        X_temp = X_background.copy()
        X_temp[:, feature_idx] = val_scaled
        pred_scaled = model.predict(X_temp, batch_size=256, verbose=0)
        y_avg_scaled[i] = pred_scaled[:, target_idx].mean()
    # 将平均预测逆变换回原始尺度
    dummy_y = np.zeros((n_points, y_scaler.mean_.shape[0]))
    dummy_y[:, target_idx] = y_avg_scaled
    pdp_raw = y_scaler.inverse_transform(dummy_y)[:, target_idx]
    return pdp_raw

# 获取背景数据（标准化后的训练集）
X_background = X_train  # 使用训练集作为背景
n_features = X_background.shape[1]
feature_names = feature_columns
target_names = target_columns

# 为每个特征生成网格点（原始尺度 + 标准化尺度）
n_grid = 50
grids_raw = []      # 原始尺度网格点（用于绘图横轴）
grids_scaled = []   # 标准化尺度网格点（用于模型输入）
for i, feat_name in enumerate(feature_names):
    # 原始数据中的取值范围
    raw_min = df[feat_name].min()
    raw_max = df[feat_name].max()
    grid_raw = np.linspace(raw_min, raw_max, n_grid)
    grids_raw.append(grid_raw)
    # 手动标准化：z = (x - mean) / scale
    mean = x_scaler.mean_[i]
    scale = x_scaler.scale_[i]
    grid_scaled = (grid_raw - mean) / scale
    grids_scaled.append(grid_scaled)

# 为每个目标变量生成3×3子图
for t_idx, target_name in enumerate(target_names):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for f_idx, (feat_name, grid_raw, grid_scaled) in enumerate(zip(feature_names, grids_raw, grids_scaled)):
        print(f"正在计算 {target_name} 关于特征 {feat_name} 的PDP...")
        pdp_vals = compute_pdp_for_feature(
            model, X_background, f_idx, grid_scaled,
            y_scaler, t_idx
        )
        ax = axes[f_idx]
        ax.plot(grid_raw, pdp_vals, 'b-', linewidth=2)
        ax.set_xlabel(feat_name, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(target_name, fontsize=LABEL_FONT_SIZE)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    # 隐藏多余的子图（如果有9个特征则正好，否则隐藏）
    for j in range(len(feature_names), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Partial Dependence Plots for {target_name}', fontsize=TITLE_FONT_SIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'PDP_{target_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"{target_name} 的PDP图已保存为 PDP_{target_name}.png")

print("PDP分析完成。")