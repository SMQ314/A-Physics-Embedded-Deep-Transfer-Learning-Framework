import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ===============================================
# 全局字体配置 - 使用您指定的大字体
# ===============================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 使用您指定的大字体设置
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 24  # 增大X轴刻度字体
plt.rcParams['ytick.labelsize'] = 24  # 增大Y轴刻度字体
plt.rcParams['legend.fontsize'] = 20

# ===============================================
# 全局配置
# ===============================================
TRAIN_FLAG = False
WEIGHTS_PATH = 'pecnn_weights.weights.h5'
BEST_WEIGHTS_PATH = 'best_weights.weights.h5'
X_SCALER_PATH = 'x_scaler_pecnn.pkl'
Y_SCALER_PATH = 'y_scaler_pecnn.pkl'

# --------------------------------------------------
# 1. 读取数据 (Source Domain Dataset)
# --------------------------------------------------
df = pd.read_excel('D20-60 H40-120 s0.25-0.5.xlsx')
feature_columns = ['L', 'D', 'H', 'S', 'C', 'lamda', 'p', 'a', 'X']
target_columns = ['Ce', 'Ci', 'Ei', 'Qe']
X = df[feature_columns].values.astype('float32')
y = df[target_columns].values.astype('float32')
print(f"特征列: {feature_columns}")
print(f"目标列: {target_columns}")
print(f"数据形状: X={X.shape}, y={y.shape}")

# 检查原始数据范围
print("\n原始目标变量范围：")
print(df[target_columns].describe())

# --------------------------------------------------
# 2. 数据预处理
# --------------------------------------------------
x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)
X_scaled = x_scaler.transform(X)
y_scaled = y_scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
)
print(f'训练集: {X_train.shape}, 测试集: {X_test.shape}')

# 保存原始数据用于PDP计算
X_train_original = X_train.copy()


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


# --------------------------------------------------
# 4. PDP计算函数（修复：返回标准化值用于反归一化）
# --------------------------------------------------
def partial_dependence_1d(model, scaler, feature_idx, X_background, output_idx, n_grid=50):
    """
    计算单个特征的一维部分依赖
    """
    # 获取特征在原始数据中的取值范围（不添加边距）
    feature_values = X_background[:, feature_idx]
    grid_min, grid_max = feature_values.min(), feature_values.max()
    grid_values = np.linspace(grid_min, grid_max, n_grid)

    # 创建修改后的数据集
    X_modified = np.tile(X_background, (n_grid, 1))

    # 替换目标特征值
    for i, val in enumerate(grid_values):
        start_idx = i * len(X_background)
        end_idx = (i + 1) * len(X_background)
        X_modified[start_idx:end_idx, feature_idx] = val

    # 预测（输出是标准化尺度）
    predictions = model.predict(X_modified, verbose=0)

    # 提取目标输出并计算平均值（标准化尺度）
    target_preds = predictions[:, output_idx].reshape(n_grid, len(X_background))
    pdp_values_scaled = target_preds.mean(axis=1)  # 标准化尺度的PDP

    # 反归一化到原始尺度
    grid_full = np.zeros((n_grid, X_background.shape[1]))
    grid_full[:, feature_idx] = grid_values
    grid_original = scaler.inverse_transform(grid_full)[:, feature_idx]

    return grid_original, pdp_values_scaled


# --------------------------------------------------
# 5. 绘制PDP图（3x3布局，修复：反归一化PDP值）
# --------------------------------------------------
print("\n" + "=" * 50)
print("开始绘制PDP图...")
print("=" * 50)

os.makedirs('pdp_results', exist_ok=True)

# 使用训练集作为背景数据（或采样）
np.random.seed(42)
if len(X_train) > 500:
    pdp_background = X_train[np.random.choice(len(X_train), 500, replace=False)]
else:
    pdp_background = X_train

# 为每个目标绘制3x3的PDP图
for output_idx, target_name in enumerate(target_columns):
    print(f"\n绘制目标: {target_name} ({output_idx + 1}/{len(target_columns)})")

    # 创建图形，减小整体宽度使X轴变窄
    fig = plt.figure(figsize=(10, 20), dpi=300)

    # 使用GridSpec来精确控制子图布局
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(3, 3, figure=fig,
                  left=0.08, right=0.96,
                  bottom=0.08, top=0.92,
                  wspace=0.9, hspace=0.30)

    for j, feature in enumerate(feature_columns):
        ax = fig.add_subplot(gs[j // 3, j % 3])

        # 计算PDP
        grid, pdp_scaled = partial_dependence_1d(model, x_scaler, j, pdp_background, output_idx)

        # 反归一化PDP到原始尺度
        pdp_full = np.zeros((len(pdp_scaled), len(target_columns)))
        pdp_full[:, output_idx] = pdp_scaled
        pdp = y_scaler.inverse_transform(pdp_full)[:, output_idx]

        # 使用seaborn绘制线条
        sns.lineplot(x=grid, y=pdp, color="#2b83ba", lw=2.5, ax=ax)
        ax.fill_between(grid, pdp, alpha=0.2, color="#2b83ba")

        # 样式设置
        ax.set_ylabel("PDP", fontsize=22, labelpad=5)
        ax.tick_params(axis='both', labelsize=24, pad=2)
        ax.grid(True, linestyle=':', alpha=0.3)

        # 保持原始数据范围
        x_min, x_max = grid.min(), grid.max()
        ax.set_xlim(x_min, x_max)

        # 自动调整y轴边距
        y_min, y_max = pdp.min(), pdp.max()
        if y_min >= 0:
            ax.set_ylim(0, y_max * 1.1)
        else:
            y_margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # 设置稀疏的刻度
        ax.locator_params(axis='x', nbins=3)
        ax.locator_params(axis='y', nbins=5)

        # 特征p特殊处理：刻度值除以10的幂次，X轴标签添加×10ⁿ
        if feature == 'p':
            # 计算合适的10的幂次（如10⁵）
            magnitude = np.floor(np.log10(abs(x_max)))
            power = int(magnitude)
            scale = 10 ** power

            # 获取当前刻度值并缩放
            xticks = ax.get_xticks()
            # 过滤在范围内的刻度
            xticks = xticks[(xticks >= x_min) & (xticks <= x_max)]
            # 刻度值除以缩放因子，保留1位整数
            xtick_scaled = xticks / scale
            # 格式化为1位整数
            xtick_labels = [f'{x:.0f}' for x in xtick_scaled]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)

            # X轴标签添加×10ⁿ
            ax.set_xlabel(f'p ($\\times$10$^{power}$)', fontsize=22, labelpad=5)
        else:
            ax.set_xlabel(feature, fontsize=22, labelpad=5)
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        # 调试：打印PDP范围
        print(f"  特征 {feature}: PDP范围 [{pdp.min():.4f}, {pdp.max():.4f}]")

    # 总标题
    fig.suptitle(f"1D Partial Dependence Plots ({target_name})",
                 y=0.98, fontsize=26, fontweight='semibold')

    # 保存
    save_path = f'pdp_results/PDP_1D_{target_name}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"  已保存: {save_path}")
    plt.close()

print("\n" + "=" * 50)
print("所有PDP图绘制完成！保存在 pdp_results/ 目录")
print("=" * 50)