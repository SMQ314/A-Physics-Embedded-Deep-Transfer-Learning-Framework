import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ===============================================
# 配置参数（请根据实际情况修改）
# ===============================================
DATA_FILE = 'ALL.xlsx'  # 包含所有参数组合的Excel文件
SOURCE_WEIGHTS = 'best_weights.weights.h5'  # 源模型权重
X_SCALER_PATH = 'x_scaler_pecnn.pkl'  # 源域的特征标准化器
Y_SCALER_PATH = 'y_scaler_pecnn.pkl'  # 源域的目标标准化器

# 迁移学习超参数
INITIAL_LR = 1e-4  # 微调初始学习率
BATCH_SIZE = 64
MAX_EPOCHS_PER_STAGE = 300  # 每个阶段最大轮数
PATIENCE = 15  # 早停耐心值
LAMBDA_PHYSICS = 0.5  # 物理约束损失权重（标准化空间下，建议设为1.0或0.5）
MIGRATION_FRACTION = 0.2  # 每个阶段新参数组合使用的样本比例（分层抽样）

OUTPUT_DIR = 'transfer_learning_results'  # 输出根目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================================
# 辅助函数（保持不变）
# ===============================================
def plot_scatter(y_true, y_pred, target_names, save_path):
    n_targets = len(target_names)
    n_cols = 2
    n_rows = (n_targets + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, name in enumerate(target_names):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=30, edgecolors='none')
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        ax.set_xlabel(f'Actual {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} (R²={r2:.3f})')
        ax.grid(True, alpha=0.3)
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle('Real vs Predicted')
    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=300)
    plt.close()


def calc_metrics(y_true, y_pred, suffix='', return_dict=False):
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
    if return_dict:
        return {'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    else:
        return {'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}


# ===============================================
# 修改点：物理约束损失现在返回标准化空间的值
# ===============================================
def physics_loss(y_true_scaled, y_pred_scaled, x_original, y_scaler, x_scaler):
    """
    计算标准化空间的物理约束损失（Qe）
    """
    # 反标准化预测值到原始空间
    y_scale = tf.constant(y_scaler.scale_, dtype=tf.float32)
    y_mean = tf.constant(y_scaler.mean_, dtype=tf.float32)
    y_pred_denorm = y_pred_scaled * y_scale + y_mean
    Ei_pred = y_pred_denorm[:, 2:3]
    Qe_pred = y_pred_denorm[:, 3:4]  # 原始空间 Qe 预测值

    # 反标准化输入
    x_scale = tf.constant(x_scaler.scale_, dtype=tf.float32)
    x_mean = tf.constant(x_scaler.mean_, dtype=tf.float32)
    x_denorm = x_original * x_scale + x_mean
    D_nm = x_denorm[:, 1:2]
    S = x_denorm[:, 3:4]
    a_deg = x_denorm[:, 7:8]

    # 计算原始空间的物理 Qe
    denominator = tf.square(D_nm / (S + 1e-8))
    a_rad = a_deg * tf.constant(np.pi / 180.0, dtype=tf.float32)
    Qe_physics_raw = (Ei_pred / (denominator + 1e-8) * 1e18) / tf.sin(a_rad)

    # 将 Qe_physics_raw 标准化，得到 Qe_physics_scaled
    Qe_physics_scaled = (Qe_physics_raw - y_mean[3]) / y_scale[3]

    # 标准化空间的 Qe 预测值
    Qe_pred_scaled = y_pred_scaled[:, 3:4]

    # 物理损失 = 标准化空间 MSE
    loss_physics = tf.reduce_mean(tf.square(Qe_pred_scaled - Qe_physics_scaled))
    return loss_physics


def combined_loss(y_true_scaled, y_pred_scaled, x_original, y_scaler, x_scaler, lambda_phys):
    loss_mse = tf.reduce_mean(tf.square(y_true_scaled - y_pred_scaled))
    loss_phys = physics_loss(y_true_scaled, y_pred_scaled, x_original, y_scaler, x_scaler)
    return loss_mse + lambda_phys * loss_phys


def train_phase(model, X_train, y_train, X_val, y_val, epochs, lr, lambda_phys, phase_name,
                y_scaler, x_scaler):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        epoch_losses = []
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_x = X_train[i:i + BATCH_SIZE]
            batch_y = y_train[i:i + BATCH_SIZE]
            with tf.GradientTape() as tape:
                y_pred = model(batch_x, training=True)
                loss = combined_loss(batch_y, y_pred, batch_x, y_scaler, x_scaler, lambda_phys)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(loss.numpy())

        val_pred = model.predict(X_val, verbose=0)
        val_loss = combined_loss(y_val, val_pred, X_val, y_scaler, x_scaler, lambda_phys).numpy()
        train_loss = np.mean(epoch_losses)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"{phase_name} - Epoch {epoch + 1}/{epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.get_weights()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停触发于 epoch {epoch + 1}")
                break

    if best_weights is not None:
        model.set_weights(best_weights)
        print(f"已恢复最佳验证损失: {best_val_loss:.6f}")

    return history


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
    return model


def sample_by_param_group(df, fraction, random_state=42):
    if fraction >= 1.0:
        return df, pd.DataFrame(columns=df.columns)
    sampled_groups = []
    remaining_groups = []
    for group, group_df in df.groupby('param_group'):
        n_samples = len(group_df)
        if n_samples == 1:
            sampled_groups.append(group_df)
            remaining_groups.append(group_df.iloc[0:0])
        else:
            n_sampled = max(1, int(np.ceil(n_samples * fraction)))
            sampled = group_df.sample(n=n_sampled, random_state=random_state)
            remaining = group_df.drop(sampled.index)
            sampled_groups.append(sampled)
            remaining_groups.append(remaining)
    sampled_df = pd.concat(sampled_groups, ignore_index=True)
    remaining_df = pd.concat(remaining_groups, ignore_index=True)
    return sampled_df, remaining_df


# ===============================================
# 1. 读取数据
# ===============================================
print("=" * 60)
print("1. 读取数据")
print("=" * 60)

df = pd.read_excel(DATA_FILE)
feature_columns = ['L', 'D', 'H', 'S', 'C', 'lamda', 'p', 'a', 'X']
target_columns = ['Ce', 'Ci', 'Ei', 'Qe']

# 添加组合列用于分层抽样
df['param_group'] = df['D'].astype(str) + '_' + df['H'].astype(str) + '_' + df['S'].astype(str)

# ===============================================
# 2. 数据标准化（使用源域scaler）
# ===============================================
print("\n" + "=" * 60)
print("2. 加载预训练模型的scaler并标准化所有数据")
print("=" * 60)

x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

X_all = df[feature_columns].values.astype('float32')
y_all = df[target_columns].values.astype('float32')
X_all_scaled = x_scaler.transform(X_all)
y_all_scaled = y_scaler.transform(y_all)

# 将标准化后的数据附加到DataFrame
df['scaled_X'] = list(X_all_scaled)
df['scaled_y'] = list(y_all_scaled)

print("标准化完成。")

# ===============================================
# 3. 定义源域和目标域参数组合（修复：互斥且完备的阶段定义）
# ===============================================
print("\n" + "=" * 60)
print("3. 定义源域和目标域参数组合")
print("=" * 60)

source_D = [20, 40, 60]
source_H = [40, 80, 120]
source_S = [0.25, 0.5]
target_D_new = [80, 100]  # 新扩展的D
target_H_new = [160, 200]  # 新扩展的H
target_S_new = [0.75]  # 新扩展的S

# 获取源域数据（初始训练集）
source_mask = (df['D'].isin(source_D)) & (df['H'].isin(source_H)) & (df['S'].isin(source_S))
source_df = df[source_mask].copy()
print(f"源域数据量: {len(source_df)}")

# ===============================================
# 修复点：重新定义阶段候选集，确保互斥且完备
# ===============================================

# 阶段1候选：D新，H和S为源域（且不在源域中）
stage1_mask = (df['D'].isin(target_D_new)) & (df['H'].isin(source_H)) & (df['S'].isin(source_S)) & (~source_mask)
stage1_candidates_df = df[stage1_mask].copy()
print(f"阶段1候选数据量 (D新, H源, S源): {len(stage1_candidates_df)}")

# 阶段2候选：H新，S为源域，D任意（但排除已分配的源域和阶段1数据）
# D可以是源域或新域，但H必须是新的，且数据未被分配
assigned_mask = source_mask | stage1_mask
stage2_mask = (df['H'].isin(target_H_new)) & (df['S'].isin(source_S)) & (~assigned_mask)
stage2_candidates_df = df[stage2_mask].copy()
print(f"阶段2候选数据量 (H新, S源, D任意): {len(stage2_candidates_df)}")

# 阶段3候选：S新，D和H任意（但排除已分配的数据）
assigned_mask = source_mask | stage1_mask | stage2_mask
stage3_mask = (df['S'].isin(target_S_new)) & (~assigned_mask)
stage3_candidates_df = df[stage3_mask].copy()
print(f"阶段3候选数据量 (S新, D任意, H任意): {len(stage3_candidates_df)}")

# 验证完备性
total_assigned_mask = source_mask | stage1_mask | stage2_mask | stage3_mask
unassigned_df = df[~total_assigned_mask].copy()
print(f"\n验证：未分配到任何阶段的数据量: {len(unassigned_df)}")

if len(unassigned_df) > 0:
    print("警告：发现未分配的数据！")
    print(unassigned_df.groupby(['D', 'H', 'S']).size())
    # 将这些数据作为额外的测试集保留
else:
    print("验证通过：所有数据已分配到相应阶段")

total_samples = len(df)
print(f"总样本数: {total_samples}")
print(
    f"各阶段候选总和: {len(source_df)} + {len(stage1_candidates_df)} + {len(stage2_candidates_df)} + {len(stage3_candidates_df)} = {len(source_df) + len(stage1_candidates_df) + len(stage2_candidates_df) + len(stage3_candidates_df)}")

# ===============================================
# 4. 构建模型并加载预训练权重
# ===============================================
print("\n" + "=" * 60)
print("4. 构建模型并加载预训练权重")
print("=" * 60)

feat_dim = len(feature_columns)
output_dim = len(target_columns)
model = build_pecnn_model(feat_dim, output_dim)

if os.path.exists(SOURCE_WEIGHTS):
    model.load_weights(SOURCE_WEIGHTS)
    print(f"成功加载预训练权重: {SOURCE_WEIGHTS}")
else:
    raise FileNotFoundError(f"预训练权重文件 {SOURCE_WEIGHTS} 未找到，请检查路径。")

# ===============================================
# 5. 渐进式迁移学习（先D，再H，最后S）
# ===============================================
print("\n" + "=" * 60)
print("5. 开始渐进式迁移学习")
print("=" * 60)

# 用于存储各阶段信息的列表
stages_info = []

# 初始化训练数据（源域数据）
train_df_current = source_df.copy()
print(f"初始训练数据量: {len(train_df_current)}")

# 初始化各个阶段未使用的数据容器
stage1_unused = pd.DataFrame()
stage2_unused = pd.DataFrame()
stage3_unused = pd.DataFrame()

# 跟踪所有已用于训练的数据索引
train_indices = set(train_df_current.index)

# ========== 阶段0：预训练模型在全部未训练数据上的表现（基准） ==========
print("\n阶段0：预训练模型在全部未训练数据上的表现（基准）")
phase0_dir = os.path.join(OUTPUT_DIR, 'phase_0')
os.makedirs(phase0_dir, exist_ok=True)

# 保存当前scaler
joblib.dump(x_scaler, os.path.join(phase0_dir, 'x_scaler.pkl'))
joblib.dump(y_scaler, os.path.join(phase0_dir, 'y_scaler.pkl'))
model.save_weights(os.path.join(phase0_dir, 'weights_phase0.weights.h5'))

# 阶段0测试集：所有不在训练集中的数据（包括所有候选参数）
test_df_phase0 = df[~df.index.isin(train_indices)].copy()
X_test_phase0 = np.vstack(test_df_phase0['scaled_X'].values)
y_test_phase0 = np.vstack(test_df_phase0['scaled_y'].values)
print(f"阶段0测试集大小: {len(test_df_phase0)}")

y_pred_phase0 = model.predict(X_test_phase0, verbose=0)
y_pred_phase0_denorm = y_scaler.inverse_transform(y_pred_phase0)
y_test_phase0_denorm = y_scaler.inverse_transform(y_test_phase0)
plot_scatter(y_test_phase0_denorm, y_pred_phase0_denorm, target_columns, os.path.join(phase0_dir, 'scatter_phase0'))
print("阶段0散点图已保存。")

metrics0 = calc_metrics(y_test_phase0_denorm, y_pred_phase0_denorm, '阶段0', return_dict=True)
stages_info.append({
    '阶段': '阶段0 (预训练)',
    '模型类型': '预训练（基准）',
    'R2': metrics0['R2'],
    'MSE': metrics0['MSE'],
    'RMSE': metrics0['RMSE'],
    'MAE': metrics0['MAE'],
    '训练集大小': len(train_df_current),
    '测试集大小': len(test_df_phase0),
    '新引入数据量': 0
})

# ========== 阶段1：扩展D ==========
print("\n" + "=" * 60)
print("阶段1：扩展 D（引入 D=80,100，H和S保持源域范围）")
print("=" * 60)

if len(stage1_candidates_df) > 0:
    stage1_used, stage1_unused = sample_by_param_group(stage1_candidates_df, MIGRATION_FRACTION, random_state=42)
    print(f"新扩展D实际使用的数据量: {len(stage1_used)}")
    print(f"新扩展D未使用的数据量: {len(stage1_unused)}")

    # 更新训练集索引跟踪
    train_indices.update(stage1_used.index)
else:
    stage1_used = pd.DataFrame()
    stage1_unused = pd.DataFrame()
    print("新扩展D无候选数据，跳过阶段1")

# 更新训练集
train_df_current = pd.concat([train_df_current, stage1_used], ignore_index=True)
print(f"阶段1训练集大小: {len(train_df_current)}")

if len(stage1_used) > 0:
    # 阶段1测试集 = 阶段1未使用的数据
    test_df_phase1 = stage1_unused.copy()
    X_test_phase1 = np.vstack(test_df_phase1['scaled_X'].values)
    y_test_phase1 = np.vstack(test_df_phase1['scaled_y'].values)
    y_test_phase1_denorm = y_scaler.inverse_transform(y_test_phase1)
    print(f"阶段1测试集大小: {len(test_df_phase1)}")

    # 未迁移模型性能（阶段0模型）
    y_pred_before = model.predict(X_test_phase1, verbose=0)
    y_pred_before_denorm = y_scaler.inverse_transform(y_pred_before)
    metrics_before = calc_metrics(y_test_phase1_denorm, y_pred_before_denorm, '阶段1 (未迁移)', return_dict=True)

    stages_info.append({
        '阶段': '阶段1 (扩展D)',
        '模型类型': '未迁移（阶段0模型）',
        'R2': metrics_before['R2'],
        'MSE': metrics_before['MSE'],
        'RMSE': metrics_before['RMSE'],
        'MAE': metrics_before['MAE'],
        '训练集大小': len(train_df_current) - len(stage1_used),
        '测试集大小': len(test_df_phase1),
        '新引入数据量': 0
    })

    # 准备训练数据
    X_train1 = np.vstack(train_df_current['scaled_X'].values)
    y_train1 = np.vstack(train_df_current['scaled_y'].values)
    X_train1, X_val1, y_train1, y_val1 = train_test_split(
        X_train1, y_train1, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {len(X_train1)}，验证集大小: {len(X_val1)}")

    # 微调
    history1 = train_phase(model, X_train1, y_train1, X_val1, y_val1,
                           epochs=MAX_EPOCHS_PER_STAGE, lr=INITIAL_LR,
                           lambda_phys=LAMBDA_PHYSICS, phase_name="阶段1",
                           y_scaler=y_scaler, x_scaler=x_scaler)

    phase1_dir = os.path.join(OUTPUT_DIR, 'phase_1')
    os.makedirs(phase1_dir, exist_ok=True)
    joblib.dump(x_scaler, os.path.join(phase1_dir, 'x_scaler.pkl'))
    joblib.dump(y_scaler, os.path.join(phase1_dir, 'y_scaler.pkl'))
    model.save_weights(os.path.join(phase1_dir, 'weights_phase1.weights.h5'))

    # 迁移后模型性能
    y_pred_after = model.predict(X_test_phase1, verbose=0)
    y_pred_after_denorm = y_scaler.inverse_transform(y_pred_after)
    metrics_after = calc_metrics(y_test_phase1_denorm, y_pred_after_denorm, '阶段1 (迁移后)', return_dict=True)

    plot_scatter(y_test_phase1_denorm, y_pred_after_denorm, target_columns,
                 os.path.join(phase1_dir, 'scatter_phase1_after'))

    stages_info.append({
        '阶段': '阶段1 (扩展D)',
        '模型类型': '迁移后',
        'R2': metrics_after['R2'],
        'MSE': metrics_after['MSE'],
        'RMSE': metrics_after['RMSE'],
        'MAE': metrics_after['MAE'],
        '训练集大小': len(train_df_current),
        '测试集大小': len(test_df_phase1),
        '新引入数据量': len(stage1_used)
    })

    print(f"阶段1模型和散点图已保存至 {phase1_dir}")
else:
    print("阶段1无有效数据，跳过训练。")

# ========== 阶段2：扩展H ==========
print("\n" + "=" * 60)
print("阶段2：扩展 H（引入 H=160,200，S保持源域范围）")
print("=" * 60)

# 从候选数据中排除已经使用的（避免重复使用）
stage2_candidates_remaining = stage2_candidates_df[~stage2_candidates_df.index.isin(train_indices)].copy()
print(f"新扩展H剩余候选数据量: {len(stage2_candidates_remaining)}")

if len(stage2_candidates_remaining) > 0:
    stage2_used, stage2_unused = sample_by_param_group(stage2_candidates_remaining, MIGRATION_FRACTION, random_state=42)
    print(f"新扩展H实际使用的数据量: {len(stage2_used)}")
    print(f"新扩展H未使用的数据量: {len(stage2_unused)}")

    # 更新训练集索引跟踪
    train_indices.update(stage2_used.index)
else:
    stage2_used = pd.DataFrame()
    stage2_unused = pd.DataFrame()
    print("新扩展H无剩余候选数据，跳过阶段2")

# 更新训练集
train_df_current = pd.concat([train_df_current, stage2_used], ignore_index=True)
print(f"阶段2训练集大小: {len(train_df_current)}")

if len(stage2_used) > 0:
    # 阶段2测试集 = 阶段1未使用数据 + 阶段2未使用数据
    test_df_phase2 = pd.concat([stage1_unused, stage2_unused], ignore_index=True)
    X_test_phase2 = np.vstack(test_df_phase2['scaled_X'].values)
    y_test_phase2 = np.vstack(test_df_phase2['scaled_y'].values)
    y_test_phase2_denorm = y_scaler.inverse_transform(y_test_phase2)
    print(f"阶段2测试集大小: {len(test_df_phase2)}")

    # 未迁移模型性能（阶段1迁移后模型）
    y_pred_before = model.predict(X_test_phase2, verbose=0)
    y_pred_before_denorm = y_scaler.inverse_transform(y_pred_before)
    metrics_before = calc_metrics(y_test_phase2_denorm, y_pred_before_denorm, '阶段2 (未迁移)', return_dict=True)

    stages_info.append({
        '阶段': '阶段2 (扩展H)',
        '模型类型': '未迁移（阶段1模型）',
        'R2': metrics_before['R2'],
        'MSE': metrics_before['MSE'],
        'RMSE': metrics_before['RMSE'],
        'MAE': metrics_before['MAE'],
        '训练集大小': len(train_df_current) - len(stage2_used),
        '测试集大小': len(test_df_phase2),
        '新引入数据量': 0
    })

    # 准备训练数据
    X_train2 = np.vstack(train_df_current['scaled_X'].values)
    y_train2 = np.vstack(train_df_current['scaled_y'].values)
    X_train2, X_val2, y_train2, y_val2 = train_test_split(
        X_train2, y_train2, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {len(X_train2)}，验证集大小: {len(X_val2)}")

    # 微调
    history2 = train_phase(model, X_train2, y_train2, X_val2, y_val2,
                           epochs=MAX_EPOCHS_PER_STAGE, lr=INITIAL_LR * 0.5,
                           lambda_phys=LAMBDA_PHYSICS, phase_name="阶段2",
                           y_scaler=y_scaler, x_scaler=x_scaler)

    phase2_dir = os.path.join(OUTPUT_DIR, 'phase_2')
    os.makedirs(phase2_dir, exist_ok=True)
    joblib.dump(x_scaler, os.path.join(phase2_dir, 'x_scaler.pkl'))
    joblib.dump(y_scaler, os.path.join(phase2_dir, 'y_scaler.pkl'))
    model.save_weights(os.path.join(phase2_dir, 'weights_phase2.weights.h5'))

    # 迁移后模型性能
    y_pred_after = model.predict(X_test_phase2, verbose=0)
    y_pred_after_denorm = y_scaler.inverse_transform(y_pred_after)
    metrics_after = calc_metrics(y_test_phase2_denorm, y_pred_after_denorm, '阶段2 (迁移后)', return_dict=True)

    plot_scatter(y_test_phase2_denorm, y_pred_after_denorm, target_columns,
                 os.path.join(phase2_dir, 'scatter_phase2_after'))

    stages_info.append({
        '阶段': '阶段2 (扩展H)',
        '模型类型': '迁移后',
        'R2': metrics_after['R2'],
        'MSE': metrics_after['MSE'],
        'RMSE': metrics_after['RMSE'],
        'MAE': metrics_after['MAE'],
        '训练集大小': len(train_df_current),
        '测试集大小': len(test_df_phase2),
        '新引入数据量': len(stage2_used)
    })

    print(f"阶段2模型和散点图已保存至 {phase2_dir}")
else:
    print("阶段2无有效数据，跳过训练。")

# ========== 阶段3：扩展S ==========
print("\n" + "=" * 60)
print("阶段3：扩展 S（引入 S=0.75，使用所有D和H）")
print("=" * 60)

stage3_candidates_remaining = stage3_candidates_df[~stage3_candidates_df.index.isin(train_indices)].copy()
print(f"新扩展S剩余候选数据量: {len(stage3_candidates_remaining)}")

if len(stage3_candidates_remaining) > 0:
    stage3_used, stage3_unused = sample_by_param_group(stage3_candidates_remaining, MIGRATION_FRACTION, random_state=42)
    print(f"新扩展S实际使用的数据量: {len(stage3_used)}")
    print(f"新扩展S未使用的数据量: {len(stage3_unused)}")

    # 更新训练集索引跟踪
    train_indices.update(stage3_used.index)
else:
    stage3_used = pd.DataFrame()
    stage3_unused = pd.DataFrame()
    print("新扩展S无剩余候选数据，跳过阶段3")

# 更新训练集
train_df_current = pd.concat([train_df_current, stage3_used], ignore_index=True)
print(f"阶段3训练集大小: {len(train_df_current)}")

if len(stage3_used) > 0:
    # 阶段3测试集 = 所有未使用的数据（stage1_unused + stage2_unused + stage3_unused）
    test_df_phase3 = pd.concat([stage1_unused, stage2_unused, stage3_unused], ignore_index=True)
    X_test_phase3 = np.vstack(test_df_phase3['scaled_X'].values)
    y_test_phase3 = np.vstack(test_df_phase3['scaled_y'].values)
    y_test_phase3_denorm = y_scaler.inverse_transform(y_test_phase3)
    print(f"阶段3测试集大小: {len(test_df_phase3)}")

    # 未迁移模型性能（阶段2迁移后模型）
    y_pred_before = model.predict(X_test_phase3, verbose=0)
    y_pred_before_denorm = y_scaler.inverse_transform(y_pred_before)
    metrics_before = calc_metrics(y_test_phase3_denorm, y_pred_before_denorm, '阶段3 (未迁移)', return_dict=True)

    stages_info.append({
        '阶段': '阶段3 (扩展S)',
        '模型类型': '未迁移（阶段2模型）',
        'R2': metrics_before['R2'],
        'MSE': metrics_before['MSE'],
        'RMSE': metrics_before['RMSE'],
        'MAE': metrics_before['MAE'],
        '训练集大小': len(train_df_current) - len(stage3_used),
        '测试集大小': len(test_df_phase3),
        '新引入数据量': 0
    })

    # 准备训练数据
    X_train3 = np.vstack(train_df_current['scaled_X'].values)
    y_train3 = np.vstack(train_df_current['scaled_y'].values)
    X_train3, X_val3, y_train3, y_val3 = train_test_split(
        X_train3, y_train3, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {len(X_train3)}，验证集大小: {len(X_val3)}")

    # 微调
    history3 = train_phase(model, X_train3, y_train3, X_val3, y_val3,
                           epochs=MAX_EPOCHS_PER_STAGE, lr=INITIAL_LR * 0.2,
                           lambda_phys=LAMBDA_PHYSICS, phase_name="阶段3",
                           y_scaler=y_scaler, x_scaler=x_scaler)

    phase3_dir = os.path.join(OUTPUT_DIR, 'phase_3')
    os.makedirs(phase3_dir, exist_ok=True)
    joblib.dump(x_scaler, os.path.join(phase3_dir, 'x_scaler.pkl'))
    joblib.dump(y_scaler, os.path.join(phase3_dir, 'y_scaler.pkl'))
    model.save_weights(os.path.join(phase3_dir, 'weights_phase3.weights.h5'))

    # 迁移后模型性能
    y_pred_after = model.predict(X_test_phase3, verbose=0)
    y_pred_after_denorm = y_scaler.inverse_transform(y_pred_after)
    metrics_after = calc_metrics(y_test_phase3_denorm, y_pred_after_denorm, '阶段3 (迁移后)', return_dict=True)

    plot_scatter(y_test_phase3_denorm, y_pred_after_denorm, target_columns,
                 os.path.join(phase3_dir, 'scatter_phase3_after'))

    stages_info.append({
        '阶段': '阶段3 (扩展S)',
        '模型类型': '迁移后',
        'R2': metrics_after['R2'],
        'MSE': metrics_after['MSE'],
        'RMSE': metrics_after['RMSE'],
        'MAE': metrics_after['MAE'],
        '训练集大小': len(train_df_current),
        '测试集大小': len(test_df_phase3),
        '新引入数据量': len(stage3_used)
    })

    print(f"阶段3模型和散点图已保存至 {phase3_dir}")
else:
    print("阶段3无有效数据，跳过训练。")

# ===============================================
# 6. 最终评估（使用所有未使用的数据作为测试集）
# ===============================================
print("\n" + "=" * 60)
print("6. 最终评估（使用所有未使用的数据作为测试集）")
print("=" * 60)

# 修复：确保最终测试集包含所有未使用的数据（包括未分配的数据）
final_test_df = pd.concat([stage1_unused, stage2_unused, stage3_unused, unassigned_df], ignore_index=True)

# 去重（以防万一）
final_test_df = final_test_df.drop_duplicates(subset=df.columns.difference(['scaled_X', 'scaled_y', 'param_group']))

X_final_test = np.vstack(final_test_df['scaled_X'].values)
y_final_test = np.vstack(final_test_df['scaled_y'].values)
print(f"最终测试集大小: {len(final_test_df)}")

# 验证数据完整性
print(f"\n数据完整性验证:")
print(f"训练集大小: {len(train_df_current)}")
print(f"测试集大小: {len(final_test_df)}")
print(f"总和: {len(train_df_current) + len(final_test_df)}")
print(f"原始数据总量: {len(df)}")
print(f"是否相等: {len(train_df_current) + len(final_test_df) == len(df)}")

# 检查是否有数据同时存在于训练集和测试集
train_test_overlap = set(train_df_current.index).intersection(set(final_test_df.index))
print(f"训练集和测试集交集大小: {len(train_test_overlap)}")

y_pred_final = model.predict(X_final_test, verbose=0)
y_pred_final_denorm = y_scaler.inverse_transform(y_pred_final)
y_final_test_denorm = y_scaler.inverse_transform(y_final_test)

print("\n最终模型在测试集上的整体表现：")
final_metrics = calc_metrics(y_final_test_denorm, y_pred_final_denorm, 'Final Test', return_dict=True)

stages_info.append({
    '阶段': '最终模型',
    '模型类型': '阶段3迁移后',
    'R2': final_metrics['R2'],
    'MSE': final_metrics['MSE'],
    'RMSE': final_metrics['RMSE'],
    'MAE': final_metrics['MAE'],
    '训练集大小': len(train_df_current),
    '测试集大小': len(final_test_df),
    '新引入数据量': 0
})

# ===============================================
# 7. 保存汇总结果和性能对比图
# ===============================================
print("\n" + "=" * 60)
print("7. 保存各阶段汇总信息和性能对比图")
print("=" * 60)

stages_df = pd.DataFrame(stages_info)
stages_df.to_excel(os.path.join(OUTPUT_DIR, 'stages_summary.xlsx'), index=False)
print(f"各阶段汇总信息已保存至 {os.path.join(OUTPUT_DIR, 'stages_summary.xlsx')}")

# 绘制性能对比图（未迁移 vs 迁移后）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

stages_with_comparison = ['阶段1 (扩展D)', '阶段2 (扩展H)', '阶段3 (扩展S)']
r2_before = []
r2_after = []
rmse_before = []
rmse_after = []

for stage in stages_with_comparison:
    stage_data = stages_df[stages_df['阶段'] == stage]
    before = stage_data[stage_data['模型类型'].str.contains('未迁移')]
    after = stage_data[stage_data['模型类型'] == '迁移后']
    if not before.empty and not after.empty:
        r2_before.append(before['R2'].values[0])
        r2_after.append(after['R2'].values[0])
        rmse_before.append(before['RMSE'].values[0])
        rmse_after.append(after['RMSE'].values[0])
    else:
        r2_before.append(np.nan)
        r2_after.append(np.nan)
        rmse_before.append(np.nan)
        rmse_after.append(np.nan)

x = np.arange(len(stages_with_comparison))
width = 0.35
ax1.bar(x - width / 2, r2_before, width, label='未迁移', color='lightblue')
ax1.bar(x + width / 2, r2_after, width, label='迁移后', color='steelblue')
ax1.set_xlabel('阶段')
ax1.set_ylabel('R²')
ax1.set_title('R² 对比')
ax1.set_xticks(x)
ax1.set_xticklabels(stages_with_comparison, rotation=45, ha='right')
ax1.legend()
ax1.set_ylim(0, 1)

ax2.bar(x - width / 2, rmse_before, width, label='未迁移', color='lightcoral')
ax2.bar(x + width / 2, rmse_after, width, label='迁移后', color='darkred')
ax2.set_xlabel('阶段')
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE 对比')
ax2.set_xticks(x)
ax2.set_xticklabels(stages_with_comparison, rotation=45, ha='right')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'performance_comparison.png'), dpi=300)
plt.close()
print(f"性能对比图已保存至 {os.path.join(OUTPUT_DIR, 'performance_comparison.png')}")

# ===============================================
# 8. 保存最终测试预测和指标
# ===============================================
print("\n" + "=" * 60)
print("8. 保存最终测试结果")
print("=" * 60)

results_test = pd.DataFrame({
    'Ce_true': y_final_test_denorm[:, 0], 'Ce_pred': y_pred_final_denorm[:, 0],
    'Ci_true': y_final_test_denorm[:, 1], 'Ci_pred': y_pred_final_denorm[:, 1],
    'Ei_true': y_final_test_denorm[:, 2], 'Ei_pred': y_pred_final_denorm[:, 2],
    'Qe_true': y_final_test_denorm[:, 3], 'Qe_pred': y_pred_final_denorm[:, 3]
})
for col in feature_columns:
    results_test[col] = final_test_df[col].values
results_test.to_excel(os.path.join(OUTPUT_DIR, 'test_predictions.xlsx'), index=False)

metrics_df = pd.DataFrame([final_metrics])
metrics_df.to_excel(os.path.join(OUTPUT_DIR, 'final_metrics.xlsx'), index=False)

plot_scatter(y_final_test_denorm, y_pred_final_denorm, target_columns, os.path.join(OUTPUT_DIR, 'final_scatter'))

print(f"所有结果已保存到目录: {OUTPUT_DIR}")
print("迁移学习完成。")