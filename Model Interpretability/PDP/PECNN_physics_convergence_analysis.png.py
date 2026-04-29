import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ===============================================
# 物理损失权重（仅用于饼图显示，与训练时一致）
# ===============================================
PHYSICS_WEIGHT = 0.5   # 实际训练使用的λ

# ===============================================
# 全局字体配置 - 统一使用第一段的字体设置（较大字体）
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
# 加载历史记录数据
# ===============================================
history = joblib.load('train_history_pecnn.pkl')

# 确保所有必要的键都存在
default_history = {
    'loss': [], 'mse_loss': [], 'physics_loss': [], 'physics_abs_error': [],
    'val_loss': [], 'val_mse_loss': [], 'val_physics_loss': [], 'val_physics_abs_error': []
}
for key in default_history:
    if key not in history:
        history[key] = default_history[key]

# ===============================================
# 1. 绘制组图 PECNN_physics_convergence_analysis.png
#    所有子图均为线性坐标，且增大子图间距
# ===============================================
print('\n【绘制物理约束收敛分析图（全部线性坐标，大间距）...】')

# 增大 hspace 和 wspace 以加大子图间隔
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.5)   # 原为0.3，现增大到0.5

# 子图1: Loss Decomposition (Training) - 线性坐标双轴
ax1 = fig.add_subplot(gs[0, 0])
ax1_twin = ax1.twinx()
line1 = ax1.plot(history['mse_loss'], 'b-', linewidth=2.5, label='MSE Loss')[0]
line2 = ax1_twin.plot(history['physics_loss'], 'r--', linewidth=2.5, label='Physics Loss')[0]
ax1.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax1.set_ylabel('MSE Loss', color='b', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax1_twin.set_ylabel('Physics Loss (λ=0.5)', color='r', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='b', labelsize=TICK_FONT_SIZE)
ax1_twin.tick_params(axis='y', labelcolor='r', labelsize=TICK_FONT_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
ax1.set_title('Loss Decomposition (Training)', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both', linestyle='--')
ax1.legend([line1, line2], ['MSE Loss', 'Physics Loss'], loc='upper right', fontsize=LEGEND_FONT_SIZE)

# 子图2: Loss Decomposition (Validation) - 线性坐标双轴
ax2 = fig.add_subplot(gs[0, 1])
ax2_twin = ax2.twinx()
line3 = ax2.plot(history['val_mse_loss'], 'b-', linewidth=2.5, label='Val MSE')[0]
line4 = ax2_twin.plot(history['val_physics_loss'], 'r--', linewidth=2.5, label='Val Physics')[0]
ax2.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax2.set_ylabel('Val MSE Loss', color='b', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax2_twin.set_ylabel('Val Physics Loss', color='r', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='b', labelsize=TICK_FONT_SIZE)
ax2_twin.tick_params(axis='y', labelcolor='r', labelsize=TICK_FONT_SIZE)
ax2.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
ax2.set_title('Loss Decomposition (Validation)', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both', linestyle='--')
ax2.legend([line3, line4], ['Val MSE', 'Val Physics'], loc='upper right', fontsize=LEGEND_FONT_SIZE)

# 子图3: Physics Constraint Satisfaction (Absolute Error) - 线性坐标
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(history['physics_abs_error'], 'g-', linewidth=2.5, label='Train', alpha=0.8)
ax3.plot(history['val_physics_abs_error'], 'orange', linewidth=2.5, linestyle='--', label='Validation', alpha=0.8)
train_median = np.median(history['physics_abs_error']) if history['physics_abs_error'] else 1e-3
ax3.axhline(y=train_median, color='r', linestyle=':', linewidth=2, label=f'Median: {train_median:.2e}')
ax3.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax3.set_ylabel('Mean Absolute Error of Qe', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax3.tick_params(labelsize=TICK_FONT_SIZE)
ax3.set_title('Physics Constraint Satisfaction\n(Absolute Error)', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
ax3.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
ax3.grid(True, alpha=0.3)

# 子图4: Total Physics-Informed Loss - 线性坐标
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(history['loss'], 'b-', linewidth=2.5, label='Train Loss', alpha=0.8)
ax4.plot(history['val_loss'], 'orange', linewidth=2.5, linestyle='--', label='Val Loss', alpha=0.8)
best_epoch = np.argmin(history['val_loss'])
ax4.axvline(x=best_epoch, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Best epoch: {best_epoch}')
ax4.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax4.set_ylabel('Total Loss', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax4.tick_params(labelsize=TICK_FONT_SIZE)
ax4.set_title('Total Physics-Informed Loss', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
ax4.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
ax4.grid(True, alpha=0.3, linestyle='--')

# 子图5: Final Epoch Loss Composition - 饼图（不变）
ax5 = fig.add_subplot(gs[1, 1])
final_mse = history['mse_loss'][-1] if history['mse_loss'] else 0
final_physics_weighted = (history['physics_loss'][-1] * PHYSICS_WEIGHT) if history['physics_loss'] else 0
sizes = [final_mse, final_physics_weighted]
colors_pie = ['#1f77b4', '#ff7f0e']
explode = (0.05, 0.05)
wedges, texts, autotexts = ax5.pie(sizes, explode=explode, colors=colors_pie, autopct='%1.1f%%',
                                   shadow=True, startangle=90,
                                   textprops={'fontsize': ANNOT_FONT_SIZE, 'fontweight': 'bold'})
'''ax5.legend(wedges, ['MSE Component', f'Physics Component (weighted, λ={PHYSICS_WEIGHT})'],
           title="Final Loss Composition", loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1),
           fontsize=LEGEND_FONT_SIZE)'''
ax5.set_title(f'Final Epoch Loss Composition (λ={PHYSICS_WEIGHT})', fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')

# 子图6: Overfitting Detection - 线性坐标
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
print('Saved: PECNN_physics_convergence_analysis.png (all linear axes, larger spacing)')

# ===============================================
# 2. 绘制单图（所有单图也改为线性坐标）
# ===============================================
print('\n【绘制单图（全部线性坐标）...】')

# 单图1: 总损失曲线（线性坐标）
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

# 单图2: MSE Loss分解（线性坐标）
plt.figure(figsize=(8, 5))
plt.plot(history['mse_loss'], 'b-', linewidth=2.5, label='Train MSE')
plt.plot(history['val_mse_loss'], 'orange', linewidth=2.5, linestyle='--', label='Val MSE')
plt.xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
plt.ylabel('MSE Loss', fontsize=LABEL_FONT_SIZE)
plt.title('MSE Loss Decomposition', fontsize=TITLE_FONT_SIZE, fontweight='bold')
plt.legend(fontsize=LEGEND_FONT_SIZE)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('PECNN_mse_loss_single.png', dpi=600)
print('Saved: PECNN_mse_loss_single.png')
plt.close()

# 单图3: Physics Loss分解（线性坐标）
plt.figure(figsize=(8, 5))
plt.plot(history['physics_loss'], 'r-', linewidth=2.5, label='Train Physics')
plt.plot(history['val_physics_loss'], 'orange', linewidth=2.5, linestyle='--', label='Val Physics')
plt.xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
plt.ylabel('Physics Loss', fontsize=LABEL_FONT_SIZE)
plt.title('Physics Loss Decomposition', fontsize=TITLE_FONT_SIZE, fontweight='bold')
plt.legend(fontsize=LEGEND_FONT_SIZE)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('PECNN_physics_loss_single.png', dpi=600)
print('Saved: PECNN_physics_loss_single.png')
plt.close()

# 单图4: 物理约束绝对误差（线性坐标）
plt.figure(figsize=(8, 5))
plt.plot(history['physics_abs_error'], 'g-', linewidth=2.5, label='Train', alpha=0.8)
plt.plot(history['val_physics_abs_error'], 'orange', linewidth=2.5, linestyle='--', label='Validation', alpha=0.8)
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

print('\n所有图已按线性坐标、大间距格式生成完毕。')