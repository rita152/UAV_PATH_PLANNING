#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务完成率与编队保持率可视化
展示1-4 Follower配置下的性能对比
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据（来自测试结果）
follower_counts = [1, 2, 3, 4]
task_completion_rates = [72, 97, 99, 92]  # 任务完成率 (%)
formation_keep_rates = [21.90, 5.05, 2.60, 1.00]  # 编队保持率 (%)

# 创建图表
fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')

# 绘制任务完成率（左Y轴）
color1 = '#2ecc71'  # 绿色
ax1.set_xlabel('Follower数量', fontsize=14, fontweight='bold')
ax1.set_ylabel('任务完成率 (%)', color=color1, fontsize=14, fontweight='bold')
line1 = ax1.plot(follower_counts, task_completion_rates, 
                 color=color1, marker='o', markersize=12, 
                 linewidth=3, label='任务完成率', 
                 markeredgewidth=2, markeredgecolor='white')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.set_ylim([0, 105])
ax1.grid(True, alpha=0.3)

# 在数据点上标注数值
for i, (x, y) in enumerate(zip(follower_counts, task_completion_rates)):
    ax1.annotate(f'{y}%', 
                xy=(x, y), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=11,
                fontweight='bold',
                color=color1,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor=color1, 
                         alpha=0.8))

# 创建右侧Y轴用于编队保持率
ax2 = ax1.twinx()
color2 = '#e74c3c'  # 红色
ax2.set_ylabel('编队保持率 (%)', color=color2, fontsize=14, fontweight='bold')
line2 = ax2.plot(follower_counts, formation_keep_rates, 
                 color=color2, marker='s', markersize=12, 
                 linewidth=3, label='编队保持率',
                 markeredgewidth=2, markeredgecolor='white')
ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
ax2.set_ylim([0, 25])

# 在数据点上标注数值
for i, (x, y) in enumerate(zip(follower_counts, formation_keep_rates)):
    ax2.annotate(f'{y}%', 
                xy=(x, y), 
                xytext=(0, -20),
                textcoords='offset points',
                ha='center',
                fontsize=11,
                fontweight='bold',
                color=color2,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor=color2, 
                         alpha=0.8))

# 添加标题
plt.title('任务完成率 vs 编队保持率\n(1-4 Follower配置对比)', 
          fontsize=16, fontweight='bold', pad=20)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper right', fontsize=12, 
          framealpha=0.95, edgecolor='black', 
          fancybox=True, shadow=True)

# 添加网格
ax1.set_axisbelow(True)

# 设置X轴刻度
ax1.set_xticks(follower_counts)
ax1.set_xticklabels([f'{n}F' for n in follower_counts])

# 添加性能矛盾的注释
ax1.text(0.02, 0.98, 
         '⚠️ 性能矛盾：\n任务完成率高达92%\n但编队率仅有1%',
         transform=ax1.transAxes,
         fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         color='#e74c3c',
         fontweight='bold')

# 调整布局
plt.tight_layout()

# 保存图表
output_path = 'formation_rate_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 图表已保存至: {output_path}")

# 显示图表
plt.show()

# ============================================
# 绘制第二个图表：双指标柱状图对比
# ============================================

fig2, ax = plt.subplots(figsize=(12, 7), dpi=300)

x = np.arange(len(follower_counts))
width = 0.35

# 绘制柱状图
bars1 = ax.bar(x - width/2, task_completion_rates, width, 
               label='任务完成率', color='#2ecc71', 
               edgecolor='black', linewidth=1.5, alpha=0.85)
bars2 = ax.bar(x + width/2, formation_keep_rates, width, 
               label='编队保持率', color='#e74c3c', 
               edgecolor='black', linewidth=1.5, alpha=0.85)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 设置标签和标题
ax.set_xlabel('Follower数量', fontsize=14, fontweight='bold')
ax.set_ylabel('百分比 (%)', fontsize=14, fontweight='bold')
ax.set_title('任务完成率与编队保持率对比\n(柱状图)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f'{n}F' for n in follower_counts], fontsize=12)
ax.legend(loc='upper right', fontsize=12, framealpha=0.95, 
         edgecolor='black', fancybox=True, shadow=True)

# 添加网格
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# 设置Y轴范围
ax.set_ylim([0, 110])

# 添加趋势线注释
ax.annotate('', xy=(3, 1), xytext=(0, 21.9),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', 
                          lw=2, linestyle='--', alpha=0.7))
ax.text(1.5, 15, '编队率急剧下降', fontsize=11, 
       color='#e74c3c', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 调整布局
plt.tight_layout()

# 保存第二个图表
output_path2 = 'formation_rate_bar_comparison.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 图表已保存至: {output_path2}")

# 显示图表
plt.show()

print("\n" + "="*60)
print("📊 可视化完成！生成了两张图表：")
print(f"   1. {output_path} - 双Y轴折线图")
print(f"   2. {output_path2} - 柱状图对比")
print("="*60)

