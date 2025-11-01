#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Completion Rate vs Formation Keep Rate Visualization
Performance comparison for 1-4 Follower configurations
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set font for better display
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False

# 数据（来自测试结果）
follower_counts = [1, 2, 3, 4]
task_completion_rates = [72, 97, 99, 92]  # 任务完成率 (%)
formation_keep_rates = [21.90, 5.05, 2.60, 1.00]  # 编队保持率 (%)

# 创建图表
fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')

# Plot Task Completion Rate (Left Y-axis)
color1 = '#2ecc71'  # Green
ax1.set_xlabel('Number of Followers', fontsize=14, fontweight='bold')
ax1.set_ylabel('Task Completion Rate (%)', color=color1, fontsize=14, fontweight='bold')
line1 = ax1.plot(follower_counts, task_completion_rates, 
                 color=color1, marker='o', markersize=12, 
                 linewidth=3, label='Task Completion Rate', 
                 markeredgewidth=2, markeredgecolor='white')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.set_ylim([0, 105])
ax1.grid(True, alpha=0.3)

# Add value annotations on data points
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

# Create right Y-axis for Formation Keep Rate
ax2 = ax1.twinx()
color2 = '#e74c3c'  # Red
ax2.set_ylabel('Formation Keep Rate (%)', color=color2, fontsize=14, fontweight='bold')
line2 = ax2.plot(follower_counts, formation_keep_rates, 
                 color=color2, marker='s', markersize=12, 
                 linewidth=3, label='Formation Keep Rate',
                 markeredgewidth=2, markeredgecolor='white')
ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
ax2.set_ylim([0, 25])

# Add value annotations on data points
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

# Add title
plt.title('Task Completion Rate vs Formation Keep Rate\n(1-4 Follower Configuration Comparison)', 
          fontsize=16, fontweight='bold', pad=20)

# Merge legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper right', fontsize=12, 
          framealpha=0.95, edgecolor='black', 
          fancybox=True, shadow=True)

# Add grid
ax1.set_axisbelow(True)

# Set X-axis ticks
ax1.set_xticks(follower_counts)
ax1.set_xticklabels([f'{n}F' for n in follower_counts])

# Add performance contradiction annotation
ax1.text(0.02, 0.98, 
         'Performance Paradox:\nTask completion: 92%\nFormation rate: only 1%',
         transform=ax1.transAxes,
         fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         color='#e74c3c',
         fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save figure
output_path = 'formation_rate_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Chart saved to: {output_path}")

# Display figure
plt.show()

# ============================================
# Second Chart: Bar Chart Comparison
# ============================================

fig2, ax = plt.subplots(figsize=(12, 7), dpi=300)

x = np.arange(len(follower_counts))
width = 0.35

# Draw bar chart
bars1 = ax.bar(x - width/2, task_completion_rates, width, 
               label='Task Completion Rate', color='#2ecc71', 
               edgecolor='black', linewidth=1.5, alpha=0.85)
bars2 = ax.bar(x + width/2, formation_keep_rates, width, 
               label='Formation Keep Rate', color='#e74c3c', 
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels
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

# Set labels and title
ax.set_xlabel('Number of Followers', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
ax.set_title('Task Completion Rate vs Formation Keep Rate\n(Bar Chart Comparison)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f'{n}F' for n in follower_counts], fontsize=12)
ax.legend(loc='upper right', fontsize=12, framealpha=0.95, 
         edgecolor='black', fancybox=True, shadow=True)

# Add grid
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Set Y-axis range
ax.set_ylim([0, 110])

# Add trend line annotation
ax.annotate('', xy=(3, 1), xytext=(0, 21.9),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', 
                          lw=2, linestyle='--', alpha=0.7))
ax.text(1.5, 15, 'Sharp Decline', fontsize=11, 
       color='#e74c3c', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust layout
plt.tight_layout()

# Save second chart
output_path2 = 'formation_rate_bar_comparison.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Chart saved to: {output_path2}")

# Display figure
plt.show()

print("\n" + "="*60)
print("📊 Visualization completed! Generated two charts:")
print(f"   1. {output_path} - Dual Y-axis Line Chart")
print(f"   2. {output_path2} - Bar Chart Comparison")
print("="*60)

