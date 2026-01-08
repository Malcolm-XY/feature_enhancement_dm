# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 01:33:53 2025

@author: 18307
"""

# -*- coding: utf-8 -*-
"""
Generate a simulated scalp potential and plot a 64-channel (10–20 compatible) topomap with MNE.
"""

import numpy as np
import mne
import matplotlib.pyplot as plt

# ----------------------------
# 1) 准备 64 通道头皮电极布局（BioSemi64 与 10–20 扩展兼容）
# ----------------------------
montage = mne.channels.make_standard_montage('biosemi64')  # 64ch
ch_names = montage.ch_names                                # 64 个通道名
info = mne.create_info(ch_names=ch_names, sfreq=256., ch_types='eeg')
info.set_montage(montage)

# ----------------------------
# 2) 生成“可控”的模拟电位（两个高斯源 + 少量噪声）
#    思路：在传感器空间里，以指定电极为中心放两个空间高斯，再叠加
# ----------------------------
rng = np.random.default_rng(42)

# 选两个“源”中心（可改成你想看的电极，如 'Cz','Pz','Fz','C3','C4' 等）
centers = ['C3', 'Pz']   # 一个偏中央-左，另一个枕区中线
sigmas = [0.06, 0.07]    # 空间扩散尺度（单位约为米，0.05~0.10 常见）
sigmas = [0.03, 0.03]    # 空间扩散尺度（单位约为米，0.05~0.10 常见）
weights = [1.0, -0.7]    # 两个源的极性与幅度（正/负峰），可自行调整

# 取三维坐标（单位米）
ch_pos = montage.get_positions()['ch_pos']  # dict: name -> xyz
pos_mat = np.array([ch_pos[name] for name in ch_names])  # (64, 3)

# 根据中心电极坐标构造高斯场并叠加
data = np.zeros(len(ch_names), dtype=float)
for c_name, sigma, w in zip(centers, sigmas, weights):
    c = ch_pos[c_name]                                   # 中心坐标 (3,)
    d2 = np.sum((pos_mat - c)**2, axis=1)               # 各电极到中心的平方距离
    data += w * np.exp(-d2 / (2 * sigma**2))            # 高斯分布

# 加一点小噪声让图更自然
data += 0.05 * rng.standard_normal(len(ch_names))

# 归一化到便于显示的量级（可选）
data -= data.mean()
data /= (np.abs(data).max() + 1e-12)
data *= 10e-6  # 转成 ~μV 级别（仅为可视化单位感）

# ----------------------------
# 3) 绘制 Topomap
# ----------------------------
fig, ax = plt.subplots(figsize=(5, 5))
im, cn = mne.viz.plot_topomap(
    data, info,
    axes=ax,
    cmap='RdBu_r',       # 红蓝反转常用于电位（红=正，蓝=负）
    contours=4,          # 等高线条数
    sensors=True,        # 显示电极位置
    outlines='head',     # 画头型轮廓
    sphere=None,         # 用默认头模（由 montage 推断）
    show=False
)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Potential (approx. μV)', rotation=270, labelpad=12)
ax.set_title('Simulated Scalp Topography (64-ch BioSemi / 10–20 compatible)')

plt.tight_layout()
plt.show()

# ----------------------------
# 4) 进阶：多图演示（不同时间/条件）
#    若想要一排图：传入不同 data 数组到 plot_topomap 即可
# ----------------------------
# 示例：再做一个不同中心/极性的图（可注释掉）
if False:
    centers2 = ['Fz', 'C4']
    sigmas2 = [0.05, 0.06]
    weights2 = [0.8, 0.6]
    data2 = np.zeros(len(ch_names))
    for c_name, sigma, w in zip(centers2, sigmas2, weights2):
        c = ch_pos[c_name]
        d2 = np.sum((pos_mat - c)**2, axis=1)
        data2 += w * np.exp(-d2 / (2 * sigma**2))
    data2 = (data2 - data2.mean()) / (np.abs(data2).max() + 1e-12) * 10e-6

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    mne.viz.plot_topomap(data, info, axes=axes[0], cmap='RdBu_r', contours=4, sensors=True, outlines='head', show=False)
    axes[0].set_title('Condition A')
    mne.viz.plot_topomap(data2, info, axes=axes[1], cmap='RdBu_r', contours=4, sensors=True, outlines='head', show=False)
    axes[1].set_title('Condition B')
    plt.tight_layout()
    plt.show()
