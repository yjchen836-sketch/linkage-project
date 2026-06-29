#!/usr/bin/env python3
"""
連桿機構運動分析 — Linkage Mechanism Kinematic Analysis
══════════════════════════════════════════════════════════

機構拓撲 (Topology):
  K(滑塊輸入) ──[L_KB]──► B ──[紅色剛體 | 轉軸 A]──► C
  ──[L_CD(藍)]──► D ◄──[L_ED(約束)]── E(固定)
  ──[L_DG(青)]──► G ──[綠色剛體 | 轉軸 F]──► I(輸出)

所需幾何量一覽:
  固定座標: A, E, F  (均屬固定框架 JAEF)
  桿長:     L_KB, L_AB, L_AC, L_CD, L_ED, L_DG, L_FG, L_FI
  角度:     α_BAC — B 與 C 在紅色剛體上的夾角
            α_GFI — I 在綠色剛體上相對於 FG 方向的偏角

使用方式:
  python linkage_analysis.py           ← Interactive mode (drag slider)
  python linkage_analysis.py --static  ← Static output, saves PNG
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ══════════════════════════════════════════════════════════════════════
#  ★★★  可編輯幾何參數  ★★★
# ══════════════════════════════════════════════════════════════════════

P = {
    # ── 固定框架 JAEF 各點座標 (x, y) ─────────────────────────────
    'J': np.array([10.0,  0.0]),   # 滑塊導軌右錨點
    'A': np.array([ 9.0, -3.0]),   # 紅色桿件固定轉軸
    'E': np.array([ 5.0, -1.5]),   # ED 約束桿固定端
    'F': np.array([ 4.0,  0.0]),   # 綠色桿件固定轉軸

    # ── 滑塊 K 導軌 (水平線 y = K_y) ──────────────────────────────
    'K_y'    :  0.0,
    'K_x_min': 10.5,   # K 行程左端 (輸入最小值)
    'K_x_max': 13.0,   # K 行程右端 (輸入最大值)

    # ── 桿件長度 ───────────────────────────────────────────────────
    'L_KB'  : 2.5,     # K ─ B
    'L_AB'  : 3.5,     # A ─ B  (紅色剛體)
    'L_AC'  : 2.5,     # A ─ C  (紅色剛體)
    'α_BAC' : np.radians(55),   # ∠BAC 紅色剛體內夾角 [rad]

    'L_CD'  : 5.0,     # C ─ D  (藍色連桿)
    'L_ED'  : 3.5,     # E ─ D  (約束桿)

    'L_DG'  : 4.0,     # D ─ G  (青色連桿)
    'L_FG'  : 3.5,     # F ─ G  (綠色剛體)

    'L_FI'  : 6.0,     # F ─ I  (輸出臂)
    'α_GFI' : np.radians(-40),  # ∠GFI 綠色剛體內偏角 [rad]
}

# 分支選擇 (+1 / -1 對應兩圓交點的兩個解，若姿態不對請切換符號)
SIGNS = {
    'B': 1,    # 求 B 點時
    'D': 1,    # 求 D 點時
    'G': -1,   # 求 G 點時
}


# ══════════════════════════════════════════════════════════════════════
#  幾何求解工具
# ══════════════════════════════════════════════════════════════════════

def circle_intersect(c1, r1, c2, r2, sign=1):
    """
    求兩圓的交點。
    sign=+1 取「左側」解，sign=-1 取「右側」解。
    兩圓不相交時回傳 None。
    """
    diff = c2 - c1
    d = np.hypot(diff[0], diff[1])
    if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9 or d < 1e-12:
        return None
    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h = np.sqrt(max(0.0, r1**2 - a**2))
    mid  = c1 + a * diff / d
    perp = np.array([-diff[1], diff[0]]) / d
    return mid + sign * h * perp


# ══════════════════════════════════════════════════════════════════════
#  運動學求解 — 正向運動學 K → I
# ══════════════════════════════════════════════════════════════════════

def solve(xk, p=None, s=None):
    """
    給定滑塊 x 座標 xk，依序求解各關節位置。

    求解步驟:
      ①  B  : 圓(A, L_AB) ∩ 圓(K, L_KB)
      ②  C  : 由紅色剛體旋轉角 th_AB 推算
      ③  D  : 圓(C, L_CD) ∩ 圓(E, L_ED)
      ④  G  : 圓(F, L_FG) ∩ 圓(D, L_DG)
      ⑤  I  : 綠色剛體偏角 α_GFI 推算

    回傳: 各點座標的 dict，或 None（死點/行程外）
    """
    if p is None: p = P
    if s is None: s = SIGNS

    K = np.array([xk, p['K_y']])

    # ① B
    B = circle_intersect(p['A'], p['L_AB'], K, p['L_KB'], s['B'])
    if B is None:
        return None

    # ② C  (紅色剛體旋轉)
    th_AB = np.arctan2(B[1] - p['A'][1], B[0] - p['A'][0])
    th_AC = th_AB + p['α_BAC']
    C = p['A'] + p['L_AC'] * np.array([np.cos(th_AC), np.sin(th_AC)])

    # ③ D
    D = circle_intersect(C, p['L_CD'], p['E'], p['L_ED'], s['D'])
    if D is None:
        return None

    # ④ G
    G = circle_intersect(p['F'], p['L_FG'], D, p['L_DG'], s['G'])
    if G is None:
        return None

    # ⑤ I  (綠色剛體旋轉)
    th_FG = np.arctan2(G[1] - p['F'][1], G[0] - p['F'][0])
    th_FI = th_FG + p['α_GFI']
    I = p['F'] + p['L_FI'] * np.array([np.cos(th_FI), np.sin(th_FI)])

    return dict(
        K=K, B=B, C=C, D=D, G=G, I=I,
        A=p['A'], E=p['E'], F=p['F'],
        th_AB=float(np.degrees(th_AB)),
        th_FG=float(np.degrees(th_FG)),
    )


def sweep(p=None, s=None, n=500):
    """掃描 K 的完整行程，回傳各點軌跡。"""
    if p is None: p = P
    if s is None: s = SIGNS
    xs = np.linspace(p['K_x_min'], p['K_x_max'], n)
    keys = ['K', 'B', 'C', 'D', 'G', 'I']
    traj = {k: [] for k in keys}
    kxs = []
    for xk in xs:
        r = solve(xk, p, s)
        if r is not None:
            kxs.append(xk)
            for k in keys:
                traj[k].append(r[k])
    return np.array(kxs), {k: np.array(v) for k, v in traj.items()}


# ══════════════════════════════════════════════════════════════════════
#  繪圖配色
# ══════════════════════════════════════════════════════════════════════

C_ = {
    'frame' : '#424242',
    'rail'  : '#9E9E9E',
    'red'   : '#E53935',
    'BK'    : '#AD1457',
    'blue'  : '#1565C0',
    'ED'    : '#78909C',
    'cyan'  : '#00838F',
    'green' : '#2E7D32',
    'trail' : '#FF8F00',
    'K'     : '#7B1FA2',
    'I'     : '#E65100',
    'fixed' : '#263238',
}


# ══════════════════════════════════════════════════════════════════════
#  繪製機構圖
# ══════════════════════════════════════════════════════════════════════

def draw_mechanism(ax, r, trail_I=None, p=None):
    if p is None: p = P
    ax.cla()

    # Slider rail
    ax.axhline(p['K_y'], color=C_['rail'], lw=1.2, ls='--', zorder=0, label='Rail (y=const)')

    # I point trajectory
    if trail_I is not None and len(trail_I) > 1:
        ax.plot(trail_I[:, 0], trail_I[:, 1], '-',
                color=C_['trail'], lw=2, alpha=0.55, zorder=1, label='Trajectory of I')

    # Fixed frame JAEF
    frame = np.array([p['J'], p['A'], p['E'], p['F'], p['J']])
    ax.plot(frame[:, 0], frame[:, 1], '--', color=C_['frame'],
            lw=1.2, alpha=0.45, zorder=2)
    for name, pt in [('J', p['J']), ('A', p['A']), ('E', p['E']), ('F', p['F'])]:
        ax.plot(*pt, 's', color=C_['fixed'], ms=8, zorder=7)
        ax.annotate(name, pt, textcoords='offset points', xytext=(5, 5),
                    fontsize=11, color=C_['fixed'], fontweight='bold')

    if r is None:
        ax.set_title('Dead point / out of range', color='red', fontsize=12)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
        return

    def link(ax, p1, p2, color, lw=2.5, ls='-', **kw):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], ls,
                color=color, lw=lw, zorder=3, **kw)

    # Link B-K
    link(ax, r['B'], r['K'], C_['BK'], lw=2.5, label='B-K')

    # Red rigid body (A-B, A-C, B-C)
    link(ax, r['A'], r['B'], C_['red'], lw=3.5)
    link(ax, r['A'], r['C'], C_['red'], lw=3.5)
    link(ax, r['B'], r['C'], C_['red'], lw=2.5, label='Red body A-B-C (pivot A)')

    # Blue link C-D
    link(ax, r['C'], r['D'], C_['blue'], lw=2.5, label='Blue C-D')

    # Constraint link E-D (dashed)
    link(ax, r['E'], r['D'], C_['ED'], lw=1.8, ls='--', label='Constraint E-D')

    # Cyan link D-G
    link(ax, r['D'], r['G'], C_['cyan'], lw=2.5, label='Cyan D-G')

    # Green rigid body (F-G, F-I, G-I)
    link(ax, r['F'], r['G'], C_['green'], lw=3.5)
    link(ax, r['F'], r['I'], C_['green'], lw=3.5)
    link(ax, r['G'], r['I'], C_['green'], lw=2.5, label='Green body F-G-I (pivot F)')

    # Joint markers
    joints = [
        ('K', r['K'],  C_['K'],     14, 's'),
        ('B', r['B'],  C_['red'],    9, 'o'),
        ('C', r['C'],  C_['red'],    9, 'o'),
        ('D', r['D'],  C_['blue'],   9, 'o'),
        ('G', r['G'],  C_['cyan'],   9, 'o'),
        ('I', r['I'],  C_['I'],     13, '*'),
    ]
    for name, pt, col, ms, mk in joints:
        ax.plot(*pt, mk, color=col, ms=ms, zorder=8, markeredgecolor='white', mew=0.8)
        ax.annotate(name, pt, textcoords='offset points', xytext=(6, 5),
                    fontsize=11, color=col, fontweight='bold')

    # Angle annotation
    ax.annotate(
        f"th_AB={r['th_AB']:.1f} deg\nth_FG={r['th_FG']:.1f} deg",
        xy=(0.02, 0.04), xycoords='axes fraction',
        fontsize=9, color='gray',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
    )

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('x  [length unit]', fontsize=10)
    ax.set_ylabel('y  [length unit]', fontsize=10)
    ax.set_title('Linkage Mechanism Diagram', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.8)


# ══════════════════════════════════════════════════════════════════════
#  右側關係圖
# ══════════════════════════════════════════════════════════════════════

def draw_relations(axes, kxs, traj):
    """繪製 I_x / I_y / |ΔI| 對 K_x 的關係。"""
    ax1, ax2, ax3 = axes

    Ix = traj['I'][:, 0]
    Iy = traj['I'][:, 1]
    Kx = traj['K'][:, 0]
    dI = np.hypot(np.diff(Ix, prepend=Ix[0]),
                  np.diff(Iy, prepend=Iy[0]))
    dI_cum = np.cumsum(dI)  # Cumul. arc

    for ax, data, ylabel, title, color in [
        (ax1, Ix,     'I_x',       'I_x vs K_x', C_['I']),
        (ax2, Iy,     'I_y',       'I_y vs K_x', '#1B5E20'),
        (ax3, dI_cum, 'Cumul. arc',  'I cumul. displacement vs K_x', '#6A1B9A'),
    ]:
        ax.cla()
        ax.plot(kxs, data, '-', color=color, lw=2)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('K_x', fontsize=9)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    return ax1, ax2, ax3


# ══════════════════════════════════════════════════════════════════════
#  靜態輸出模式
# ══════════════════════════════════════════════════════════════════════

def static_mode():
    """產生靜態分析圖並儲存。"""
    kxs, traj = sweep()
    if len(kxs) == 0:
        print("ERROR: No solution in full range. Adjust params or SIGNS.")
        sys.exit(1)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Linkage Mechanism: K-I Motion Analysis', fontsize=15, fontweight='bold', y=0.98)

    # ── 上方: 機構圖 + I 點軌跡 (XY 平面) ──
    ax_mech = fig.add_axes([0.04, 0.42, 0.46, 0.52])
    ax_xy   = fig.add_axes([0.55, 0.42, 0.42, 0.52])

    # mid-range pose
    xk_mid = kxs[len(kxs) // 2]
    r_mid  = solve(xk_mid)
    draw_mechanism(ax_mech, r_mid, traj['I'])

    # I 點 XY 軌跡
    ax_xy.plot(traj['I'][:, 0], traj['I'][:, 1], '-',
               color=C_['trail'], lw=2.5, label='I 點軌跡')
    ax_xy.plot(traj['K'][:, 0], np.zeros(len(kxs)), '-',
               color=C_['K'], lw=2, alpha=0.6, label='K trajectory (rail)')
    # 標注起訖點
    for idx, marker, label in [(0, '^', 'Start'), (-1, 'v', 'End')]:
        ax_xy.plot(*traj['I'][idx], marker, color=C_['I'], ms=10)
        ax_xy.annotate(label, traj['I'][idx], textcoords='offset points',
                       xytext=(6, 4), fontsize=9)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_xlabel('x', fontsize=10)
    ax_xy.set_ylabel('y', fontsize=10)
    ax_xy.set_title('Trajectory of I (XY plane)', fontsize=11, fontweight='bold')
    ax_xy.legend(fontsize=9)

    # ── 下方: 關係曲線 ──
    ax1 = fig.add_axes([0.04, 0.06, 0.28, 0.28])
    ax2 = fig.add_axes([0.38, 0.06, 0.28, 0.28])
    ax3 = fig.add_axes([0.72, 0.06, 0.24, 0.28])
    draw_relations([ax1, ax2, ax3], kxs, traj)

    # 在 I_x / I_y 圖上標注對應 K 位置
    for ax_r, data in [(ax1, traj['I'][:, 0]), (ax2, traj['I'][:, 1])]:
        ax_r.plot(xk_mid, data[len(kxs) // 2], 'o', color='red', ms=8, zorder=5,
                  label='Current')
        ax_r.legend(fontsize=8)

    plt.savefig('linkage_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: linkage_analysis.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════
#  互動模式
# ══════════════════════════════════════════════════════════════════════

def interactive_mode():
    """Interactive mode: drag slider to update."""
    kxs, traj = sweep()
    if len(kxs) == 0:
        print("ERROR: No solution in full range. Adjust params or SIGNS.")
        sys.exit(1)

    xk0 = (P['K_x_min'] + P['K_x_max']) / 2
    r0  = solve(xk0)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle('Linkage K-I Analysis  (drag slider below)', fontsize=13, fontweight='bold')

    ax_mech  = fig.add_axes([0.03, 0.17, 0.50, 0.76])
    ax_ix    = fig.add_axes([0.58, 0.60, 0.38, 0.30])
    ax_iy    = fig.add_axes([0.58, 0.17, 0.38, 0.30])
    ax_slide = fig.add_axes([0.10, 0.05, 0.80, 0.04])

    draw_mechanism(ax_mech, r0, traj['I'])

    # static background curves
    Ix = traj['I'][:, 0]
    Iy = traj['I'][:, 1]
    ax_ix.plot(kxs, Ix, '-', color=C_['I'],  lw=2, label='I_x')
    ax_iy.plot(kxs, Iy, '-', color='#2E7D32', lw=2, label='I_y')

    for ax_r, ylabel, title in [
        (ax_ix, 'I_x', 'I_x vs K_x'),
        (ax_iy, 'I_y', 'I_y vs K_x'),
    ]:
        ax_r.set_ylabel(ylabel, fontsize=9)
        ax_r.set_xlabel('K_x', fontsize=9)
        ax_r.set_title(title, fontsize=9, fontweight='bold')
        ax_r.grid(True, alpha=0.3)

    # 移動指示點
    dot_ix, = ax_ix.plot([], [], 'o', color='red',  ms=9, zorder=5)
    dot_iy, = ax_iy.plot([], [], 'o', color='red',  ms=9, zorder=5)
    vline_ix = ax_ix.axvline(xk0, color='gray', lw=1, ls=':')
    vline_iy = ax_iy.axvline(xk0, color='gray', lw=1, ls=':')

    # info text
    txt_info = fig.text(0.58, 0.94, '', fontsize=10,
                        va='top', ha='left',
                        bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    def update(xk):
        r = solve(xk)
        draw_mechanism(ax_mech, r, traj['I'])
        if r is not None:
            dot_ix.set_data([xk], [r['I'][0]])
            dot_iy.set_data([xk], [r['I'][1]])
            vline_ix.set_xdata([xk, xk])
            vline_iy.set_xdata([xk, xk])
            txt_info.set_text(
                f"K  = ({xk:.3f}, {P['K_y']:.3f})\n"
                f"I  = ({r['I'][0]:.3f}, {r['I'][1]:.3f})\n"
                f"th_AB = {r['th_AB']:.2f} deg\n"
                f"th_FG = {r['th_FG']:.2f} deg"
            )
        fig.canvas.draw_idle()

    update(xk0)

    slider = Slider(ax_slide, 'K_x', P['K_x_min'], P['K_x_max'],
                    valinit=xk0, color='#7B1FA2')
    slider.on_changed(lambda val: update(slider.val))

    plt.show()


# ══════════════════════════════════════════════════════════════════════
#  進入點
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if '--static' in sys.argv:
        static_mode()
    else:
        interactive_mode()
