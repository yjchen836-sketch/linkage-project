#!/usr/bin/env python3
"""
Linkage Mechanism Kinematic Analysis
══════════════════════════════════════════════════════════════════════

Mechanism topology (K → I):

  K (slider input, horizontal rail)
    │  [link B-K, length L_KB]
    ▼
    B ── red rigid body, pivot at A (fixed)
    │  [body arm L_AB → B;  L_AC → C;  angle α_BAC between them]
    ▼
    C
    │  [blue link C-D, length L_CD]
    ▼
    D  ←── D is an arm of the CYAN rigid body (pivot E, fixed)
    │        cyan body EDH:  E(fixed pivot) – D(arm, L_ED) – H(arm, L_EH)
    │        α_DEH = fixed angle between ED and EH on the cyan body
    ▼
    H (on cyan body, also a vertex of the green triangle)
    │
    │  green rigid triangle I-H-G:
    │    G also constrained by light-green link F-G  (L_FG, F fixed)
    │    |GH| = L_GH (rigid triangle side)
    │    I at angle α_GHI from HG direction (on green body)
    ▼
    I  (output point)

Geometric quantities required:
  Fixed coords : A, E, F  (all in fixed frame JAEF)
  Link lengths : L_KB, L_AB, L_AC, L_CD
                 L_ED, L_EH          (cyan rigid body arms from E)
                 L_FG                (light-green link F-G)
                 L_GH                (green triangle side G-H)
                 L_HI                (green triangle side H-I)
  Angles       : α_BAC  – angle from AB to AC on red body
                 α_DEH  – angle from ED to EH on cyan body
                 α_GHI  – angle from HG to HI on green body

Usage:
  python linkage_analysis.py           # interactive (drag slider)
  python linkage_analysis.py --static  # save PNG
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ══════════════════════════════════════════════════════════════════════
#  ★★★  EDITABLE GEOMETRY  ★★★
# ══════════════════════════════════════════════════════════════════════

P = {
    # ── Fixed frame JAEF pivot coords (x, y) ──────────────────────────
    'J': np.array([  0.00,   0.00]),   # slider rail anchor (measured)
    'A': np.array([ -7.55,  -8.83]),   # red body fixed pivot (measured)
    'E': np.array([ 44.98,  -4.13]),   # cyan body fixed pivot (measured)
    'F': np.array([ 44.98,   1.87]),   # light-green link fixed pivot (measured)

    # ── Slider K rail (horizontal line y = K_y) ───────────────────────
    'K_y'    :  0.25,
    'K_x_min':  0.54,   # stroke start (input min, measured)
    'K_x_max': 10.50,   # stroke end   (input max, measured)

    # ── Red rigid body (pivot A) ──────────────────────────────────────
    'L_KB'  : 10.15,              # K – B (measured)
    'L_AB'  : 11.00,              # A – B (measured)
    'L_AC'  : 11.50,              # A – C (measured)
    'α_BAC' : np.radians(52.72),  # ∠BAC on red body (measured)

    # ── Blue link C-D ─────────────────────────────────────────────────
    'L_CD'  : 22.00,              # C – D (measured)

    # ── Cyan rigid body EDH (pivot E) ─────────────────────────────────
    # E is the FIXED pivot; D and H are both arms on this one rigid body.
    # As D is pushed by the blue link, the whole body rotates about E,
    # and H (the other arm) drives the green triangle.
    'L_ED'  :  5.50,              # E – D  arm (measured)
    'L_EH'  : 30.00,              # E – H  arm (measured)
    'α_DEH' : np.radians(58.18),  # angle from ED to EH on cyan body (measured)

    # ── Light-green link F-G (F fixed, G floats) ──────────────────────
    'L_FG'  : 31.82,              # F – G (measured)

    # ── Green rigid triangle I-H-G ────────────────────────────────────
    'L_GH'  :  5.70,              # G – H  triangle side (measured)
    'L_HI'  : 26.50,              # H – I  output arm (measured)
    'α_GHI' : np.radians(86.36),  # angle from HG dir to HI dir (measured)
}

# Branch selection (+1 / -1).  Flip a sign if the mechanism pose is mirrored.
SIGNS = {
    'B': 1,    # circle(A,L_AB) ∩ circle(K,L_KB)
    'D': -1,   # circle(C,L_CD) ∩ circle(E,L_ED)
    'G':  1,   # circle(F,L_FG) ∩ circle(H,L_GH)
}


# ══════════════════════════════════════════════════════════════════════
#  Geometry helper
# ══════════════════════════════════════════════════════════════════════

def circle_intersect(c1, r1, c2, r2, sign=1):
    """
    Intersection of two circles.  sign=+1/-1 selects one of the two solutions.
    Returns None when circles do not intersect.
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
#  Forward kinematics:  K_x  →  all joint positions
# ══════════════════════════════════════════════════════════════════════

def solve(xk, p=None, s=None):
    """
    Given slider x-coord xk, compute every joint position.

    Steps:
      ①  B : circle(A, L_AB)  ∩  circle(K, L_KB)
      ②  C : red rigid body rotation th_AB → C
      ③  D : circle(C, L_CD)  ∩  circle(E, L_ED)   [D is arm of cyan body]
      ④  H : cyan body (pivot E):  th_ED + α_DEH  → H
      ⑤  G : circle(F, L_FG)  ∩  circle(H, L_GH)
      ⑥  I : green triangle:  th_HG + α_GHI  → I

    Returns dict of positions, or None on dead-point / out-of-range.
    """
    if p is None: p = P
    if s is None: s = SIGNS

    K = np.array([xk, p['K_y']])

    # ① B
    B = circle_intersect(p['A'], p['L_AB'], K, p['L_KB'], s['B'])
    if B is None:
        return None

    # ② C  (red rigid body rotates about A)
    th_AB = np.arctan2(B[1] - p['A'][1], B[0] - p['A'][0])
    th_AC = th_AB + p['α_BAC']
    C = p['A'] + p['L_AC'] * np.array([np.cos(th_AC), np.sin(th_AC)])

    # ③ D  (D lies on cyan body arm from E; also on circle from C)
    D = circle_intersect(C, p['L_CD'], p['E'], p['L_ED'], s['D'])
    if D is None:
        return None

    # ④ H  (cyan rigid body EDH pivots at E; α_DEH is fixed body angle)
    th_ED = np.arctan2(D[1] - p['E'][1], D[0] - p['E'][0])
    th_EH = th_ED + p['α_DEH']
    H = p['E'] + p['L_EH'] * np.array([np.cos(th_EH), np.sin(th_EH)])

    # ⑤ G  (F-G light-green link + green triangle side G-H)
    G = circle_intersect(p['F'], p['L_FG'], H, p['L_GH'], s['G'])
    if G is None:
        return None

    # ⑥ I  (green rigid triangle; α_GHI from HG direction at H)
    th_HG = np.arctan2(G[1] - H[1], G[0] - H[0])
    th_HI = th_HG + p['α_GHI']
    I = H + p['L_HI'] * np.array([np.cos(th_HI), np.sin(th_HI)])

    return dict(
        K=K, B=B, C=C, D=D, H=H, G=G, I=I,
        A=p['A'], E=p['E'], F=p['F'],
        th_AB=float(np.degrees(th_AB)),
        th_ED=float(np.degrees(th_ED)),
        th_HG=float(np.degrees(th_HG)),
    )


def sweep(p=None, s=None, n=500):
    """Sweep full K stroke; return (K_x_array, trajectory_dict)."""
    if p is None: p = P
    if s is None: s = SIGNS
    xs   = np.linspace(p['K_x_min'], p['K_x_max'], n)
    keys = ['K', 'B', 'C', 'D', 'H', 'G', 'I']
    traj = {k: [] for k in keys}
    kxs  = []
    for xk in xs:
        r = solve(xk, p, s)
        if r is not None:
            kxs.append(xk)
            for k in keys:
                traj[k].append(r[k])
    return np.array(kxs), {k: np.array(v) for k, v in traj.items()}


# ══════════════════════════════════════════════════════════════════════
#  Colours
# ══════════════════════════════════════════════════════════════════════

C_ = {
    'frame'  : '#424242',
    'rail'   : '#9E9E9E',
    'red'    : '#E53935',
    'BK'     : '#AD1457',
    'blue'   : '#1565C0',
    'cyan'   : '#00838F',
    'lgreen' : '#8BC34A',
    'green'  : '#2E7D32',
    'trail'  : '#FF8F00',
    'K'      : '#7B1FA2',
    'I'      : '#E65100',
    'fixed'  : '#263238',
}


# ══════════════════════════════════════════════════════════════════════
#  Draw one mechanism pose
# ══════════════════════════════════════════════════════════════════════

def draw_mechanism(ax, r, trail_I=None, p=None):
    if p is None: p = P
    ax.cla()

    ax.axhline(p['K_y'], color=C_['rail'], lw=1.2, ls='--', zorder=0,
               label='Rail (y=const)')

    if trail_I is not None and len(trail_I) > 1:
        ax.plot(trail_I[:, 0], trail_I[:, 1], '-',
                color=C_['trail'], lw=2, alpha=0.5, zorder=1,
                label='Trajectory of I')

    # fixed frame
    frame = np.array([p['J'], p['A'], p['E'], p['F'], p['J']])
    ax.plot(frame[:, 0], frame[:, 1], '--', color=C_['frame'],
            lw=1.2, alpha=0.4, zorder=2)
    for name, pt in [('J', p['J']), ('A', p['A']),
                     ('E', p['E']), ('F', p['F'])]:
        ax.plot(*pt, 's', color=C_['fixed'], ms=8, zorder=7)
        ax.annotate(name, pt, textcoords='offset points', xytext=(5, 5),
                    fontsize=11, color=C_['fixed'], fontweight='bold')

    if r is None:
        ax.set_title('Dead point / out of range', color='red', fontsize=12)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
        return

    def seg(p1, p2, color, lw=2.5, ls='-', **kw):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], ls,
                color=color, lw=lw, zorder=3, **kw)

    seg(r['B'], r['K'],  C_['BK'],     lw=2.5, label='B-K')
    seg(r['A'], r['B'],  C_['red'],    lw=3.5)
    seg(r['A'], r['C'],  C_['red'],    lw=3.5)
    seg(r['B'], r['C'],  C_['red'],    lw=2,   label='Red body A-B-C  (pivot A)')
    seg(r['C'], r['D'],  C_['blue'],   lw=2.5, label='Blue  C-D')
    seg(r['E'], r['D'],  C_['cyan'],   lw=3.5)
    seg(r['E'], r['H'],  C_['cyan'],   lw=3.5)
    seg(r['D'], r['H'],  C_['cyan'],   lw=2,   label='Cyan body EDH  (pivot E)')
    seg(r['F'], r['G'],  C_['lgreen'], lw=2.5, label='Lt-green  F-G')
    seg(r['H'], r['G'],  C_['green'],  lw=3.5)
    seg(r['H'], r['I'],  C_['green'],  lw=3.5)
    seg(r['G'], r['I'],  C_['green'],  lw=2.5, label='Green triangle I-H-G')

    joints = [
        ('K', r['K'],  C_['K'],     13, 's'),
        ('B', r['B'],  C_['red'],    8, 'o'),
        ('C', r['C'],  C_['red'],    8, 'o'),
        ('D', r['D'],  C_['cyan'],   8, 'o'),
        ('H', r['H'],  C_['cyan'],   8, 'o'),
        ('G', r['G'],  C_['lgreen'], 8, 'o'),
        ('I', r['I'],  C_['I'],     13, '*'),
    ]
    for name, pt, col, ms, mk in joints:
        ax.plot(*pt, mk, color=col, ms=ms, zorder=8,
                markeredgecolor='white', mew=0.8)
        ax.annotate(name, pt, textcoords='offset points', xytext=(6, 5),
                    fontsize=11, color=col, fontweight='bold')

    ax.annotate(
        f"th_AB = {r['th_AB']:.1f} deg\n"
        f"th_ED = {r['th_ED']:.1f} deg\n"
        f"th_HG = {r['th_HG']:.1f} deg",
        xy=(0.02, 0.03), xycoords='axes fraction', fontsize=8.5,
        color='#37474F',
        bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.75)
    )

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('x  [length unit]', fontsize=10)
    ax.set_ylabel('y  [length unit]', fontsize=10)
    ax.set_title('Linkage Mechanism Diagram', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.85)


# ══════════════════════════════════════════════════════════════════════
#  Relationship plots
# ══════════════════════════════════════════════════════════════════════

def draw_relations(axes, kxs, traj):
    ax1, ax2, ax3 = axes
    Ix     = traj['I'][:, 0]
    Iy     = traj['I'][:, 1]
    dI_cum = np.cumsum(np.hypot(np.diff(Ix, prepend=Ix[0]),
                                np.diff(Iy, prepend=Iy[0])))
    for ax, data, ylabel, title, col in [
        (ax1, Ix,     'I_x',        'I_x  vs  K_x',               C_['I']),
        (ax2, Iy,     'I_y',        'I_y  vs  K_x',               '#1B5E20'),
        (ax3, dI_cum, 'Cumul. arc', 'I cumul. displacement vs K_x','#6A1B9A'),
    ]:
        ax.cla()
        ax.plot(kxs, data, '-', color=col, lw=2)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('K_x', fontsize=9)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)


# ══════════════════════════════════════════════════════════════════════
#  Static output
# ══════════════════════════════════════════════════════════════════════

def static_mode():
    kxs, traj = sweep()
    if len(kxs) == 0:
        print("ERROR: No solution across full stroke — adjust P or SIGNS.")
        sys.exit(1)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Linkage Mechanism: K-I Motion Analysis',
                 fontsize=15, fontweight='bold', y=0.98)

    ax_mech = fig.add_axes([0.04, 0.40, 0.46, 0.54])
    ax_xy   = fig.add_axes([0.55, 0.40, 0.42, 0.54])
    ax1     = fig.add_axes([0.04, 0.06, 0.28, 0.28])
    ax2     = fig.add_axes([0.38, 0.06, 0.28, 0.28])
    ax3     = fig.add_axes([0.72, 0.06, 0.24, 0.28])

    xk_mid = kxs[len(kxs) // 2]
    draw_mechanism(ax_mech, solve(xk_mid), traj['I'])

    ax_xy.plot(traj['I'][:, 0], traj['I'][:, 1], '-',
               color=C_['trail'], lw=2.5, label='I trajectory')
    ax_xy.plot(traj['K'][:, 0],
               np.full(len(kxs), P['K_y']), '-',
               color=C_['K'], lw=2, alpha=0.5, label='K (rail)')
    for idx, mk, lbl in [(0, '^', 'Start'), (-1, 'v', 'End')]:
        ax_xy.plot(*traj['I'][idx], mk, color=C_['I'], ms=10)
        ax_xy.annotate(lbl, traj['I'][idx],
                       textcoords='offset points', xytext=(6, 4), fontsize=9)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_xlabel('x', fontsize=10)
    ax_xy.set_ylabel('y', fontsize=10)
    ax_xy.set_title('Trajectory of I  (XY plane)', fontsize=11, fontweight='bold')
    ax_xy.legend(fontsize=9)

    draw_relations([ax1, ax2, ax3], kxs, traj)
    for ax_r, data in [(ax1, traj['I'][:, 0]), (ax2, traj['I'][:, 1])]:
        ax_r.plot(xk_mid, data[len(kxs) // 2], 'o', color='red', ms=8,
                  zorder=5, label='Current')
        ax_r.legend(fontsize=8)

    plt.savefig('linkage_analysis.png', dpi=150, bbox_inches='tight')
    print('Saved: linkage_analysis.png')
    plt.show()


# ══════════════════════════════════════════════════════════════════════
#  Interactive mode
# ══════════════════════════════════════════════════════════════════════

def interactive_mode():
    kxs, traj = sweep()
    if len(kxs) == 0:
        print("ERROR: No solution across full stroke — adjust P or SIGNS.")
        sys.exit(1)

    xk0 = (P['K_x_min'] + P['K_x_max']) / 2
    r0  = solve(xk0)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle('Linkage K-I Analysis  (drag slider below)',
                 fontsize=13, fontweight='bold')

    ax_mech  = fig.add_axes([0.03, 0.17, 0.50, 0.76])
    ax_ix    = fig.add_axes([0.58, 0.60, 0.38, 0.30])
    ax_iy    = fig.add_axes([0.58, 0.17, 0.38, 0.30])
    ax_slide = fig.add_axes([0.10, 0.05, 0.80, 0.04])

    draw_mechanism(ax_mech, r0, traj['I'])

    Ix = traj['I'][:, 0]
    Iy = traj['I'][:, 1]
    ax_ix.plot(kxs, Ix, '-', color=C_['I'],   lw=2)
    ax_iy.plot(kxs, Iy, '-', color='#2E7D32', lw=2)
    for ax_r, ylabel, title in [
        (ax_ix, 'I_x', 'I_x  vs  K_x'),
        (ax_iy, 'I_y', 'I_y  vs  K_x'),
    ]:
        ax_r.set_ylabel(ylabel, fontsize=9)
        ax_r.set_xlabel('K_x',  fontsize=9)
        ax_r.set_title(title, fontsize=9, fontweight='bold')
        ax_r.grid(True, alpha=0.3)

    dot_ix, = ax_ix.plot([], [], 'o', color='red', ms=9, zorder=5)
    dot_iy, = ax_iy.plot([], [], 'o', color='red', ms=9, zorder=5)
    vl_ix = ax_ix.axvline(xk0, color='gray', lw=1, ls=':')
    vl_iy = ax_iy.axvline(xk0, color='gray', lw=1, ls=':')

    txt = fig.text(0.58, 0.94, '', fontsize=9, va='top',
                   bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    def update(xk):
        r = solve(xk)
        draw_mechanism(ax_mech, r, traj['I'])
        if r:
            dot_ix.set_data([xk], [r['I'][0]])
            dot_iy.set_data([xk], [r['I'][1]])
            vl_ix.set_xdata([xk, xk])
            vl_iy.set_xdata([xk, xk])
            txt.set_text(
                f"K  = ({xk:.3f}, {P['K_y']:.2f})\n"
                f"I  = ({r['I'][0]:.3f}, {r['I'][1]:.3f})\n"
                f"th_AB = {r['th_AB']:.2f} deg\n"
                f"th_ED = {r['th_ED']:.2f} deg\n"
                f"th_HG = {r['th_HG']:.2f} deg"
            )
        fig.canvas.draw_idle()

    update(xk0)
    slider = Slider(ax_slide, 'K_x', P['K_x_min'], P['K_x_max'],
                    valinit=xk0, color='#7B1FA2')
    slider.on_changed(lambda v: update(slider.val))
    plt.show()


# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if '--static' in sys.argv:
        static_mode()
    else:
        interactive_mode()
