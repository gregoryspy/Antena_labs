# -*- coding: utf-8 -*-
"""Допоміжні функції для аналізу та побудови діаграм спрямованості (ДС).

Використовуються в усіх лабораторних роботах з дисципліни
«Антенні пристрої та комплекси».
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 11,
    "mathtext.fontset": "dejavuserif",
})


def analyze(theta_deg, F, main_dir=None):
    """Аналіз нормованої ДС |F(θ)|.

    Повертає словник:
      main  – напрямок головного максимуму, град;
      left, right – кути, на яких |F| = 0,707 ліворуч і праворуч від максимуму;
      hpbw  – ширина головної пелюстки на рівні 0,707, град;
      nulls – напрямки нульового (мінімального) випромінювання, град;
      lobes – список (кут, рівень) максимумів бічних пелюсток.
    """
    th = np.asarray(theta_deg, float)
    a = np.abs(np.asarray(F, float))
    a = a / a.max()
    i0 = int(np.argmax(a)) if main_dir is None else int(np.argmin(np.abs(th - main_dir)))
    lvl = 1 / math.sqrt(2)

    def cross(direction):
        i = i0
        while a[i] > lvl and 0 <= i + direction < len(a):
            i += direction
        if a[i] > lvl or i == i0:
            return None
        j = i - direction
        return float(np.interp(lvl, [a[i], a[j]], [th[i], th[j]]))

    left, right = cross(-1), cross(+1)
    if left is None and right is not None and abs(th[0]) < 1e-9:
        left = -right  # симетрична ДС, побудована на [0, 90°]
    hpbw = (right - left) if (left is not None and right is not None) else None
    nulls, lobes = [], []
    for i in range(1, len(a) - 1):
        if a[i] <= a[i - 1] and a[i] < a[i + 1] and a[i] < 0.5:
            nulls.append(float(th[i]))
        if a[i] >= a[i - 1] and a[i] > a[i + 1] and i != i0 and a[i] < 0.999:
            lobes.append((float(th[i]), float(a[i])))
    return {"main": float(th[i0]), "left": left, "right": right, "hpbw": hpbw,
            "nulls": nulls, "lobes": lobes, "sll_db": 20 * math.log10(max(l[1] for l in lobes)) if lobes else None}


def plot_linear(fname, theta, curves, res, title=None, xlim=(0, 90), ylabel=r"$|F(\theta)|$",
                xlabel=r"$\theta$, град", mark_key=None, step=10, annotate=True):
    """Графік ДС у прямокутній системі координат з позначенням рівня 0,707, нулів і бічних пелюсток.

    curves – список (масив F, підпис, стиль); res – словник з результатами analyze() для кривих,
    на яких треба зробити позначки: {підпис: res}."""
    fig, ax = plt.subplots(figsize=(17 / 2.54, 10 / 2.54))
    for F, label, style in curves:
        ax.plot(theta, np.abs(F), style, lw=1.2, label=label)
    ax.axhline(1 / math.sqrt(2), color="0.4", ls=":", lw=0.8)
    ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), 0.72, "0,707", fontsize=8, color="0.3")
    markers = [("o", "C3"), ("s", "C2"), ("^", "C4"), ("D", "C5")]
    for k, (lab, r) in enumerate(res.items()):
        mk, col = markers[k % len(markers)]
        if r["left"] is not None and r["right"] is not None:
            for x in (r["left"], r["right"]):
                if xlim[0] <= x <= xlim[1]:
                    ax.plot([x, x], [0, 1 / math.sqrt(2)], "--", color=col, lw=0.7)
            ax.plot([max(r["left"], xlim[0]), min(r["right"], xlim[1])], [1 / math.sqrt(2)] * 2, "-", color=col, lw=1.6,
                    label=f"2θ0,5 {lab} = {r['hpbw']:.2f}°".replace(".", ","))
        nx = [x for x in r["nulls"] if xlim[0] <= x <= xlim[1]]
        ax.plot(nx, [0] * len(nx), "x", color=col, ms=7, mew=1.5, label=f"нулі {lab}")
        lx = [(x, y) for x, y in r["lobes"] if xlim[0] <= x <= xlim[1]]
        if lx:
            ax.plot([x for x, _ in lx], [y for _, y in lx], mk, color=col, ms=4.5, mfc="none", mew=1.2,
                    label=f"бічні пелюстки {lab}")
            if annotate:
                for x, y in lx[:8]:
                    ax.annotate(f"{y:.3f}".replace(".", ","), (x, y), textcoords="offset points", xytext=(2, 5),
                                fontsize=7, color=col)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.01, step))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(ls="--", lw=0.3, color="0.6")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def plot_polar(fname, theta, curves, title=None, symmetric=True, full=False):
    """Полярна ДС. Якщо symmetric=True, ДС задана на [0, 90°] і дзеркально відображається."""
    fig = plt.figure(figsize=(14 / 2.54, 12 / 2.54))
    ax = fig.add_subplot(111, projection="polar")
    th = np.radians(theta)
    for F, label, style in curves:
        a = np.abs(F)
        ax.plot(th, a, style, lw=1.1, label=label)
        if symmetric:
            ax.plot(-th, a, style, lw=1.1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if not full:
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
    ax.set_thetagrids(range(-90, 91, 15) if not full else range(0, 360, 30))
    ax.set_rlim(0, 1.0)
    ax.set_rticks([0.25, 0.5, 0.707, 1.0])
    ax.set_yticklabels(["0,25", "0,5", "0,707", "1"], fontsize=7)
    ax.grid(ls="--", lw=0.3)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(-0.05, -0.05))
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def print_table(title, header, rows):
    """Друк таблиці в консоль."""
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(header)]
    line = "+" + "+".join("-" * x for x in w) + "+"
    print(title)
    print(line)
    print("|" + "|".join(str(h).center(x) for h, x in zip(header, w)) + "|")
    print(line)
    for r in rows:
        print("|" + "|".join(str(c).center(x) for c, x in zip(r, w)) + "|")
    print(line)


# ============================================================================
# Лабораторна робота 7. ДОСЛІДЖЕННЯ ДЗЕРКАЛЬНОЇ АНТЕНИ (ДзА)
# Параболоїд обертання з опромінювачем «диполь з дисковим рефлектором». Варіант 7
# ============================================================================
import json, os
from scipy.special import jv

lam_list = [3, 3.2, 3.4, 3.5, 3.65, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 4.45, 3, 3, 2.8, 2.8, 3.1, 3.1, 3.6, 3.4, 2.5,
            3.2, 3.3, 3.4]
D_list = [0.8, 0.9, 1, 1.1, 1.2, 0.5, 0.5, 0.6, 0.6, 0.7, 0.6, 0.7, 1.2, 0.6, 0.8, 1.1, 0.7, 1, 1.3, 1.35, 1.4, 0.9,
          1.2, 1.3, 1.4]
f_list = [0.4, 0.4, 0.45, 0.45, 0.45, 0.25, 0.2, 0.25, 0.3, 0.3, 0.2, 0.4, 0.4, 0.25, 0.3, 0.25, 0.3, 0.3, 0.4, 0.3,
          0.4, 0.25, 0.3, 0.35, 0.4]
VARIANT = 7
OUT = "figures_lab7"
os.makedirs(OUT, exist_ok=True)

lam = lam_list[VARIANT - 1] * 1e-2
D = D_list[VARIANT - 1]
fF = f_list[VARIANT - 1]
k = 2 * math.pi / lam                 # 1. хвильове число
R0 = D / 2                            # 2. радіус дзеркала
p = 2 * fF                            # 3. параметр параболоїда
v = 3.5 * R0 / p                      # параметр у формулах (7.16), (7.17)
psi0 = 2 * math.atan(R0 / p)          # половина кута розкриву: tg(ψ0/2) = R0/p
h_m = R0 ** 2 / (2 * p)               # глибина дзеркала
print(f"Варіант {VARIANT}: λ = {lam*100:.1f} см, D = {D} м, f = {fF} м")
print(f"k = {k:.2f} рад/м, R0 = {R0} м, p = 2f = {p} м, R0/p = {R0/p:.4f}, v = 3,5R0/p = {v:.4f}")
print(f"ψ0 = 2arctg(R0/p) = {math.degrees(psi0):.2f}°, глибина h = R0²/(2p) = {h_m*100:.2f} см, kR0 = {k*R0:.2f}, D/λ = {D/lam:.1f}")

theta = np.linspace(0, 12, 120001)    # головна пелюстка дуже вузька – детальна сітка до 12°
theta_w = np.linspace(0, 90, 90001)


def lam1(x):
    x = np.asarray(x, float); o = np.ones_like(x); m = np.abs(x) > 1e-9
    o[m] = 2 * jv(1, x[m]) / x[m]; return o


def lam2(x):
    x = np.asarray(x, float); o = np.ones_like(x); m = np.abs(x) > 1e-6
    o[m] = 8 * jv(2, x[m]) / x[m] ** 2; return o


def F_current(th_deg, sign):
    """ДС за розподілом щільності струму (7.16) (sign=+1, площина E) та (7.17) (sign=−1, площина H)."""
    t = np.radians(th_deg)
    u = k * R0 * np.sin(t)
    u = np.where(u < 1e-9, 1e-9, u)
    uu = np.where(np.abs(v ** 2 - u ** 2) < 1e-9, u + 1e-6, u)
    T1 = (v * jv(1, v) * jv(0, uu) - uu * jv(1, uu) * jv(0, v)) / (v ** 2 - uu ** 2)
    T2 = jv(1, u) / u
    w = 1.5 * v
    uw = np.where(np.abs(w ** 2 - u ** 2) < 1e-9, u + 1e-6, u)
    T3 = (uw * jv(1, uw) * jv(2, w) - w * jv(1, w) * jv(2, uw)) / (w ** 2 - uw ** 2)
    norm = 0.74 * jv(1, v) / v + 0.13
    return np.cos(t / 2) ** 2 * (0.74 * T1 + 0.26 * T2 + sign * 0.25 * T3) / norm


# П'єдестали (7.25), (7.26) та ДС апертурним методом (7.27), (7.28)
TE = math.cos(psi0 / 2) ** 2 * math.cos(psi0) * math.sin(math.pi / 2 * math.cos(psi0))
TH = math.cos(psi0 / 2) ** 2 * math.sin(math.pi / 2 * math.cos(psi0))


def F_aperture(th_deg, T):
    t = np.radians(th_deg)
    u = k * R0 * np.sin(t)
    return 2 / (1 + T) * (T * lam1(u) + (1 - T) / 2 * lam2(u)) * np.cos(t / 2) ** 2


FE_c, FH_c = F_current(theta, +1), F_current(theta, -1)
FE_a, FH_a = F_aperture(theta, TE), F_aperture(theta, TH)
res = {"Ec": analyze(theta, FE_c), "Hc": analyze(theta, FH_c), "Ea": analyze(theta, FE_a), "Ha": analyze(theta, FH_a)}
for kk, r in res.items():
    print(f"{kk}: 2θ0,5 = {r['hpbw']:.3f}°, перші нулі {[round(x,3) for x in r['nulls'][:4]]}, "
          f"бічні {[(round(a,3), round(b,4)) for a,b in r['lobes'][:4]]}, РБП = {r['sll_db']:.1f} дБ")
print(f"TE = {TE:.4f}, TH = {TH:.4f}")
# Оцінка дальніх пелюсток (7.29) на всьому діапазоні кутів
tw = np.radians(theta_w)
uw_ = k * R0 * np.sin(tw)
F_far = np.abs((1 + np.cos(tw)) / 2 * np.where(uw_ < 1e-9, 1.0, np.sin(uw_) / np.where(uw_ < 1e-9, 1, uw_)))
env = (1 + np.cos(tw)) / 2 / np.where(uw_ < 1e-9, 1, uw_)
far_pts = {a: float(env[int(a * 1000)]) for a in (10, 20, 30, 45, 60, 90)}
# КВП, ККД, КСД, КП
nuA1 = 7.5 * (R0 / p) ** 2 / (1 + (R0 / p) ** 2) ** 4                    # (7.31)
nuA2 = {T_: 0.75 * (1 + T_) ** 2 / (1 + T_ + T_ ** 2) for T_ in (TE, TH)}  # (7.32)
nuA2m = math.sqrt(nuA2[TE] * nuA2[TH])
eta_p = 1 - 5 / 8 * math.cos(psi0) ** 3 - 3 / 8 * math.cos(psi0) ** 5   # (7.35)
Sp = math.pi * R0 ** 2
Dm1 = 4 * math.pi / lam ** 2 * Sp * nuA1                                  # (7.30)
Dm2 = 4 * math.pi / lam ** 2 * Sp * nuA2m
Gm1 = Dm1 * eta_p                                                         # (7.33)
# Допуски (7.36)–(7.39)
d_rho = lam / (16 * (1 + math.cos(psi0)))
d_rho0 = lam / 32
zs1 = lam / (8 * (1 - math.cos(psi0)))                                    # (7.37)
zs2 = lam * (R0 ** 2 + p ** 2) / (16 * R0 ** 2)                           # (7.38)
hp = res["Ea"]["hpbw"]
xs = hp * R0 / (60 * math.sin(psi0))                                      # з умови 60°·xs·sinψ0/R0 ≤ 2θ0,5
print(f"νA (7.31) = {nuA1:.4f}; νA (7.32): E {nuA2[TE]:.4f}, H {nuA2[TH]:.4f}; ηп = {eta_p:.4f}")
print(f"Dm (7.31) = {Dm1:.0f} ({10*math.log10(Dm1):.2f} дБ), Dm (7.32) = {Dm2:.0f}; Gm = Dm·ηп = {Gm1:.0f} ({10*math.log10(Gm1):.2f} дБ)")
print(f"Допуски: Δρ ≤ {d_rho*1e3:.2f} мм (біля вершини {d_rho0*1e3:.2f} мм), zs ≤ {zs1*1e3:.1f} мм (7.37), {zs2*1e3:.1f} мм (7.38), xs ≤ {xs*1e3:.1f} мм")

# Графіки
plot_linear(f"{OUT}/ds_E_linear.png", theta,
            [(FE_c, "розподіл струму (7.16)", "C3-"), (FE_a, "апертурний метод (7.27)", "C1--")],
            {"(7.16)": res["Ec"]}, xlim=(0, 6), step=0.5, ylabel=r"$|F_E(\theta)|$")
plot_linear(f"{OUT}/ds_H_linear.png", theta,
            [(FH_c, "розподіл струму (7.17)", "C0-"), (FH_a, "апертурний метод (7.28)", "C9--")],
            {"(7.17)": res["Hc"]}, xlim=(0, 6), step=0.5, ylabel=r"$|F_H(\theta)|$")
fig = plt.figure(figsize=(14 / 2.54, 12 / 2.54))
ax = fig.add_subplot(111, projection="polar")
FEw, FHw = F_current(theta_w, +1), F_current(theta_w, -1)
trw = np.radians(theta_w)
for F, lab_, st in ((FHw, r"$F_H(\theta)$", "C0-"), (FEw, r"$F_E(\theta)$", "C3-")):
    r_db = np.clip(20 * np.log10(np.maximum(np.abs(F), 1e-6)), -40, 0) + 40
    ax.plot(trw, r_db, st, lw=0.9, label=lab_); ax.plot(-trw, r_db, st, lw=0.9)
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_thetamin(-90); ax.set_thetamax(90)
ax.set_thetagrids(range(-90, 91, 15)); ax.set_rlim(0, 40); ax.set_rticks([0, 10, 20, 30, 40])
ax.set_yticklabels(["−40", "−30", "−20", "−10", "0 дБ"], fontsize=7); ax.grid(ls="--", lw=0.3)
ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(-0.05, -0.05)); fig.tight_layout(); fig.savefig(f"{OUT}/ds_polar.png", dpi=200); plt.close(fig)
fig, ax = plt.subplots(figsize=(17 / 2.54, 8.5 / 2.54))
ax.plot(theta, 20 * np.log10(np.maximum(np.abs(FE_c), 1e-5)), "C3-", lw=1.1, label=r"$F_E$ (7.16)")
ax.plot(theta, 20 * np.log10(np.maximum(np.abs(FH_c), 1e-5)), "C0-", lw=1.1, label=r"$F_H$ (7.17)")
ax.plot(theta_w[theta_w <= 12], 20 * np.log10(np.maximum(env[theta_w <= 12], 1e-5)), "k:", lw=1, label="обвідна (7.29)")
ax.axhline(-3, color="0.5", ls="-.", lw=0.7)
ax.set_xlim(0, 12); ax.set_ylim(-50, 1); ax.grid(ls="--", lw=0.3); ax.set_xlabel(r"$\theta$, град"); ax.set_ylabel("20 lg |F|, дБ")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/ds_db.png", dpi=200); plt.close(fig)
# Амплітудний розподіл у розкриві (7.12)–(7.15)
ps = np.linspace(0, psi0, 300)
Rn = np.tan(ps / 2) / math.tan(psi0 / 2)
AE = (1 + np.cos(ps)) / 2 * np.cos(ps) * np.sin(np.pi / 2 * np.cos(ps))
AH = (1 + np.cos(ps)) / 2 * np.sin(np.pi / 2 * np.cos(ps))
fig, ax = plt.subplots(figsize=(15 / 2.54, 7.5 / 2.54))
ax.plot(Rn, AE, "C3-", label=r"$A_E(R')$ (7.14)"); ax.plot(Rn, AH, "C0-", label=r"$A_H(R')$ (7.15)")
ax.plot(Rn, 1 - (1 - TE) * Rn ** 2, "C3:", label=f"1−(1−TE)R′², TE = {TE:.3f}".replace(".", ","))
ax.plot(Rn, 1 - (1 - TH) * Rn ** 2, "C0:", label=f"1−(1−TH)R′², TH = {TH:.3f}".replace(".", ","))
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.grid(ls="--", lw=0.3); ax.set_xlabel(r"$R' = R/R_0$"); ax.set_ylabel("A(R′)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/aperture_distr.png", dpi=200); plt.close(fig)

json.dump({"variant": VARIANT, "lam": lam, "D": D, "f": fF, "k": k, "R0": R0, "p": p, "v": v, "psi0": math.degrees(psi0),
           "h": h_m, "TE": TE, "TH": TH, "res": res, "far": far_pts, "nuA1": nuA1, "nuA2E": nuA2[TE], "nuA2H": nuA2[TH],
           "nuA2m": nuA2m, "eta_p": eta_p, "Sp": Sp, "Dm1": Dm1, "Dm2": Dm2, "Gm1": Gm1, "d_rho": d_rho, "d_rho0": d_rho0,
           "zs1": zs1, "zs2": zs2, "xs": xs, "kR0": k * R0}, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
