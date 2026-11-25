# -*- coding: utf-8 -*-
"""
ЛР1. Перевірка середовища Python для розрахунку діаграм спрямованості антен.
Виводить версії інтерпретатора та бібліотек, виконує тестовий розрахунок ДС
напівхвильового вібратора і зберігає графіки у папку figures_lab1.
"""
import os
import sys
import platform

import numpy as np
import scipy
from scipy.special import j1
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures_lab1"
os.makedirs(OUT, exist_ok=True)

print("Перевірка середовища")
print(f"  ОС:         {platform.system()} {platform.release()}")
print(f"  Python:     {sys.version.split()[0]}")
print(f"  NumPy:      {np.__version__}")
print(f"  SciPy:      {scipy.__version__}")
print(f"  Matplotlib: {matplotlib.__version__}")

# тестовий розрахунок: ДС напівхвильового вібратора у площині E
theta = np.radians(np.linspace(0.001, 179.999, 3601))
F = np.abs(np.cos(np.pi / 2 * np.cos(theta)) / np.sin(theta))
F /= F.max()
th_deg = np.degrees(theta)

# ширина головної пелюстки за рівнем 0,707
above = th_deg[F >= 1 / np.sqrt(2)]
hpbw = above.max() - above.min()
print(f"Тест: ширина ДС напівхвильового вібратора 2θ0,5 = {hpbw:.2f}° (теорія 78°)")

# тест SciPy: лямбда-функція 2J1(x)/x для круглого розкриву
x = np.linspace(1e-6, 15, 1500)
L1 = 2 * j1(x) / x
x0 = brentq(lambda t: j1(t), 3.0, 4.5)
print(f"Тест: перший нуль 2J1(x)/x при x = {x0:.3f} (теорія 3,832)")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(th_deg, F, label="F(θ) – напівхвильовий вібратор")
ax.axhline(1 / np.sqrt(2), color="r", ls="--", lw=0.8, label="рівень 0,707")
ax.set_xlabel("θ, °")
ax.set_ylabel("F(θ)")
ax.set_xlim(0, 180)
ax.grid(True, alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "test_dipole_linear.png"), dpi=150)

fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(projection="polar")
ax.plot(theta, F, "b")
ax.plot(-theta, F, "b")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title("ДС напівхвильового вібратора (площина E)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "test_dipole_polar.png"), dpi=150)

print(f"Графіки збережено у папку {OUT}. Середовище налаштовано правильно.")
