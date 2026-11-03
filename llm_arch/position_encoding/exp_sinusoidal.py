"""Sinusoidal 位置编码的四组实验绘图。

设计原则：
1. 不修改 llm_arch/position_encoding/sinusoidal.py 的实现，只 import 它的 SinusoidalPE；
2. 输入全 0 张量调用模块，返回的 X + PE 就是 PE 本身（X=0）；
3. 只做信号级实验，不训练模型，每个图对应博客里的一个论断。

运行：
    MPLCONFIGDIR=/tmp python llm_arch/position_encoding/exp_sinusoidal.py

输出（llm_arch/position_encoding/figs/）：
    fig1_pe_distribution.png  PE 元素分布 / 每维标准差 / 每位置范数
    fig2_dot_product.png      PE 的 Gram 矩阵与点积随偏移的变化
    fig3_attention_decay.png  随机 q、k 下 attention score 随距离的衰减
    fig4_relative.png         以某个位置为中心向两侧扩散的相对性
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sinusoidal import SinusoidalPE

BASE = 1e4
FIG_DIR = Path(__file__).resolve().parent / "figs"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def get_pe(slen: int, d: int) -> torch.Tensor:
    """取 PE 矩阵 [slen, d]。

    用全 0 输入调用自己的模块：X + PE = 0 + PE，因此返回的就是位置编码本身。
    这样实验用的就是 sinusoidal.py 里的实现，不需要复制一份公式。
    """
    module = SinusoidalPE()
    with torch.no_grad():
        out = module(torch.zeros(slen, d))
    # 兼容两种返回：直接返回 X+PE，或返回 (X+PE, PE)
    return out[-1] if isinstance(out, (tuple, list)) else out


def omega(d: int) -> torch.Tensor:
    """每对维度的角频率 ω_i = base^(-2i/d)，i = 0..d/2-1。"""
    i = torch.arange(0, d, 2, dtype=torch.float64)
    return BASE ** (-i / d)


def theory_dot(k: torch.Tensor, d: int) -> torch.Tensor:
    """PE_t · PE_(t+k) 的理论值：Σ_i cos(ω_i k)，与 t 无关。"""
    return torch.cos(k[:, None] * omega(d)[None, :]).sum(-1)


def arcsine_pdf(x: np.ndarray) -> np.ndarray:
    """sin(ωt) 在 t 均匀采样时的极限分布：p(x) = 1 / (π√(1-x²))。"""
    return 1.0 / (np.pi * np.sqrt(np.clip(1.0 - x ** 2, 1e-12, None)))


# ---------------------------------------------------------------------------
# 实验 1：PE 的“大小”分布
# ---------------------------------------------------------------------------
def exp1_distribution(slen: int = 512, d: int = 128) -> None:
    pe = get_pe(slen, d).double()
    values = pe.flatten().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))

    # (a) 元素分布按“波长是否小于序列长度”拆成两组：
    #     高频维（λ<s）走过多个周期，接近单频的反正弦分布；
    #     低频维（λ≥s）没走完一个周期，值集中在 0 附近。
    lam = 2 * math.pi / omega(d).numpy()
    high = np.repeat(lam < slen, 2)          # 每对 (sin, cos) 占两维
    bins = np.linspace(-1, 1, 80)
    hi = pe[:, torch.tensor(high)].flatten().numpy()
    lo = pe[:, torch.tensor(~high)].flatten().numpy()
    axes[0].hist(hi, bins=bins, density=True, alpha=0.65, color="#4c72b0",
                 label=f"$\\lambda<s$ ({high.sum()} dims)")
    axes[0].hist(lo, bins=bins, density=True, alpha=0.65, color="#dd8452",
                 label=f"$\\lambda\\geq s$ ({(~high).sum()} dims)")
    grid = np.linspace(-0.999, 0.999, 500)
    axes[0].plot(grid, arcsine_pdf(grid), "r--", lw=1.4,
                 label="arcsine (single freq, $s\\gg\\lambda$)")
    axes[0].set_xlim(-1.05, 1.05)
    axes[0].set_xlabel("PE value")
    axes[0].set_ylabel("density")
    axes[0].annotate("low-freq dims:\n$\\sin\\approx 0,\\ \\cos\\approx 1$",
                     xy=(0.3, 5.0), fontsize=7, color="#8c4b1f")
    axes[0].set_title(f"element distribution (s={slen}, d={d})")
    axes[0].legend(fontsize=6.5)

    # (b) 每一维的标准差：高频维接近 1/√2，低频维因为序列没覆盖完整周期而变小
    dim_std = pe.std(dim=0).numpy()
    t = torch.arange(slen, dtype=torch.float64)
    w = omega(d)
    std_sin = torch.sin(t[:, None] * w[None, :]).std(dim=0)
    std_cos = torch.cos(t[:, None] * w[None, :]).std(dim=0)
    theory_std = torch.stack([std_sin, std_cos], dim=-1).flatten().numpy()
    axes[1].plot(theory_std, "k--", lw=1.2, label="theory for finite $s$")
    axes[1].plot(dim_std, "o", ms=2.5, mfc="none", label="empirical")
    axes[1].axhline(1 / math.sqrt(2), color="r", ls=":", lw=1.2,
                    label=r"$1/\sqrt{2}$ (infinite $s$)")
    axes[1].set_ylim(0, 0.85)
    axes[1].set_xlabel("dimension (high -> low frequency)")
    axes[1].set_ylabel("std over positions")
    axes[1].set_title("finite-length effect: low-freq dims vary slowly")
    axes[1].legend(fontsize=7)

    # (c) 每个位置的范数：恒为 √(d/2)
    norms = pe.norm(dim=1).numpy()
    axes[2].plot(norms, lw=2.5, color="#55a868",
                 label=rf"empirical (theory $\sqrt{{d/2}}={math.sqrt(d/2):.3f}$)")
    axes[2].set_ylim(0, math.sqrt(d / 2) * 1.2)
    axes[2].set_xlabel("position")
    axes[2].set_ylabel(r"$\|PE_t\|_2$")
    axes[2].set_title("per-position norm is constant")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_pe_distribution.png")
    plt.close(fig)

    print(f"[实验1] 元素范围=[{pe.min():.4f}, {pe.max():.4f}], "
          f"整体 std={pe.std():.4f}")
    print(f"        每维 std 范围=[{dim_std.min():.4f}, {dim_std.max():.4f}] "
          f"(理论 1/√2={1/math.sqrt(2):.4f})")
    print(f"        范数范围=[{norms.min():.6f}, {norms.max():.6f}] "
          f"(理论 √(d/2)={math.sqrt(d/2):.6f})")


# ---------------------------------------------------------------------------
# 实验 2：PE 的点积 / 相关性（不涉及 attention）
# ---------------------------------------------------------------------------
def exp2_dot_product(slen: int = 128, d: int = 128) -> None:
    pe = get_pe(slen, d).double()
    gram = pe @ pe.T                                   # [slen, slen]
    k = torch.arange(slen, dtype=torch.float64)
    theory = theory_dot(k, d)                          # 只依赖偏移

    # 把所有元素按 |i-j| 分组，检查是否都落在同一条曲线上
    idx = torch.arange(slen)
    dist = (idx[:, None] - idx[None, :]).abs()
    gram_by_dist = torch.stack([gram[dist == kk].mean() for kk in range(slen)])
    max_dev = (gram - theory[dist]).abs().max().item()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) Gram 矩阵：值只沿对角线带状变化
    im = axes[0].imshow(gram.numpy(), cmap="viridis")
    axes[0].set_xlabel("position $j$")
    axes[0].set_ylabel("position $i$")
    axes[0].set_title(r"$PE\,PE^\top$: constant along diagonals")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

    # (b) 把所有位置、所有偏移相同的元素聚在一起：
    #     每个偏移 k 上取 G[t, t+k] 的均值，应该与理论曲线完全重合
    means = torch.stack([gram.diagonal(kk).mean() for kk in range(slen)])
    axes[1].plot(theory.numpy(), color="0.6", lw=3.0, alpha=0.8,
                 label=r"theory $\sum_i\cos(\omega_i k)$")
    axes[1].plot(means.numpy(), "o", ms=2.8, color="#1f4e79", mfc="none",
                 label="empirical mean over all $t$")
    axes[1].set_xlabel("offset $k$")
    axes[1].set_ylabel(r"$PE_t \cdot PE_{t+k}$")
    axes[1].set_title("similarity depends on offset only")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_dot_product.png")
    plt.close(fig)

    print(f"[实验2] Gram 与理论值 max|dev|={max_dev:.2e}; "
          f"k=0 时自相似={gram[0, 0]:.3f}, k={slen-1} 时={gram[0, -1]:.3f}")


# ---------------------------------------------------------------------------
# 实验 3：远程衰减 —— 随机 q、k 加上 PE 后的 attention score
# ---------------------------------------------------------------------------
def exp3_attention_decay(d: int = 64, center: int = 32, max_k: int = 64,
                         n_samples: int = 2048, seed: int = 0) -> None:
    """score(m, n) = (q + PE_m)·(k + PE_n)，m 固定、n = m + k。

    单次 q、k 的 score 噪声很大，所以对 512 组随机 q、k 取均值和标准差：
    均值随距离衰减，标准差基本不变 —— 这正是“相对位置可感知、但被内容噪声淹没”的体现。
    """
    torch.manual_seed(seed)
    pe = get_pe(center + 16 + max_k + 1, d).double()
    q = torch.randn(n_samples, d, dtype=torch.float64)
    k = torch.randn(n_samples, d, dtype=torch.float64)
    radii = torch.arange(0, max_k + 1)

    def scores_at(c: int, ks: torch.Tensor) -> torch.Tensor:
        # (B, d) 与 PE 相加后逐样本点积 -> (B, len(ks))
        qm = q + pe[c]
        kn = k[:, None, :] + pe[(c + ks)][None, :, :]
        return (qm[:, None, :] * kn).sum(-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))

    # (a) 不同中心点的均值 ± 标准差，曲线应重合
    for c, color in [(center, "#4c72b0"), (center + 16, "#dd8452")]:
        vals = scores_at(c, radii)
        mean, std = vals.mean(0).numpy(), vals.std(0).numpy()
        axes[0].plot(radii.numpy(), mean, color=color, lw=1.6, label=f"center={c}")
        axes[0].fill_between(radii.numpy(), mean - std, mean + std,
                             color=color, alpha=0.15)
    axes[0].axhline(0.0, color="k", ls=":", lw=1)
    axes[0].set_xlabel("distance $k$")
    axes[0].set_ylabel("attention score")
    axes[0].set_title("mean decays, std stays flat (shaded: ±1σ)")
    axes[0].legend(fontsize=8)

    # (b) 分解：q·k + 交叉项 + PE·PE，衰减只来自 PE 之间
    vals = scores_at(center, radii)
    pe_term = theory_dot(radii + 0, d)                 # PE_m · PE_n（只依赖 k）
    qk = (q * k).sum(-1).mean().expand_as(radii)
    cross = vals.mean(0) - qk - pe_term                # 两个交叉项，均值≈0
    axes[1].plot(radii.numpy(), qk.numpy(), "--", lw=1.2, label=r"$q\cdot k$ (const)")
    axes[1].plot(radii.numpy(), cross.numpy(), ":", lw=1.4, label="cross terms (mean≈0)")
    axes[1].plot(radii.numpy(), pe_term.numpy(), lw=1.8, label=r"$PE_m\cdot PE_n$")
    axes[1].plot(radii.numpy(), vals.mean(0).numpy(), "k-", lw=1.2, alpha=0.8,
                 label="total mean score")
    axes[1].set_xlabel("distance $k$")
    axes[1].set_ylabel("mean value")
    axes[1].set_title("where the decay comes from")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_attention_decay.png")
    plt.close(fig)

    m = vals.mean(0)
    print(f"[实验3] center={center}: mean score k=0:{m[0]:.3f} -> k={max_k}:{m[-1]:.3f}; "
          f"std 范围=[{vals.std(0).min():.3f}, {vals.std(0).max():.3f}]")


# ---------------------------------------------------------------------------
# 实验 4：相对性 —— 取一个中心点向两侧扩散
# ---------------------------------------------------------------------------
def exp4_relative(d: int = 128, centers=(32, 64, 96), radius: int = 32) -> None:
    """固定中心 c，计算 PE_c · PE_(c+δ)，δ ∈ [-radius, radius]。

    - 曲线关于 δ=0 对称：sim(c, c-δ) = sim(c, c+δ) => 无法区分左右（无方向性）；
    - 不同中心 c 的三条曲线完全重合 => 只依赖相对偏移（平移不变）。
    """
    slen = max(centers) + radius + 1
    pe = get_pe(slen, d).double()
    deltas = torch.arange(-radius, radius + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # (a) 三个中心的曲线应该完全重合，且左右对称
    for c, marker in zip(centers, ["o", "s", "^"]):
        sim = torch.stack([pe[c] @ pe[c + dd] for dd in deltas]).numpy()
        axes[0].plot(deltas.numpy(), sim, marker=marker, ms=3, lw=1.2,
                     label=f"center={c}")
    axes[0].axvline(0, color="k", ls=":", lw=1)
    axes[0].set_xlabel(r"offset $\delta$ (signed)")
    axes[0].set_ylabel(r"$PE_c \cdot PE_{c+\delta}$")
    axes[0].set_title("all centers collapse onto one symmetric curve")
    axes[0].legend(fontsize=7)

    # (b) 行=中心，列=偏移：每一行都一样 => 与绝对位置无关
    band = torch.stack([torch.stack([pe[c] @ pe[c + dd] for dd in deltas])
                        for c in range(radius, slen - radius)])
    im = axes[1].imshow(band.numpy(), aspect="auto", cmap="viridis",
                        extent=[-radius, radius, band.shape[0] - 1, 0])
    axes[1].set_xlabel(r"offset $\delta$")
    axes[1].set_ylabel("center position")
    axes[1].set_title("rows are identical: only $\\delta$ matters")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_relative.png")
    plt.close(fig)

    # 数值检查：对称性与平移不变性
    c = centers[0]
    asym = max(abs((pe[c] @ pe[c + dd] - pe[c] @ pe[c - dd]).item())
               for dd in range(1, radius + 1))
    shift = max(abs((pe[c] @ pe[c + dd] - pe[c + 1] @ pe[c + 1 + dd]).item())
                for dd in range(-radius, radius + 1)
                if c + 1 + dd < slen)
    print(f"[实验4] 左右对称 max|Δ|={asym:.2e}; "
          f"平移 1 位后 max|Δ|={shift:.2e}")


def main() -> None:
    exp1_distribution()
    exp2_dot_product()
    exp3_attention_decay()
    exp4_relative()
    print(f"\n图已保存到 {FIG_DIR}")


if __name__ == "__main__":
    main()
