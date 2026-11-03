"""RoPE 的三组实验：相似度、Attention Score 影响、远程衰减。

约束：
1. 只 import llm_arch/position_encoding/rope.py 里的 RoPE，不修改它；
2. rope.py 的 forward 里有调试 print，这里用临时替换 builtins.print 的方式静默，
   文件本身一行没动；
3. 所有旋转结果都来自 RoPE 模块，公式只用于解释和画理论曲线。

运行：
    MPLCONFIGDIR=/tmp python llm_arch/position_encoding/exp_rope.py

输出（llm_arch/position_encoding/figs_rope/）：
    fig1_rope_similarity.png  固定向量旋转后的 Gram 矩阵与相似度衰减
    fig2_attention_score.png  有/无 RoPE 的 score 矩阵，以及 score 随距离的振荡
    fig3_decay_mechanism.png  衰减来源：各频率相位失配 + 频段分解 + 对齐/随机内容对比
    fig4_bound.png            博客里 Σ|S_j| 上界随距离的下降
"""

import builtins
import contextlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rope import RoPE

FIG_DIR = Path(__file__).resolve().parent / "figs_rope"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

_rope = RoPE()


# ---------------------------------------------------------------------------
# 调用 RoPE：屏蔽它 forward 里的 print（rope.py 不能改，只能从调用侧处理）
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _silent():
    original = builtins.print
    builtins.print = lambda *args, **kwargs: None
    try:
        yield
    finally:
        builtins.print = original


def rotate(X: torch.Tensor) -> torch.Tensor:
    """对 [s, d] 的每一行做 RoPE，第 t 行按位置 t 旋转。"""
    with torch.no_grad(), _silent():
        return _rope(X.contiguous())


def rotate_all(vec: torch.Tensor, slen: int) -> torch.Tensor:
    """把同一个向量复制到每一行，得到 R_t v，t = 0..slen-1。"""
    d = vec.numel()
    return rotate(vec.expand(slen, d))


# ---------------------------------------------------------------------------
# 基础表：每个频率对在位置差 Δ 上的 cos / sin
# ---------------------------------------------------------------------------
def frequency_tables(slen: int, d: int):
    """用单位基向量过一遍 RoPE，得到每对频率的 cos(Δθ_i)、sin(Δθ_i)。

    cos_tab[Δ, i] = cos(Δ θ_i)，sin_tab[Δ, i] = sin(Δ θ_i)。
    直接由 RoPE 的输出读取，避免在实验里重写公式。
    """
    cos_tab = torch.empty(slen, d // 2)
    sin_tab = torch.empty(slen, d // 2)
    for i in range(d // 2):
        X = torch.zeros(slen, d)
        X[:, 2 * i] = 1.0
        Y = rotate(X)
        cos_tab[:, i] = Y[:, 2 * i]
        sin_tab[:, i] = Y[:, 2 * i + 1]
    return cos_tab.double(), sin_tab.double()


# ---------------------------------------------------------------------------
# 实验 1：RoPE 本身的相似度
# ---------------------------------------------------------------------------
def exp1_similarity(slen: int = 512, d: int = 64, seed: int = 0) -> dict:
    """固定一个向量 x，看 R_t x 之间的相似度。

    (R_m x)·(R_n x) = xᵀ R_{n-m} x = Σ_i (x_{2i}²+x_{2i+1}²) cos((n-m)θ_i)
    因此 Gram 矩阵是 Toeplitz：只依赖相对距离，不依赖绝对位置。
    """
    torch.manual_seed(seed)
    x = torch.randn(d)
    Y = rotate_all(x, slen)
    gram = (Y @ Y.T).double()

    cos_tab, _ = frequency_tables(slen, d)
    w = (x[0::2] ** 2 + x[1::2] ** 2).double()      # 每个频率对的能量
    theory = cos_tab @ w                             # 理论相似度曲线

    sim = gram[0]
    norm0 = x.norm().item()
    norm_dev = (Y.norm(dim=1) - norm0).abs().max().item()
    toeplitz_dev = max((gram[i, j] - theory[abs(i - j)]).abs().item()
                       for i in range(slen) for j in range(slen))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    im = axes[0].imshow(gram.numpy(), cmap="viridis")
    axes[0].set_xlabel("position $n$")
    axes[0].set_ylabel("position $m$")
    axes[0].set_title(r"$(R_m x)\cdot(R_n x)$: constant along diagonals")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

    axes[1].plot(sim.numpy(), color="#1f4e79", lw=1.6,
                 label=r"empirical $(R_0x)\cdot(R_\Delta x)$")
    axes[1].plot(theory.numpy(), "r--", lw=1.2, label="theory $\\sum_i w_i\\cos(\\Delta\\theta_i)$")
    axes[1].axhline(norm0 ** 2, color="gray", ls=":", lw=1.2,
                    label=r"no RoPE: $\|x\|^2$ (constant)")
    axes[1].set_xlabel(r"relative distance $\Delta$")
    axes[1].set_ylabel("similarity")
    axes[1].set_title("similarity decays with distance")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_rope_similarity.png")
    plt.close(fig)

    print(f"[实验1] ||x||={norm0:.3f}, 旋转后范数最大偏差={norm_dev:.2e}")
    print(f"        Toeplitz(只依赖 Δ) 最大偏差={toeplitz_dev:.2e}")
    print(f"        sim(Δ=0)={sim[0]:.3f}, sim(Δ={slen//2})={sim[slen//2]:.3f}, "
          f"sim(Δ={slen-1})={sim[-1]:.3f}")
    return {"norm_dev": norm_dev, "toeplitz_dev": toeplitz_dev,
            "sim0": sim[0].item(), "sim_end": sim[-1].item()}


# ---------------------------------------------------------------------------
# 实验 2：对 Attention Score 的影响
# ---------------------------------------------------------------------------
def exp2_attention_score(slen: int = 64, d: int = 64, seed: int = 0) -> dict:
    """固定 q、k，比较有无 RoPE 的 score 矩阵。

    无 RoPE：S[m,n] = q·k，与位置完全无关（常数矩阵）。
    有 RoPE：S[m,n] = (R_m q)·(R_n k) = f(m-n)，变成 Toeplitz 结构。
    """
    torch.manual_seed(seed)
    q, k = torch.randn(d), torch.randn(d)

    S0 = (q * k).sum().item() * torch.ones(slen, slen)
    RQ = rotate_all(q, slen)
    RK = rotate_all(k, slen)
    S1 = (RQ @ RK.T).double()

    # 相对性检查：S1[m,n] 应只依赖 n-m
    rel_dev = max((S1[i, j] - S1[0, j - i]).abs().item()
                  for i in range(slen) for j in range(i, slen))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})

    vmin = min(S0.min().item(), S1.min().item())
    vmax = max(S0.max().item(), S1.max().item())
    for ax, mat, title in [
        (axes[0], S0, "no RoPE: $q\\cdot k$ (constant)"),
        (axes[1], S1, "RoPE: $(R_m q)\\cdot(R_n k)$"),
    ]:
        im = ax.imshow(mat.numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_xlabel("position $n$")
        ax.set_ylabel("position $m$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # 不同内容的 q、k：score 随距离振荡，衰减形状由内容决定
    for t in range(3):
        qq, kk = torch.randn(d), torch.randn(d)
        curve = (rotate_all(qq, slen)[0] * rotate_all(kk, slen)).sum(-1)
        axes[2].plot(curve.numpy(), lw=1.2, alpha=0.85, label=f"pair {t+1}")
    axes[2].axhline(0, color="k", ls=":", lw=1)
    axes[2].set_xlabel(r"relative distance $\Delta$")
    axes[2].set_ylabel("attention score")
    axes[2].set_title("score vs distance depends on content")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_attention_score.png")
    plt.close(fig)

    print(f"[实验2] 无 RoPE score 矩阵标准差={S0.std():.2e}（常数）")
    print(f"        有 RoPE score 范围=[{S1.min():.2f}, {S1.max():.2f}]，"
          f"只依赖 Δ 的最大偏差={rel_dev:.2e}")
    return {"rel_dev": rel_dev, "roPE_range": (S1.min().item(), S1.max().item())}


# ---------------------------------------------------------------------------
# 实验 3：远程衰减的来源
# ---------------------------------------------------------------------------
def exp3_decay_mechanism(slen: int = 512, d: int = 64, seed: int = 0) -> dict:
    """把 score 拆到每个频率对上，解释为什么距离变大后会衰减。

    q=k=1 时：score(Δ) = Σ_i 2cos(Δθ_i)。
    Δ=0 时所有项都等于 2，全部同相 -> 最大值 d；
    Δ 增大后不同 θ_i 的相位差不同，高频项先互相抵消，只剩低频项的残余。
    """
    torch.manual_seed(seed)
    cos_tab, _ = frequency_tables(slen, d)

    # (a) 每个频率的 cos(Δθ_i)：行=频率对，列=距离
    # (b) 按频率分成三段，看哪一段先衰减
    n_freq = d // 2
    bands = [torch.arange(0, n_freq // 3),
             torch.arange(n_freq // 3, 2 * n_freq // 3),
             torch.arange(2 * n_freq // 3, n_freq)]
    band_names = ["high freq (i small)", "mid freq", "low freq (i large)"]
    total = 2 * cos_tab.sum(-1)                      # q=k=ones 时的 score

    # (c) 对齐内容 vs 随机内容
    steps = torch.arange(0, slen, 16)
    aligned = []
    random_scores = []
    x = torch.randn(d)
    w = (x[0::2] ** 2 + x[1::2] ** 2).double()
    aligned_x = cos_tab @ w                          # q=k=x 的典型 score
    for Delta in steps.tolist():
        aligned.append(aligned_x[Delta].item())
        vals = []
        for _ in range(16):                          # 随机 q、k，取 |score| 的均值
            q, k = torch.randn(d), torch.randn(d)
            yq = rotate_all(q, Delta + 1)
            yk = rotate_all(k, Delta + 1)
            vals.append((yq[0] * yk[Delta]).sum().item())
        random_scores.append(float(np.mean(np.abs(vals))))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2),
                             gridspec_kw={"width_ratios": [1.1, 1, 1]})

    im = axes[0].imshow(cos_tab.T.numpy(), aspect="auto", cmap="RdBu_r",
                        vmin=-1, vmax=1, extent=[0, slen - 1, n_freq, 0])
    axes[0].set_xlabel(r"distance $\Delta$")
    axes[0].set_ylabel("frequency pair $i$")
    axes[0].set_title(r"$\cos(\Delta\theta_i)$: each frequency rotates at its own speed")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

    for band, name in zip(bands, band_names):
        axes[1].plot(2 * cos_tab[:, band].sum(-1).numpy(), lw=1.3, label=name)
    axes[1].plot(total.numpy(), "k--", lw=1.2, label="total ($q=k=1$)")
    axes[1].axhline(0, color="k", ls=":", lw=1)
    axes[1].set_xlabel(r"distance $\Delta$")
    axes[1].set_ylabel("contribution to score")
    axes[1].annotate("slow dims stay aligned\nwithin this range",
                     xy=(0.45, 0.62), xycoords="axes fraction", fontsize=6.5,
                     color="#2ca02c")
    axes[1].set_title("high-freq terms cancel first")
    axes[1].legend(fontsize=6.5)

    ones_curve = (2 * cos_tab.sum(-1))[steps].numpy()
    axes[2].plot(steps.numpy(), np.array(aligned), "o-", ms=3,
                 label=r"aligned content $q=k=x$ (decays)")
    axes[2].plot(steps.numpy(), ones_curve, "^-", ms=3,
                 label=r"$q=k=\mathbf{1}$: $\sum_i 2\cos(\Delta\theta_i)$")
    axes[2].plot(steps.numpy(), np.array(random_scores), "s-", ms=3,
                 label=r"random $q,k$: $\mathbb{E}|score|$ (flat)")
    axes[2].axhline(math.sqrt(d), color="gray", ls=":", lw=1,
                    label=r"$\sqrt{d}$")
    axes[2].set_xlabel(r"distance $\Delta$")
    axes[2].set_ylabel("score / mean $|score|$")
    axes[2].set_title("decay needs aligned content")
    axes[2].legend(fontsize=6.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_decay_mechanism.png")
    plt.close(fig)

    ratio_aligned = aligned[0] / max(abs(aligned[-1]), 1e-9)
    ratio_random = random_scores[0] / max(random_scores[-1], 1e-9)
    print(f"[实验3] q=k=1: score(0)={total[0]:.2f} -> score({slen-1})={total[-1]:.2f}")
    print(f"        对齐内容衰减倍数={ratio_aligned:.2f}; "
          f"随机 q,k 的 |score| 衰减倍数={ratio_random:.2f}")
    return {"aligned_ratio": ratio_aligned, "random_ratio": ratio_random,
            "score0": total[0].item(), "score_end": total[-1].item()}


# ---------------------------------------------------------------------------
# 实验 4：博客里 Σ|S_j| 的上界
# ---------------------------------------------------------------------------
def exp4_bound(d: int = 128, max_delta: int = 256) -> None:
    """S_j = Σ_{i<j} e^{iΔθ_i}，画 Σ_j |S_j| 随 Δ 的变化。

    这就是博客里 |Σ h_i e^{iΔθ_i}| ≤ max|h_{i+1}-h_i| · Σ|S_j| 的那个求和项：
    它随相对距离下降，说明上界在变小，但这是“上界”，不是每个具体 score 都单调衰减。
    """
    cos_tab, sin_tab = frequency_tables(2, d)
    theta = torch.atan2(sin_tab[1], cos_tab[1]).double()      # θ_i，从 RoPE 输出反推

    deltas = torch.arange(1, max_delta + 1, dtype=torch.float64)
    phases = deltas[:, None] * theta[None, :]
    S = torch.cumsum(torch.exp(1j * phases), dim=1)            # (Δ, d/2)
    bound = S.abs().sum(dim=1)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(deltas.numpy(), bound.numpy(), lw=1.6)
    ax.set_xlabel(r"relative distance $\Delta$")
    ax.set_ylabel(r"$\sum_j |S_j|$")
    ax.set_title("the upper-bound term decays with distance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_bound.png")
    plt.close(fig)

    print(f"[实验4] Σ|S_j|: Δ=1 -> {bound[0]:.1f}, Δ={max_delta} -> {bound[-1]:.1f}")


def main() -> None:
    exp1_similarity()
    exp2_attention_score()
    exp3_decay_mechanism()
    exp4_bound()
    print(f"\n图已保存到 {FIG_DIR}")


if __name__ == "__main__":
    main()
