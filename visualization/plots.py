"""
Módulo de visualização para análise de linearidade do PA.

Gráficos incluídos:
    1. Diagrama de Constelação (IQ Plot)
       - Compara sinal original, Modelo semente do PA e Modelo otimizado do PA.
    2. Curva AM-AM
       - Ganho (|Y|/|X|) vs amplitude de entrada |X|.
       - Um PA linear > curva horizontal.
    3. Curva AM-PM
       - Rotação de fase (∠Y - ∠X) vs amplitude de entrada |X|.
       - Um PA linear > curva horizontal em 0°.
    4. Curva AM-AM (entrada-saída)
        - Amplitude da saída |Y| vs entrada |X|
        - Um pA linear > curva afim passando pela origem.
    4. Evolução do RMSE pelo GA
       - Mostra a convergência da otimização.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional
from deap import tools as deap_tools

# --- Paleta de cores
_COLORS = {
    "original"   : "#2196F3",  # azul
    "seed"       : "#F44336",  # vermelho
    "optimized"  : "#4CAF50",  # verde
    "accent"     : "#FF9800",  # laranja (convergência GA)
    "grid"       : "#E0E0E0",
    "background" : "#FAFAFA",
}

# --- Diagrama de Constelação
def plot_constellation(
    X_original   : np.ndarray,
    Y_seed       : np.ndarray,
    Y_optimized  : np.ndarray,
    title_suffix : str = "",
    max_points   : int = 2000,
    save_path    : Optional[str] = None,
) -> plt.Figure  :
    """
    Plota o Diagrama de Constelação IQ comparando três sinais.

    O diagrama IQ mostra a parte real (eixo I) vs a parte imaginária (eixo Q) de cada amostra.
    Num PA ideal:
        - Os pontos de Y devem coincidir com os de X (apenas escalados por G).
    Distorções não-lineares causam espalhamento (EVM alto) e rotação dos pontos.

    :parameter
    X_original   : sinal de entrada ideal (referência)
    Y_seed       : saída do PA não-linear (modelo original)
    Y_optimized  : saída com betas otimizados pelo GA
    max_points   : subsample para não sobrecarregar o plot
    save_path    : caminho para salvar a figura (opcional)
    """
    # Subsample para visualização
    idx = np.linspace(0, len(X_original) - 1, min(max_points, len(X_original)), dtype=int)
    Xo  = X_original[idx]
    Ysd  = Y_seed[idx]
    Yopt  = Y_optimized[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=_COLORS["background"])
    fig.suptitle(f"Diagrama de Constelação IQ {title_suffix}", fontsize=14, fontweight="bold")

    datasets = [
        (Xo, "Entrada X[n]\n(Referência)", _COLORS["original"], axes[0]),
        (Ysd, "PA Semente\n(Modelo Base)", _COLORS["seed"], axes[1]),
        (Yopt, "PA Otimizado\n(Modelo GA)", _COLORS["optimized"], axes[2]),
    ]

    for signal, label, color, ax in datasets:
        ax.scatter(
            signal.real, signal.imag,
            c=color, alpha=0.4, s=5, linewidths=0,
            label=label,
        )
        # Centraliza os eixos simetricamente
        lim = max(np.abs(signal).max() * 1.1, 1e-6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("I (Real)", fontsize=10)
        ax.set_ylabel("Q (Imag)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, color=_COLORS["grid"], linewidth=0.5)
        ax.set_facecolor(_COLORS["background"])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)

        # EVM estimado (quanto os pontos se espalharam)
        if label != "Entrada X[n]\n(Referência)":
            evm_rms = _compute_evm(Xo, signal)
            ax.text(
                0.05, 0.95, f"EVM: {evm_rms:.2f}%",
                transform=ax.transAxes,
                fontsize=9, color="black",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Constelação salva em: {save_path}")

    return fig

def _compute_evm(X_ref: np.ndarray, Y_test: np.ndarray) -> float:
    """
    EVM (Error Vector Magnitude) em %.

    Normaliza pelo PA esperado (X escalado pelo ganho médio) para ter uma métrica relativa ao nível do sinal.

        EVM_rms = sqrt( E[|e|²] / E[|X_scaled|²] ) * 100%
    """
    # Ganho médio (estimativa)
    G_mean = np.mean(np.abs(Y_test)) / (np.mean(np.abs(X_ref)) + 1e-10)
    X_scaled = X_ref * G_mean
    error = Y_test - X_scaled
    evm = np.sqrt(np.mean(np.abs(error)**2) / (np.mean(np.abs(X_scaled)**2) + 1e-10))
    return float(evm * 100.0)

# --- Curvas AM-AM e AM-PM
def plot_am_am_pm(
    X:            np.ndarray,
    Y_seed:  np.ndarray,
    Y_optimized: np.ndarray,
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Plota as curvas AM-AM e AM-PM do PA.

    AM-AM (Amplitude-Amplitude):
        - Mostra a compressão de ganho.
        - Eixo X: amplitude da entrada |X[n]|
        - Eixo Y: amplitude da saída |Y[n]|
        - Linear ideal > linha reta

    AM-PM (Amplitude-Phase):
        - Mostra a rotação de fase induzida pela não-linearidade.
        - Eixo X: amplitude da entrada |X[n]|
        - Eixo Y: ∠Y[n] - ∠X[n] (em graus)
        - Linear ideal > linha horizontal em 0°
    """
    amp_in = np.abs(X)
    gain_d = np.abs(Y_seed) / (amp_in + 1e-10)
    gain_l = np.abs(Y_optimized) / (amp_in + 1e-10)

    phase_d = np.angle(Y_seed, deg=True) - np.angle(X, deg=True)
    phase_l = np.angle(Y_optimized, deg=True) - np.angle(X, deg=True)
    # Normaliza para [-180, 180]
    phase_d = ((phase_d + 180) % 360) - 180
    phase_l = ((phase_l + 180) % 360) - 180

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=_COLORS["background"])
    fig.suptitle("Caracterização do PA: AM-AM e AM-PM", fontsize=14, fontweight="bold")

    # --- AM-AM
    ax1.scatter(amp_in, gain_d, c=_COLORS["seed"], alpha=0.3, s=3, label="Semente")
    ax1.scatter(amp_in, gain_l, c=_COLORS["optimized"], alpha=0.3, s=3, label="Otimizado")
    ax1.set_xlabel("|X[n]| — Amplitude de Entrada", fontsize=10)
    ax1.set_ylabel("|Y[n]| / |X[n]| — Ganho", fontsize=10)
    ax1.set_title("AM-AM (Compressão de Ganho)", fontsize=11, fontweight="bold")
    ax1.legend(markerscale=3)
    ax1.grid(True, color=_COLORS["grid"])
    ax1.set_facecolor(_COLORS["background"])

    # ── AM-PM ────────────────────────────────────────────────
    ax2.scatter(amp_in, phase_d, c=_COLORS["seed"], alpha=0.3, s=3, label="Semente")
    ax2.scatter(amp_in, phase_l, c=_COLORS["optimized"], alpha=0.3, s=3, label="Otimizado")
    ax2.axhline(0, color="black", linewidth=1, linestyle="--", label="Ideal (0°)")
    ax2.set_xlabel("|X[n]| — Amplitude de Entrada", fontsize=10)
    ax2.set_ylabel("∠Y[n] - ∠X[n] (graus)", fontsize=10)
    ax2.set_title("AM-PM (Rotação de Fase)", fontsize=11, fontweight="bold")
    ax2.legend(markerscale=3)
    ax2.grid(True, color=_COLORS["grid"])
    ax2.set_facecolor(_COLORS["background"])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"AM-AM/AM-PM salvo em: {save_path}")

    return fig

# --- Convergência do GA
def plot_ga_convergence(
    logbook:   deap_tools.Logbook,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota a evolução do RMSE ao longo das gerações do GA.

    Mostra as curvas de mínimo, média e desvio padrão, permitindo avaliar:
        - Velocidade de convergência
        - Diversidade da população (std alto = ainda explorando)
        - Estagnação prematura (plateau)
    """
    gen    = logbook.select("gen")
    rm_min = logbook.select("min")
    rm_avg = logbook.select("avg")
    rm_std = logbook.select("std")

    rm_min = np.array(rm_min)
    rm_avg = np.array(rm_avg)
    rm_std = np.array(rm_std)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_COLORS["background"])
    ax.set_facecolor(_COLORS["background"])

    ax.plot(gen, rm_min, color=_COLORS["optimized"], linewidth=2, label="RMSE mínimo (melhor)")
    ax.plot(gen, rm_avg, color=_COLORS["original"],   linewidth=1.5,
            linestyle="--", label="RMSE médio (população)")
    ax.fill_between(
        gen,
        np.maximum(rm_avg - rm_std, 0),
        rm_avg + rm_std,
        alpha=0.15, color=_COLORS["original"], label="±1 std"
    )

    ax.set_xlabel("Geração", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title("Convergência do Algoritmo Genético — Linearização do PA",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, color=_COLORS["grid"], linewidth=0.5)

    # Anota o melhor valor final
    best_rmse = rm_min[-1]
    ax.annotate(
        f"  Melhor RMSE:\n  {best_rmse:.6f}",
        xy=(gen[-1], best_rmse),
        xytext=(gen[-1] * 0.7, best_rmse * 1.5),
        arrowprops=dict(arrowstyle="->", color=_COLORS["accent"]),
        fontsize=10, color=_COLORS["accent"], fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Convergência GA salva em: {save_path}")

    return fig

# --- Dashboard
def plot_full_dashboard(
    X:            np.ndarray,
    Y_seed:  np.ndarray,
    Y_optimized: np.ndarray,
    logbook:      Optional[deap_tools.Logbook] = None,
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Painel completo com constelação, AM-AM, AM-PM e convergência GA.
    """
    n_rows = 2 if logbook else 1
    fig = plt.figure(figsize=(16, 10 if logbook else 5), facecolor=_COLORS["background"])
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.4, wspace=0.35)

    #  Linha 1: Constelações
    idx = np.linspace(0, len(X) - 1, min(1500, len(X)), dtype=int)

    for col, (sig, label, color) in enumerate(zip(
        [X[idx], Y_seed[idx], Y_optimized[idx]],
        ["Entrada X[n]", "PA Distorcido", "PA Linearizado"],
        [_COLORS["original"], _COLORS["seed"], _COLORS["optimized"]],
    )):
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(sig.real, sig.imag, c=color, alpha=0.4, s=4, linewidths=0)
        lim = max(np.abs(sig).max() * 1.1, 1e-6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.grid(True, color=_COLORS["grid"], linewidth=0.4)
        ax.set_facecolor(_COLORS["background"])

    #  AM-AM (último coluna)
    ax_am = fig.add_subplot(gs[0, 3])
    amp = np.abs(X)
    ax_am.scatter(amp, np.abs(Y_seed),  c=_COLORS["seed"],  alpha=0.3, s=3, label="Semente")
    ax_am.scatter(amp, np.abs(Y_optimized), c=_COLORS["optimized"], alpha=0.3, s=3, label="Otimizado")
    ax_am.set_title("AM-AM", fontsize=10, fontweight="bold")
    ax_am.set_xlabel("|X[n]|"); ax_am.set_ylabel("|Y[n]|")
    ax_am.legend(markerscale=3, fontsize=8)
    ax_am.grid(True, color=_COLORS["grid"], linewidth=0.4)
    ax_am.set_facecolor(_COLORS["background"])

    #  Linha 2: Convergência GA
    if logbook:
        ax_ga = fig.add_subplot(gs[1, :])
        gen    = np.array(logbook.select("gen"))
        rm_min = np.array(logbook.select("min"))
        rm_avg = np.array(logbook.select("avg"))
        rm_std = np.array(logbook.select("std"))
        ax_ga.plot(gen, rm_min, color=_COLORS["seed"], linewidth=2, label="RMSE min")
        ax_ga.plot(gen, rm_avg, color=_COLORS["original"], linewidth=1.5,
                   linestyle="--", label="RMSE médio")
        ax_ga.fill_between(gen, np.maximum(rm_avg - rm_std, 0), rm_avg + rm_std,
                           alpha=0.15, color=_COLORS["original"])
        ax_ga.set_xlabel("Geração"); ax_ga.set_ylabel("RMSE")
        ax_ga.set_title("Convergência do GA", fontsize=10, fontweight="bold")
        ax_ga.legend(fontsize=9)
        ax_ga.grid(True, color=_COLORS["grid"], linewidth=0.4)
        ax_ga.set_facecolor(_COLORS["background"])

    fig.suptitle("PA Linearization — Dashboard de Análise", fontsize=15, fontweight="bold", y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard salvo em: {save_path}")

    return fig