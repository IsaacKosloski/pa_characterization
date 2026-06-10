"""
============================================================
gerar_relatorio.py
============================================================
Lê o pa_linearization_results.csv (ou gera a partir dos dados
brutos) e produz:
  1. Um relatório de métricas em texto (.txt)
  2. Um painel de gráficos com as visualizações mais relevantes (.png)

As métricas mais relevantes para avaliação de um modelo de PA:
  - RMSE   : erro absoluto médio (raiz do erro quadrático)
  - NMSE   : erro normalizado em dB (métrica padrão da indústria)
  - EVM    : erro relativo em % (ligado à qualidade de modulação)
  - Ganho  : amplificação média em dB
  - Planicidade de ganho (std) : quão linear é a curva AM-AM
  - Planicidade de fase (std)  : quão linear é a curva AM-PM

Uso:
    python gerar_relatorio.py                      # usa dados padrão
    python gerar_relatorio.py caminho/results.csv  # usa um CSV existente
============================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem display (salva em arquivo)
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
#  Cálculo das métricas
# ─────────────────────────────────────────────────────────────
def calcular_metricas(X, Y_true, Y_pred):
    """
    Calcula todas as métricas relevantes a partir dos sinais complexos.

    Parâmetros
    ----------
    X      : np.ndarray complex — entrada
    Y_true : np.ndarray complex — saída medida (ground truth)
    Y_pred : np.ndarray complex — saída prevista pelo modelo

    Retorna
    -------
    dict com todas as métricas
    """
    err = Y_true - Y_pred

    # RMSE — raiz do erro quadrático médio
    rmse = float(np.sqrt(np.mean(np.abs(err) ** 2)))

    # NMSE — erro normalizado pela potência do sinal, em dB
    mse = np.mean(np.abs(err) ** 2)
    pwr = np.mean(np.abs(Y_true) ** 2)
    nmse_dB = float(10 * np.log10(mse / (pwr + 1e-15)))

    # EVM — erro relativo em %
    evm = float(np.sqrt(mse / (pwr + 1e-15)) * 100)

    # Ganho médio (magnitude) em dB
    amp_in = np.abs(X) + 1e-12
    ganho_true = np.abs(Y_true) / amp_in
    ganho_pred = np.abs(Y_pred) / amp_in
    ganho_dB = float(20 * np.log10(np.mean(ganho_true)))

    # Planicidade de ganho — desvio padrão do ganho em dB
    ganho_true_dB = 20 * np.log10(ganho_true + 1e-15)
    gain_flatness_std = float(np.std(ganho_true_dB))

    # Planicidade de fase — desvio padrão da rotação de fase
    fase = np.angle(Y_true, deg=True) - np.angle(X, deg=True)
    fase = ((fase + 180) % 360) - 180  # normaliza [-180, 180]
    phase_flatness_std = float(np.std(fase))

    # PAPR do sinal de entrada
    papr_dB = float(10 * np.log10(np.max(np.abs(X) ** 2) / np.mean(np.abs(X) ** 2)))

    return {
        "n_amostras":          len(X),
        "rmse":                rmse,
        "nmse_dB":             nmse_dB,
        "evm_pct":             evm,
        "ganho_dB":            ganho_dB,
        "gain_flatness_std":   gain_flatness_std,
        "phase_flatness_std":  phase_flatness_std,
        "papr_dB":             papr_dB,
        "amp_max_in":          float(np.max(np.abs(X))),
        "amp_max_out":         float(np.max(np.abs(Y_true))),
    }


def formula_narx(cfg=None):
    """
    Monta a fórmula explícita do modelo NARX, associando cada beta ao
    seu regressor em phi(X), na MESMA ordem usada por NARXEngine.

    A saída Y(t) tem duas componentes (real e imaginária):

        Yreal(t) = c_real + [A1·Y(t-1) + A2·Y(t-2)]_real + SUM(beta_real_k · phi_k)
        Yimg(t)  = c_img  + [A1·Y(t-1) + A2·Y(t-2)]_img  + SUM(beta_img_k  · phi_k)

    phi tem 18 regressores. IMPORTANTE: a ordem do vetor de betas no
    NARXEngine NÃO é a mesma de phi — o produto interno bv·phi associa
    cada beta à posição correspondente. Aqui reconstruímos a associação
    posição-a-posição corretamente.

    Parâmetros
    ----------
    cfg : NARXModelConfig | None — se fornecido, usa os betas reais; senão,
          carrega o modelo semente padrão.

    Retorna
    -------
    str com a fórmula formatada em texto.
    """
    if cfg is None:
        sys.path.insert(0, str(Path(__file__).parent))
        from core.model_config import NARXModelConfig
        cfg = NARXModelConfig()
        _semente = True
    else:
        _semente = False

    b = cfg.betas
    c = cfg.intercepts
    A1 = cfg.A1; A2 = cfg.A2

    # phi na ordem do _build_regressors:
    #  0: xr0    1: xi0    2: xr1    3: xi1    4: xr2    5: xi2
    #  6: xr0^2  7: xr1^2  8: xr2^2  9: xi0^2 10: xi1^2 11: xi2^2
    # 12: xr0^3 13: xr1^3 14: xr2^3 15: xi0^3 16: xi1^3 17: xi2^3
    regressores = [
        "Xr(t)",   "Xi(t)",   "Xr(t-1)", "Xi(t-1)", "Xr(t-2)", "Xi(t-2)",
        "Xr(t)^2", "Xr(t-1)^2","Xr(t-2)^2","Xi(t)^2","Xi(t-1)^2","Xi(t-2)^2",
        "Xr(t)^3", "Xr(t-1)^3","Xr(t-2)^3","Xi(t)^3","Xi(t-1)^3","Xi(t-2)^3",
    ]

    # vetor de betas para Yreal — MESMA ordem de _beta_vector_yreal()
    bv_real = [
        b.Xreal_Yreal,        b.Ximg_Yreal,
        b.Xreal_lag1_Yreal,   b.Ximg_lag1_Yreal,
        b.Xreal_lag2_Yreal,   b.Ximg_lag2_Yreal,
        b.Xreal_deg2_Yreal,   b.Xreal_lag1_deg2_Yreal,  b.Xreal_lag2_deg2_Yreal,
        b.Ximg_deg2_Yreal,    b.Ximg_lag1_deg2_Yreal,   b.Ximg_lag2_deg2_Yreal,
        b.Xreal_deg3_Yreal,   b.Xreal_lag1_deg3_Yreal,  b.Xreal_lag2_deg3_Yreal,
        b.Ximg_deg3_Yreal,    b.Ximg_lag1_deg3_Yreal,   b.Ximg_lag2_deg3_Yreal,
    ]
    bv_img = [
        b.Xreal_Yimg,         b.Ximg_Yimg,
        b.Xreal_lag1_Yimg,    b.Ximg_lag1_Yimg,
        b.Xreal_lag2_Yimg,    b.Ximg_lag2_Yimg,
        b.Xreal_deg2_Yimg,    b.Xreal_lag1_deg2_Yimg,   b.Xreal_lag2_deg2_Yimg,
        b.Ximg_deg2_Yimg,     b.Ximg_lag1_deg2_Yimg,    b.Ximg_lag2_deg2_Yimg,
        b.Xreal_deg3_Yimg,    b.Xreal_lag1_deg3_Yimg,   b.Xreal_lag2_deg3_Yimg,
        b.Ximg_deg3_Yimg,     b.Ximg_lag1_deg3_Yimg,    b.Ximg_lag2_deg3_Yimg,
    ]

    def termo(coef, reg):
        sinal = "+" if coef >= 0 else "-"
        return f"{sinal} {abs(coef):.6g}*{reg}"

    def bloco(bv, nome_saida):
        L = [f"  {nome_saida}(t) ="]
        # intercepto
        ic = c.Yreal if nome_saida == "Yr" else c.Yimg
        L.append(f"      {ic:+.6g}                                  [intercepto]")
        # parte autorregressiva
        if nome_saida == "Yr":
            L.append(f"      {termo(A1.Yreal_to_Yreal,'Yr(t-1)')} {termo(A1.Yimg_to_Yreal,'Yi(t-1)')}   [A1]")
            L.append(f"      {termo(A2.Yreal_to_Yreal,'Yr(t-2)')} {termo(A2.Yimg_to_Yreal,'Yi(t-2)')}   [A2]")
        else:
            L.append(f"      {termo(A1.Yreal_to_Yimg,'Yr(t-1)')} {termo(A1.Yimg_to_Yimg,'Yi(t-1)')}   [A1]")
            L.append(f"      {termo(A2.Yreal_to_Yimg,'Yr(t-2)')} {termo(A2.Yimg_to_Yimg,'Yi(t-2)')}   [A2]")
        # regressores exogenos por grupo
        grupos = [("LINEAR (deg=1)", 0, 6), ("QUADRATICO (deg=2)", 6, 12), ("CUBICO (deg=3)", 12, 18)]
        for titulo, ini, fim in grupos:
            L.append(f"      --- {titulo} ---")
            for k in range(ini, fim):
                L.append(f"      {termo(bv[k], regressores[k])}")
        return "\n".join(L)

    out = []
    out.append("=" * 60)
    out.append("  FORMULA DO MODELO NARX (betas atuais)")
    out.append("=" * 60)
    out.append("")
    if cfg is None or _semente:
        out.append("  [!] AVISO: fórmula exibida com os betas do MODELO SEMENTE.")
        out.append("      Se este CSV veio do GA (coluna Ypred_opt), os betas reais")
        out.append("      que o geraram podem diferir. Para a fórmula exata do modelo")
        out.append("      otimizado, salve os betas do GA e use formula_narx(best_config).")
        out.append("")
    out.append("  Estrutura geral:")
    out.append("    Y(t) = c + A1*Y(t-1) + A2*Y(t-2) + SUM(beta_k * phi_k(X))")
    out.append("")
    out.append("  Onde phi(X) tem 18 regressores (real/imag x lags 0,1,2 x graus 1,2,3)")
    out.append("  e Xr = parte real, Xi = parte imaginaria do sinal de entrada.")
    out.append("")
    out.append(bloco(bv_real, "Yr"))
    out.append("")
    out.append(bloco(bv_img, "Yi"))
    out.append("")
    out.append("=" * 60)
    return "\n".join(out)


def classificar(metricas):
    """Atribui um veredito qualitativo a cada métrica conforme padrões da indústria."""
    def veredito(valor, otimo, bom):
        if valor <= otimo:  return "EXCELENTE"
        if valor <= bom:    return "BOM"
        return "A MELHORAR"

    # NMSE: mais negativo = melhor. Trabalhamos com o valor absoluto do dB.
    # Ex: -27.76 dB -> 27.76. Excelente se >=40, Bom se >=30.
    def veredito_nmse(nmse_dB):
        mag = -nmse_dB  # 27.76
        if mag >= 40:  return "EXCELENTE"
        if mag >= 30:  return "BOM"
        return "A MELHORAR"

    return {
        "nmse":  veredito_nmse(metricas["nmse_dB"]),
        "evm":   veredito(metricas["evm_pct"], 1.5, 3.5),
        "gain":  veredito(metricas["gain_flatness_std"], 0.1, 0.5),
        "phase": veredito(metricas["phase_flatness_std"], 0.5, 2.0),
    }


# ─────────────────────────────────────────────────────────────
#  Relatório em texto
# ─────────────────────────────────────────────────────────────
def gerar_txt(metricas, classes, destino, formula=""):
    """Escreve o relatório de métricas num arquivo .txt legível."""
    L = []
    L.append("=" * 60)
    L.append("  RELATORIO DE METRICAS — MODELO NARX DO PA")
    L.append("=" * 60)
    L.append("")
    L.append(f"  Amostras analisadas : {metricas['n_amostras']:,}")
    L.append(f"  PAPR do sinal       : {metricas['papr_dB']:.2f} dB")
    L.append(f"  Amplitude max (in)  : {metricas['amp_max_in']:.4f}")
    L.append(f"  Amplitude max (out) : {metricas['amp_max_out']:.4f}")
    L.append("")
    L.append("-" * 60)
    L.append(f"  {'METRICA':<28}{'VALOR':>14}{'VEREDITO':>16}")
    L.append("-" * 60)
    L.append(f"  {'RMSE (erro absoluto)':<28}{metricas['rmse']:>14.5f}{'':>16}")
    L.append(f"  {'NMSE (erro normalizado)':<28}{metricas['nmse_dB']:>11.2f} dB{classes['nmse']:>16}")
    L.append(f"  {'EVM (erro relativo)':<28}{metricas['evm_pct']:>12.2f} %{classes['evm']:>16}")
    L.append(f"  {'Ganho medio':<28}{metricas['ganho_dB']:>11.2f} dB{'':>16}")
    L.append(f"  {'Planicidade ganho (std)':<28}{metricas['gain_flatness_std']:>11.3f} dB{classes['gain']:>16}")
    L.append(f"  {'Planicidade fase (std)':<28}{metricas['phase_flatness_std']:>12.3f}{chr(176)}{classes['phase']:>15}")
    L.append("-" * 60)
    L.append("")
    L.append("  REFERENCIA DE QUALIDADE (padroes 3GPP / industria):")
    L.append("    NMSE  : EXCELENTE < -40 dB | BOM < -30 dB")
    L.append("    EVM   : EXCELENTE < 1.5%   | BOM < 3.5% (256-QAM)")
    L.append("    Ganho : EXCELENTE std<0.1dB| BOM std<0.5 dB")
    L.append("    Fase  : EXCELENTE std<0.5° | BOM std<2.0°")
    L.append("")
    L.append("=" * 60)

    if formula:
        L.append("")
        L.append(formula)

    texto = "\n".join(L)
    Path(destino).write_text(texto, encoding="utf-8")
    print(texto)
    return texto


# ─────────────────────────────────────────────────────────────
#  Painel de gráficos
# ─────────────────────────────────────────────────────────────
def gerar_plot(X, Y_true, Y_pred, metricas, destino):
    """Gera um painel 2x2 com as visualizações mais relevantes."""
    idx = np.linspace(0, len(X) - 1, min(3000, len(X)), dtype=int)
    Xs, Yt, Yp = X[idx], Y_true[idx], Y_pred[idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="white")
    fig.suptitle("Painel de Métricas — Modelo NARX do PA", fontsize=15, fontweight="bold")

    # [1] Constelação: medido vs previsto
    ax = axes[0, 0]
    ax.scatter(Yt.real, Yt.imag, c="#2196F3", s=5, alpha=0.4, label="Medido")
    ax.scatter(Yp.real, Yp.imag, c="#F44336", s=5, alpha=0.4, label="Previsto")
    ax.set_title("Constelação IQ — Medido vs Previsto", fontweight="bold")
    ax.set_xlabel("I (real)"); ax.set_ylabel("Q (imag)")
    ax.legend(markerscale=3); ax.grid(True, alpha=0.3); ax.set_aspect("equal")

    # [2] AM-AM: ganho vs amplitude de entrada
    ax = axes[0, 1]
    amp = np.abs(Xs)
    ax.scatter(amp, np.abs(Yt) / (amp + 1e-12), c="#2196F3", s=4, alpha=0.4, label="Medido")
    ax.scatter(amp, np.abs(Yp) / (amp + 1e-12), c="#F44336", s=4, alpha=0.4, label="Previsto")
    ax.set_title(f"AM-AM — Ganho vs Amplitude (std={metricas['gain_flatness_std']:.3f} dB)", fontweight="bold")
    ax.set_xlabel("|X[n]|"); ax.set_ylabel("Ganho |Y|/|X|")
    ax.legend(markerscale=3); ax.grid(True, alpha=0.3)

    # [3] Erro ponto-a-ponto ao longo do tempo
    ax = axes[1, 0]
    erro = np.abs(Y_true - Y_pred)
    ax.plot(erro[:2000], c="#FF9800", lw=0.7)
    ax.axhline(erro.mean(), color="#4CAF50", ls="--", lw=1.5, label=f"Erro médio: {erro.mean():.3f}")
    ax.set_title("Erro absoluto |Y_medido − Y_previsto|", fontweight="bold")
    ax.set_xlabel("Amostra"); ax.set_ylabel("Erro")
    ax.legend(); ax.grid(True, alpha=0.3)

    # [4] Tabela-resumo de métricas
    ax = axes[1, 1]; ax.axis("off")
    linhas = [
        ["Amostras",            f"{metricas['n_amostras']:,}"],
        ["RMSE",                f"{metricas['rmse']:.5f}"],
        ["NMSE",                f"{metricas['nmse_dB']:.2f} dB"],
        ["EVM",                 f"{metricas['evm_pct']:.2f} %"],
        ["Ganho médio",         f"{metricas['ganho_dB']:.2f} dB"],
        ["Planicidade ganho",   f"{metricas['gain_flatness_std']:.3f} dB"],
        ["Planicidade fase",    f"{metricas['phase_flatness_std']:.3f}°"],
        ["PAPR entrada",        f"{metricas['papr_dB']:.2f} dB"],
    ]
    tab = ax.table(cellText=linhas, colLabels=["Métrica", "Valor"],
                   loc="center", cellLoc="left", colWidths=[0.55, 0.4])
    tab.auto_set_font_size(False); tab.set_fontsize(12); tab.scale(1, 2.2)
    for j in range(2):
        tab[(0, j)].set_facecolor("#534AB7")
        tab[(0, j)].set_text_props(color="white", fontweight="bold")
    ax.set_title("Resumo das Métricas", fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    print(f"\nPainel de gráficos salvo em: {destino}")


# ─────────────────────────────────────────────────────────────
#  Carregamento de dados
# ─────────────────────────────────────────────────────────────
def carregar(csv_path):
    """
    Carrega os sinais. Aceita:
      - pa_linearization_results.csv (com colunas Ypred_opt_*)
      - CSV bruto (Xreal, Ximg, Yreal, Yimg)
    """
    df = pd.read_csv(csv_path)
    X = (df["Xreal"] + 1j * df["Ximg"]).to_numpy()
    Y_true = (df["Yreal"] + 1j * df["Yimg"]).to_numpy()

    # Se já tem predição otimizada, usa-a; senão calcula com o modelo semente
    cfg_usada = None
    if "Ypred_opt_real" in df.columns:
        Y_pred = (df["Ypred_opt_real"] + 1j * df["Ypred_opt_img"]).to_numpy()
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from core.model_config import NARXModelConfig
        from core.narx_engine import NARXEngine
        cfg_usada = NARXModelConfig()
        Y_pred = NARXEngine(cfg_usada).predict(X)

    return X, Y_true, Y_pred, cfg_usada


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "pa_linearization_results.csv"
    out_dir = Path("./relatorio")
    out_dir.mkdir(exist_ok=True)

    X, Y_true, Y_pred, cfg_usada = carregar(csv_path)
    metricas = calcular_metricas(X, Y_true, Y_pred)
    classes = classificar(metricas)
    formula = formula_narx(cfg_usada)

    gerar_txt(metricas, classes, out_dir / "relatorio_metricas.txt", formula)
    gerar_plot(X, Y_true, Y_pred, metricas, out_dir / "painel_metricas.png")
