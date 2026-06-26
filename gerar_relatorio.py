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

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem display (salva em arquivo)
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
#  Utilitário de Diretórios
# ─────────────────────────────────────────────────────────────
def create_run_directory(base_dir="./relatorio", prefix="relatorio"):
    """
    Cria um novo subdiretório numerado dentro de `base_dir`.

    Procura subpastas já existentes no formato {prefix}NN (ex.: relatorio01,
    relatorio02, ...), descobre o maior número usado e cria o próximo da
    sequência. Cada execução gera uma pasta nova e nunca sobrescreve os
    relatórios anteriores.

    Exemplo:
        Se já existem  relatorio/relatorio01  e  relatorio/relatorio02,
        a função cria e retorna  relatorio/relatorio03.

    :param base_dir : pasta-mãe que guarda todas as execuções
    :param prefix   : prefixo do nome de cada subpasta
    :return         : Path para o subdiretório recém-criado
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)   # garante que a pasta-mãe existe

    # Regex que casa apenas nomes como "relatorio07" e captura o número (07)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    existing_nums = []
    for item in base.iterdir():               # percorre o conteúdo de base/
        if item.is_dir():                     # só interessam pastas
            match = pattern.match(item.name)
            if match:
                existing_nums.append(int(match.group(1)))

    # Se nenhuma pasta existe ainda, default=0 garante que o próximo seja 1
    next_num = max(existing_nums, default=0) + 1
    run_dir = base / f"{prefix}{next_num:02d}"   # :02d = 2 dígitos com zero à esquerda
    run_dir.mkdir()
    return run_dir


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
def gerar_txt(m_seed, m_opt, c_seed, c_opt, m_div, destino, formula=""):
    """
    Escreve o relatório COMPARATIVO com as TRÊS comparações pedidas:

      [1] SEMENTE  vs OTIMIZADO  -> quanto o GA mudou/melhorou a predição
      [2] SEMENTE  vs ORIGINAL   -> erro do modelo herdado contra os dados reais
      [3] OTIMIZADO vs ORIGINAL  -> erro do modelo do GA contra os dados reais

    Parâmetros
    ----------
    m_seed, m_opt : calcular_metricas(X, Y_true, Y_*) — erro de cada modelo vs ORIGINAL
    c_seed, c_opt : classificar() de cada modelo
    m_div         : calcular_metricas(X, Y_seed, Y_opt) — divergência entre os modelos
    """
    L = []
    L.append("=" * 72)
    L.append("  RELATORIO COMPARATIVO — 3 COMPARACOES")
    L.append("=" * 72)
    L.append("")
    L.append("  Sinais:")
    L.append("    ORIGINAL  = dados medidos (dadosIniciais) — ground truth (Yreal/Yimg)")
    L.append("    SEMENTE   = modelo herdado (linhagem VARMAX), ponto de partida do GA")
    L.append("    OTIMIZADO = modelo NARX apos refinamento dos betas pelo GA")
    L.append("")
    L.append(f"  Amostras analisadas : {m_opt['n_amostras']:,}")
    L.append(f"  PAPR do sinal       : {m_opt['papr_dB']:.2f} dB")
    L.append("")
    L.append("  Como ler a tabela: cada coluna E uma das comparacoes pedidas.")
    L.append("    [2] SEMENTE  vs ORIGINAL  -> coluna SEMENTE   (erro do herdado vs dados reais)")
    L.append("    [3] OTIMIZADO vs ORIGINAL -> coluna OTIMIZADO (erro do GA vs dados reais)")
    L.append("    [1] SEMENTE  vs OTIMIZADO -> coluna GANHO     (quanto o GA reduziu o erro)")
    L.append("")
    L.append("-" * 72)
    L.append(f"  {'METRICA':<22}{'SEMENTE':>13}{'OTIMIZADO':>13}{'GANHO':>17}")
    L.append("-" * 72)

    # RMSE — menor é melhor; ganho relativo em %
    d_rmse = (m_seed["rmse"] - m_opt["rmse"]) / (abs(m_seed["rmse"]) + 1e-15) * 100
    L.append(f"  {'RMSE (abs)':<22}{m_seed['rmse']:>13.5f}{m_opt['rmse']:>13.5f}{d_rmse:>+14.2f} %")

    # NMSE — mais negativo é melhor; ganho expresso em dB
    d_nmse = m_seed["nmse_dB"] - m_opt["nmse_dB"]      # positivo => otimizado melhor
    L.append(f"  {'NMSE (dB)':<22}{m_seed['nmse_dB']:>13.2f}{m_opt['nmse_dB']:>13.2f}{d_nmse:>+13.2f} dB")

    # EVM — menor é melhor; ganho relativo em %
    d_evm = (m_seed["evm_pct"] - m_opt["evm_pct"]) / (abs(m_seed["evm_pct"]) + 1e-15) * 100
    L.append(f"  {'EVM (%)':<22}{m_seed['evm_pct']:>13.2f}{m_opt['evm_pct']:>13.2f}{d_evm:>+14.2f} %")
    L.append("-" * 72)
    L.append("  GANHO > 0  =>  o otimizado esta mais perto do ORIGINAL que a semente.")
    L.append("")
    L.append("  Vereditos (padroes 3GPP/industria) para [2] e [3]:")
    L.append(f"    NMSE : [2] semente = {c_seed['nmse']:<11} | [3] otimizado = {c_opt['nmse']}")
    L.append(f"    EVM  : [2] semente = {c_seed['evm']:<11} | [3] otimizado = {c_opt['evm']}")
    L.append("")
    L.append("-" * 72)
    L.append("  [1] SEMENTE vs OTIMIZADO — divergencia DIRETA entre os dois modelos")
    L.append("      (mede o quanto o GA moveu a predicao; NAO envolve o ORIGINAL)")
    L.append(f"        RMS |Y_sem - Y_opt|  : {m_div['rmse']:.6f}")
    L.append(f"        Nivel de divergencia : {m_div['nmse_dB']:.2f} dB  (relativo a potencia da semente)")
    L.append("")
    L.append("-" * 72)
    L.append("  Caracteristicas do ORIGINAL (sinal medido; independem do modelo):")
    L.append(f"    Ganho medio           : {m_opt['ganho_dB']:.2f} dB")
    L.append(f"    Planicidade ganho std : {m_opt['gain_flatness_std']:.3f} dB")
    L.append(f"    Planicidade fase  std : {m_opt['phase_flatness_std']:.3f} graus")
    L.append("=" * 72)

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
def painel_par(titulo, X, Yref, lab_ref, cor_ref, Ytest, lab_test, cor_test,
               destino, modo="erro"):
    """
    Monta um painel 2x2 comparando DOIS sinais: uma REFERENCIA e um TESTE.

    A mesma função serve para as três comparações — muda só quem entra como
    referência e como teste:

      [1] Semente vs Otimizado : ref=Semente,  teste=Otimizado, modo="diverg"
      [2] Semente vs Original  : ref=Original, teste=Semente,   modo="erro"
      [3] Otimizado vs Original: ref=Original, teste=Otimizado, modo="erro"

    modo="erro"   : as métricas medem o ERRO do teste contra a referência real
                    (a referência é o ORIGINAL/medido).
    modo="diverg" : as métricas medem a DIVERGENCIA entre dois modelos
                    (a referência é a semente, não a verdade) — sem veredito.

    :return: dict de métricas calculado para o par.
    """
    # calcular_metricas(X, referencia, teste): erro/divergência de teste vs ref
    m = calcular_metricas(X, Yref, Ytest)

    idx = np.linspace(0, len(X) - 1, min(3000, len(X)), dtype=int)
    Xs, Yr, Ye = X[idx], Yref[idx], Ytest[idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="white")
    fig.suptitle(titulo, fontsize=15, fontweight="bold")

    # [1] Constelação IQ
    ax = axes[0, 0]
    ax.scatter(Yr.real, Yr.imag, c=cor_ref,  s=5, alpha=0.35, label=lab_ref)
    ax.scatter(Ye.real, Ye.imag, c=cor_test, s=5, alpha=0.35, label=lab_test)
    ax.set_title("Constelacao IQ", fontweight="bold")
    ax.set_xlabel("I (real)"); ax.set_ylabel("Q (imag)")
    ax.legend(markerscale=3); ax.grid(True, alpha=0.3); ax.set_aspect("equal")

    # [2] AM-AM: ganho vs amplitude
    ax = axes[0, 1]
    amp = np.abs(Xs)
    ax.scatter(amp, np.abs(Yr) / (amp + 1e-12), c=cor_ref,  s=4, alpha=0.35, label=lab_ref)
    ax.scatter(amp, np.abs(Ye) / (amp + 1e-12), c=cor_test, s=4, alpha=0.35, label=lab_test)
    ax.set_title("AM-AM — Ganho vs Amplitude", fontweight="bold")
    ax.set_xlabel("|X[n]|"); ax.set_ylabel("Ganho |Y|/|X|")
    ax.legend(markerscale=3); ax.grid(True, alpha=0.3)

    # [3] Diferença ponto-a-ponto ao longo do tempo
    ax = axes[1, 0]
    dif = np.abs(Yref - Ytest)
    ax.plot(dif[:2000], c=cor_test, lw=0.7, alpha=0.85)
    ax.axhline(dif.mean(), color="#333333", ls="--", lw=1.3,
               label=f"media: {dif.mean():.4f}")
    rotulo = f"|{lab_ref} - {lab_test}|"
    ax.set_title(f"Diferenca ponto-a-ponto  {rotulo}", fontweight="bold")
    ax.set_xlabel("Amostra"); ax.set_ylabel("|diferenca|")
    ax.legend(); ax.grid(True, alpha=0.3)

    # [4] Tabela de métricas do par
    ax = axes[1, 1]; ax.axis("off")
    if modo == "diverg":
        cap = f"Divergencia (ref. = {lab_ref})"
        linhas = [
            ["RMS |dif|",  f"{m['rmse']:.6f}"],
            ["Nivel (dB)", f"{m['nmse_dB']:.2f}"],
            ["EVM-equiv",  f"{m['evm_pct']:.2f} %"],
        ]
    else:
        cap = f"Erro de {lab_test} vs {lab_ref}"
        linhas = [
            ["RMSE", f"{m['rmse']:.5f}"],
            ["NMSE", f"{m['nmse_dB']:.2f} dB"],
            ["EVM",  f"{m['evm_pct']:.2f} %"],
        ]
    tab = ax.table(cellText=linhas, colLabels=["Metrica", "Valor"],
                   loc="center", cellLoc="left", colWidths=[0.5, 0.45])
    tab.auto_set_font_size(False); tab.set_fontsize(12); tab.scale(1, 2.4)
    for j in range(2):
        tab[(0, j)].set_facecolor("#534AB7")
        tab[(0, j)].set_text_props(color="white", fontweight="bold")
    ax.set_title(cap, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    print(f"Painel salvo em: {destino}")
    return m


# ─────────────────────────────────────────────────────────────
#  Carregamento de dados
# ─────────────────────────────────────────────────────────────
def carregar(csv_path):
    """
    Carrega os sinais para a comparação SEMENTE vs. OTIMIZADO.

    Espera o pa_linearization_results.csv gerado pelo main_pipeline.py, que
    contém as predições dos dois modelos.

    Retorna
    -------
    X        : entrada complexa
    Y_true   : saída medida (Yreal/Yimg) — ground truth
    Y_seed   : predição do modelo SEMENTE (herdado). Lida de Ypred_seed_* se
               existir; caso contrário, recalculada na hora pelo modelo semente.
    Y_opt    : predição do modelo OTIMIZADO pelo GA (Ypred_opt_*) — obrigatória.
    cfg_usada: NARXModelConfig se a semente foi recalculada (para a fórmula),
               senão None.
    """
    df = pd.read_csv(csv_path)
    X = (df["Xreal"] + 1j * df["Ximg"]).to_numpy()
    Y_true = (df["Yreal"] + 1j * df["Yimg"]).to_numpy()

    cfg_usada = None

    # Modelo OTIMIZADO (GA) — não dá para recalcular sem os betas do GA,
    # então a coluna precisa existir.
    if "Ypred_opt_real" not in df.columns:
        raise ValueError(
            "CSV sem colunas Ypred_opt_* — gere primeiro o "
            "pa_linearization_results.csv rodando o main_pipeline.py."
        )
    Y_opt = (df["Ypred_opt_real"] + 1j * df["Ypred_opt_img"]).to_numpy()

    # Modelo SEMENTE — usa a coluna se existir; senão recalcula pelo modelo
    # semente (que é exatamente o NARXModelConfig herdado).
    if "Ypred_seed_real" in df.columns:
        Y_seed = (df["Ypred_seed_real"] + 1j * df["Ypred_seed_img"]).to_numpy()
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from core.model_config import NARXModelConfig
        from core.narx_engine import NARXEngine
        cfg_usada = NARXModelConfig()
        Y_seed = NARXEngine(cfg_usada).predict(X)

    return X, Y_true, Y_seed, Y_opt, cfg_usada


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "pa_linearization_results.csv"
    out_dir = create_run_directory("./relatorio", prefix="relatorio")
    print(f"Relatório desta execução em: {out_dir}")

    X, Y_true, Y_seed, Y_opt, cfg_usada = carregar(csv_path)
    m_seed = calcular_metricas(X, Y_true, Y_seed)   # erro semente vs ORIGINAL  [2]
    m_opt  = calcular_metricas(X, Y_true, Y_opt)    # erro otimizado vs ORIGINAL [3]
    m_div  = calcular_metricas(X, Y_seed, Y_opt)    # divergencia entre modelos  [1]
    c_seed = classificar(m_seed)
    c_opt  = classificar(m_opt)
    formula = formula_narx(cfg_usada)

    # Cores consistentes em todos os paineis
    COR_ORIG = "#9E9E9E"   # cinza  — original (medido)
    COR_SEED = "#FF9800"   # laranja — semente
    COR_OPT  = "#4CAF50"   # verde  — otimizado

    # Relatorio de texto com as 3 comparacoes
    gerar_txt(m_seed, m_opt, c_seed, c_opt, m_div,
              out_dir / "relatorio_comparativo.txt", formula)

    # Um painel por comparacao
    painel_par("[1] Semente vs Otimizado (o que o GA mudou)", X,
               Y_seed, "Semente", COR_SEED, Y_opt, "Otimizado", COR_OPT,
               out_dir / "comp1_semente_vs_otimizado.png", modo="diverg")

    painel_par("[2] Semente vs Original (dados iniciais)", X,
               Y_true, "Original", COR_ORIG, Y_seed, "Semente", COR_SEED,
               out_dir / "comp2_semente_vs_original.png", modo="erro")

    painel_par("[3] Otimizado vs Original (dados iniciais)", X,
               Y_true, "Original", COR_ORIG, Y_opt, "Otimizado", COR_OPT,
               out_dir / "comp3_otimizado_vs_original.png", modo="erro")