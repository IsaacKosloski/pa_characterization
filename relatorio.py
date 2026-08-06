
import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem display: salva em arquivo
import matplotlib.pyplot as plt

try:
    from scipy.signal import welch
    TEM_SCIPY = True
except Exception:
    TEM_SCIPY = False


CORES = {
    "medido": "#455A64",
    "predito": "#4CAF50",
    "aic": "#FF9800",
    "bic": "#9C27B0",
    "rmse": "#2196F3",
    "grade": "#E0E0E0",
}


# ---------------------------------------------------------------------------
#  Diretório numerado por rodada
# ---------------------------------------------------------------------------
def criar_diretorio_rodada(base="relatorios", prefixo="relatorio"):
    """Cria relatorios/relatorioNN com numeração automática (nunca sobrescreve)."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)

    padrao = re.compile(rf"^{re.escape(prefixo)}(\d+)$")
    numeros_existentes = []
    for item in base.iterdir():
        if item.is_dir():
            correspondencia = padrao.match(item.name)
            if correspondencia:
                numeros_existentes.append(int(correspondencia.group(1)))

    proximo = max(numeros_existentes, default=0) + 1
    destino = base / f"{prefixo}{proximo:02d}"
    destino.mkdir()
    return destino


# ---------------------------------------------------------------------------
#  Métricas padrão de TELECOM
# ---------------------------------------------------------------------------
def calcular_metricas(entrada, saida_medida, saida_predita):
    """Calcula as métricas de avaliação de um modelo de PA a partir dos sinais."""
    erro = saida_medida - saida_predita
    erro_quadratico_medio = np.mean(np.abs(erro) ** 2)
    potencia_medida = np.mean(np.abs(saida_medida) ** 2)

    amplitude_entrada = np.abs(entrada) + 1e-12
    ganho_por_amostra = np.abs(saida_medida) / amplitude_entrada

    rotacao_fase = np.angle(saida_medida, deg=True) - np.angle(entrada, deg=True)
    rotacao_fase = ((rotacao_fase + 180) % 360) - 180  # normaliza para [-180, 180]

    return {
        "n_amostras": len(entrada),
        "rmse": float(np.sqrt(erro_quadratico_medio)),
        "nmse_dB": float(10 * np.log10(erro_quadratico_medio / (potencia_medida + 1e-15))),
        "evm_pct": float(np.sqrt(erro_quadratico_medio / (potencia_medida + 1e-15)) * 100),
        "ganho_dB": float(20 * np.log10(np.mean(ganho_por_amostra))),
        "planicidade_ganho": float(np.std(20 * np.log10(ganho_por_amostra + 1e-15))),
        "planicidade_fase": float(np.std(rotacao_fase)),
        "papr_dB": float(10 * np.log10(np.max(np.abs(entrada) ** 2) / np.mean(np.abs(entrada) ** 2))),
    }


def classificar(metricas):
    """Atribui um veredito qualitativo (padrões usuais da indústria)."""
    def veredito(valor, limite_excelente, limite_bom):
        if valor <= limite_excelente:
            return "EXCELENTE"
        if valor <= limite_bom:
            return "BOM"
        return "A MELHORAR"

    magnitude_nmse = -metricas["nmse_dB"]
    if magnitude_nmse >= 40:
        veredito_nmse = "EXCELENTE"
    elif magnitude_nmse >= 30:
        veredito_nmse = "BOM"
    else:
        veredito_nmse = "A MELHORAR"

    return {
        "nmse": veredito_nmse,
        "evm": veredito(metricas["evm_pct"], 1.5, 3.5),
        "ganho": veredito(metricas["planicidade_ganho"], 0.1, 0.5),
        "fase": veredito(metricas["planicidade_fase"], 0.5, 2.0),
    }


# ---------------------------------------------------------------------------
#  Critérios de seleção de modelo (AIC e BIC)
# ---------------------------------------------------------------------------
def _menos_duas_log_verossimilhanca(rmse, n_amostras):
    # Para regressão gaussiana: N * ln(SSE / N), com SSE = rmse^2 * N
    return n_amostras * np.log(rmse ** 2)


def valor_aic(rmse, n_coeficientes, n_amostras, parametros_por_coeficiente=2):
    numero_parametros = parametros_por_coeficiente * n_coeficientes
    return _menos_duas_log_verossimilhanca(rmse, n_amostras) + 2 * numero_parametros


def valor_bic(rmse, n_coeficientes, n_amostras, parametros_por_coeficiente=2):
    numero_parametros = parametros_por_coeficiente * n_coeficientes
    return _menos_duas_log_verossimilhanca(rmse, n_amostras) + numero_parametros * np.log(n_amostras)


def fronteira_de_pareto(historico):
    """Mantém apenas os modelos não-dominados: ao crescer em coeficientes, o RMSE cai."""
    ordenado = historico.sort_values("n_coef")
    melhor_rmse_ate_agora = np.inf
    linhas_mantidas = []
    for _, linha in ordenado.iterrows():
        if linha["score"] < melhor_rmse_ate_agora:
            linhas_mantidas.append(linha)
            melhor_rmse_ate_agora = linha["score"]
    return pd.DataFrame(linhas_mantidas)


def calcular_selecoes(historico, n_amostras_padrao):
    """
    Anexa colunas AIC e BIC ao histórico e devolve as escolhas por cada critério.

    Se o histórico já tiver a coluna 'N' (número de resíduos por combinação),
    ela é usada; caso contrário, usa-se n_amostras_padrao para todas as linhas.
    """
    historico = historico.copy()

    if "N" in historico.columns:
        amostras_por_linha = historico["N"].to_numpy()
    else:
        amostras_por_linha = np.full(len(historico), n_amostras_padrao)

    historico["aic"] = [
        valor_aic(rmse, n_coef, n)
        for rmse, n_coef, n in zip(historico["score"], historico["n_coef"], amostras_por_linha)
    ]
    historico["bic"] = [
        valor_bic(rmse, n_coef, n)
        for rmse, n_coef, n in zip(historico["score"], historico["n_coef"], amostras_por_linha)
    ]

    escolhas = {
        "RMSE": historico.loc[historico["score"].idxmin()],
        "AIC": historico.loc[historico["aic"].idxmin()],
        "BIC": historico.loc[historico["bic"].idxmin()],
    }
    return escolhas, historico


# ---------------------------------------------------------------------------
#  Equação do modelo (termos dominantes)
# ---------------------------------------------------------------------------
def montar_equacao(nomes_termos, coeficientes_reais, coeficientes_imag, quantidade=12):
    """Lista os termos com maior magnitude de coeficiente complexo."""
    coeficientes_reais = np.asarray(coeficientes_reais)
    coeficientes_imag = np.asarray(coeficientes_imag)

    if len(nomes_termos) != len(coeficientes_reais):
        return "(nomes dos termos indisponíveis no report.json)"

    magnitude = np.sqrt(coeficientes_reais ** 2 + coeficientes_imag ** 2)
    indices_dominantes = np.argsort(magnitude)[::-1][:quantidade]

    linhas = []
    for indice in indices_dominantes:
        parte_real = coeficientes_reais[indice]
        parte_imag = coeficientes_imag[indice]
        nome = nomes_termos[indice]
        linhas.append(f"  ({parte_real:+.4g}{parte_imag:+.4g}j) · {nome}")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
#  Carregamento de uma rodada
# ---------------------------------------------------------------------------
def carregar_rodada(diretorio):
    """Lê todos os artefatos de uma pasta runNN e devolve um dicionário."""
    diretorio = Path(diretorio)

    relatorio_json = json.load(open(diretorio / "report.json"))
    historico = pd.read_csv(diretorio / "gridsearch.csv")
    simulacao = pd.read_csv(diretorio / "simulation.csv")
    coeficientes = np.load(diretorio / "coefficients.npz")

    entrada = simulacao["Xreal"].to_numpy() + 1j * simulacao["Ximg"].to_numpy()
    saida_medida = simulacao["Yreal"].to_numpy() + 1j * simulacao["Yimg"].to_numpy()
    saida_predita = simulacao["Ypred_real"].to_numpy() + 1j * simulacao["Ypred_img"].to_numpy()

    # número de coeficientes: robusto, vem do próprio vetor salvo (não do nome da chave)
    numero_coeficientes = len(coeficientes["real"])

    return {
        "diretorio": diretorio,
        "nome_modelo": relatorio_json.get("model", diretorio.parent.name),
        "hiperparametros": relatorio_json.get("params", {}),
        "nomes_termos": relatorio_json.get("term_names", []),
        "n_coeficientes": numero_coeficientes,
        "historico": historico,
        "entrada": entrada,
        "saida_medida": saida_medida,
        "saida_predita": saida_predita,
        "coeficientes_reais": coeficientes["real"],
        "coeficientes_imag": coeficientes["imag"],
    }


# ---------------------------------------------------------------------------
#  Painéis de gráficos
# ---------------------------------------------------------------------------
def gerar_painel_modelo(rodada, destino):
    """Painel 2x2: constelação, AM-AM, AM-PM e PSD."""
    entrada = rodada["entrada"]
    saida_medida = rodada["saida_medida"]
    saida_predita = rodada["saida_predita"]

    indices = np.linspace(0, len(entrada) - 1, min(3000, len(entrada)), dtype=int)

    figura, eixos = plt.subplots(2, 2, figsize=(14, 11), facecolor="white")
    figura.suptitle(f"{rodada['nome_modelo']} — painel de caracterização",
                    fontsize=15, fontweight="bold")

    # Constelação IQ
    eixo = eixos[0, 0]
    eixo.scatter(saida_medida[indices].real, saida_medida[indices].imag,
                 c=CORES["medido"], s=5, alpha=0.35, label="Medido")
    eixo.scatter(saida_predita[indices].real, saida_predita[indices].imag,
                 c=CORES["predito"], s=5, alpha=0.35, label="Predito")
    eixo.set_title("Constelação IQ", fontweight="bold")
    eixo.set_xlabel("I (real)")
    eixo.set_ylabel("Q (imaginário)")
    eixo.legend(markerscale=3)
    eixo.grid(True, alpha=0.3)
    eixo.set_aspect("equal")

    # AM-AM
    eixo = eixos[0, 1]
    amplitude_entrada = np.abs(entrada[indices])
    eixo.scatter(amplitude_entrada, np.abs(saida_medida[indices]),
                 c=CORES["medido"], s=4, alpha=0.35, label="Medido")
    eixo.scatter(amplitude_entrada, np.abs(saida_predita[indices]),
                 c=CORES["predito"], s=4, alpha=0.35, label="Predito")
    eixo.set_title("AM-AM", fontweight="bold")
    eixo.set_xlabel("|X[n]|")
    eixo.set_ylabel("|Y[n]|")
    eixo.legend(markerscale=3)
    eixo.grid(True, alpha=0.3)

    # AM-PM
    eixo = eixos[1, 0]
    epsilon = 1e-12
    fase_medida = np.angle(saida_medida[indices] / (entrada[indices] + epsilon), deg=True)
    fase_predita = np.angle(saida_predita[indices] / (entrada[indices] + epsilon), deg=True)
    eixo.scatter(amplitude_entrada, fase_medida, c=CORES["medido"], s=4, alpha=0.35, label="Medido")
    eixo.scatter(amplitude_entrada, fase_predita, c=CORES["predito"], s=4, alpha=0.35, label="Predito")
    eixo.set_title("AM-PM", fontweight="bold")
    eixo.set_xlabel("|X[n]|")
    eixo.set_ylabel("∠Y - ∠X (graus)")
    eixo.legend(markerscale=3)
    eixo.grid(True, alpha=0.3)

    # PSD (espectro de saída - contexto de ACPR / regrowth espectral)
    eixo = eixos[1, 1]
    if TEM_SCIPY:
        frequencia_medida, densidade_medida = welch(saida_medida, nperseg=1024, return_onesided=False)
        frequencia_predita, densidade_predita = welch(saida_predita, nperseg=1024, return_onesided=False)
        ordem_medida = np.argsort(frequencia_medida)
        ordem_predita = np.argsort(frequencia_predita)
        eixo.plot(frequencia_medida[ordem_medida],
                  10 * np.log10(densidade_medida[ordem_medida] + 1e-15),
                  c=CORES["medido"], linewidth=1, label="Medido")
        eixo.plot(frequencia_predita[ordem_predita],
                  10 * np.log10(densidade_predita[ordem_predita] + 1e-15),
                  c=CORES["predito"], linewidth=1, label="Predito")
        eixo.set_xlabel("Frequência normalizada")
    else:
        espectro_medido = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(saida_medida))) + 1e-9)
        espectro_predito = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(saida_predita))) + 1e-9)
        eixo.plot(espectro_medido, c=CORES["medido"], linewidth=0.5, label="Medido")
        eixo.plot(espectro_predito, c=CORES["predito"], linewidth=0.5, label="Predito")
    eixo.set_title("PSD — espectro de saída", fontweight="bold")
    eixo.set_ylabel("dB")
    eixo.legend()
    eixo.grid(True, alpha=0.3)

    plt.tight_layout()
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)


def gerar_painel_fronteira(historico, escolhas, historico_com_criterios, destino, nome_modelo):
    """Painel 1x2: RMSE x complexidade (com as escolhas) e AIC/BIC x complexidade."""
    pareto = fronteira_de_pareto(historico).sort_values("n_coef")

    figura, eixos = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    figura.suptitle(f"{nome_modelo} — fronteira e seleção", fontsize=14, fontweight="bold")

    eixos[0].plot(pareto["n_coef"], pareto["score"], "-o",
                  c=CORES["rmse"], label="fronteira de Pareto")
    for criterio, cor in [("RMSE", CORES["rmse"]), ("AIC", CORES["aic"]), ("BIC", CORES["bic"])]:
        escolha = escolhas[criterio]
        eixos[0].scatter([escolha["n_coef"]], [escolha["score"]],
                         c=cor, s=140, zorder=5, edgecolors="black", label=f"escolha {criterio}")
    eixos[0].set_xlabel("número de coeficientes")
    eixos[0].set_ylabel("RMSE (validação)")
    eixos[0].set_title("RMSE × complexidade")
    eixos[0].legend()
    eixos[0].grid(True, alpha=0.3)

    ordenado = historico_com_criterios.sort_values("n_coef")
    eixos[1].plot(ordenado["n_coef"], ordenado["aic"], ".", c=CORES["aic"], alpha=0.5, label="AIC")
    eixos[1].plot(ordenado["n_coef"], ordenado["bic"], ".", c=CORES["bic"], alpha=0.5, label="BIC")
    eixos[1].set_xlabel("número de coeficientes")
    eixos[1].set_ylabel("critério (menor = melhor)")
    eixos[1].set_title("AIC / BIC × complexidade")
    eixos[1].legend()
    eixos[1].grid(True, alpha=0.3)

    plt.tight_layout()
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)


# ---------------------------------------------------------------------------
#  Montagem do relatório
# ---------------------------------------------------------------------------
def gerar(pasta_output="output", pasta_relatorios="relatorios"):
    pasta_output = Path(pasta_output)

    rodadas_para_relatar = []
    for pasta_modelo in sorted(pasta_output.iterdir()):
        if not pasta_modelo.is_dir():
            continue
        pastas_de_rodada = sorted(p for p in pasta_modelo.glob("run*") if p.is_dir())
        if pastas_de_rodada:
            rodadas_para_relatar.append(pastas_de_rodada[-1])  # a última rodada de cada modelo

    if not rodadas_para_relatar:
        print("Nenhuma rodada encontrada em", pasta_output)
        return None

    destino = criar_diretorio_rodada(pasta_relatorios)

    linhas = [
        "# Relatório de Caracterização de PA",
        "",
        f"*Gerado em {pd.Timestamp.now():%Y-%m-%d %H:%M}*  ·  fonte: `{pasta_output}/`",
        "",
        "## 1. Comparação entre modelos",
        "",
        "| Modelo | nº coef | RMSE | NMSE (dB) | EVM (%) | Ganho (dB) | Veredito EVM |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    dados_por_modelo = []
    for pasta_rodada in rodadas_para_relatar:
        rodada = carregar_rodada(pasta_rodada)
        metricas = calcular_metricas(rodada["entrada"], rodada["saida_medida"], rodada["saida_predita"])
        vereditos = classificar(metricas)
        rodada["metricas"] = metricas
        rodada["vereditos"] = vereditos
        dados_por_modelo.append(rodada)

        linhas.append(
            f"| {rodada['nome_modelo']} | {rodada['n_coeficientes']} | "
            f"{metricas['rmse']:.4f} | {metricas['nmse_dB']:.2f} | {metricas['evm_pct']:.3f} | "
            f"{metricas['ganho_dB']:.2f} | {vereditos['evm']} |"
        )

    melhor = min(dados_por_modelo, key=lambda r: r["metricas"]["rmse"])
    linhas += [
        "",
        f"**Melhor por RMSE:** {melhor['nome_modelo']} "
        f"(RMSE = {melhor['metricas']['rmse']:.4f}, EVM = {melhor['metricas']['evm_pct']:.3f}%, "
        f"{melhor['n_coeficientes']} coeficientes).",
        "",
    ]

    for rodada in dados_por_modelo:
        nome = rodada["nome_modelo"]
        metricas = rodada["metricas"]
        vereditos = rodada["vereditos"]

        gerar_painel_modelo(rodada, destino / f"{nome}_painel.png")

        n_amostras_simulacao = len(rodada["saida_medida"])
        escolhas, historico_com_criterios = calcular_selecoes(rodada["historico"], n_amostras_simulacao)
        gerar_painel_fronteira(rodada["historico"], escolhas, historico_com_criterios,
                               destino / f"{nome}_fronteira.png", nome)

        linhas += [
            f"## Modelo: {nome}",
            "",
            f"- **Hiperparâmetros:** `{rodada['hiperparametros']}`",
            f"- **Amostras (teste):** {metricas['n_amostras']:,}  ·  "
            f"**PAPR da entrada:** {metricas['papr_dB']:.2f} dB",
            "",
            "### Métricas (padrão TELECOM)",
            "",
            "| Métrica | Valor | Veredito |",
            "|---|---:|---|",
            f"| RMSE | {metricas['rmse']:.5f} | — |",
            f"| NMSE | {metricas['nmse_dB']:.2f} dB | {vereditos['nmse']} |",
            f"| EVM | {metricas['evm_pct']:.3f} % | {vereditos['evm']} |",
            f"| Ganho médio | {metricas['ganho_dB']:.2f} dB | — |",
            f"| Planicidade de ganho (σ) | {metricas['planicidade_ganho']:.3f} dB | {vereditos['ganho']} |",
            f"| Planicidade de fase (σ) | {metricas['planicidade_fase']:.3f} ° | {vereditos['fase']} |",
            "",
            "### Seleção de modelo (histórico do GridSearch)",
            "",
            f"O GridSearch avaliou **{len(rodada['historico'])}** combinações. Escolha por critério:",
            "",
            "| Critério | nº coef | RMSE |",
            "|---|---:|---:|",
        ]
        for criterio in ["RMSE", "AIC", "BIC"]:
            escolha = escolhas[criterio]
            linhas.append(f"| {criterio} | {int(escolha['n_coef'])} | {escolha['score']:.4f} |")

        if "N" not in rodada["historico"].columns:
            linhas.append("")
            linhas.append(f"> Observação: `gridsearch.csv` sem coluna `N`; AIC/BIC usaram "
                          f"N = {n_amostras_simulacao} (tamanho da simulação) para todas as combinações.")

        linhas += [
            "",
            f"![fronteira]({nome}_fronteira.png)",
            "",
            "### Equação do modelo (12 termos dominantes)",
            "",
            "```",
            montar_equacao(rodada["nomes_termos"], rodada["coeficientes_reais"], rodada["coeficientes_imag"]),
            "```",
            "",
            f"![painel]({nome}_painel.png)",
            "",
        ]

    (destino / "relatorio.md").write_text("\n".join(linhas), encoding="utf-8")
    print("Relatório gerado em:", destino)
    return destino


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "output"
    gerar(pasta)