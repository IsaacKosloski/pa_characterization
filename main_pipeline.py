"""
Pipeline principal de Linearização do PA.

Orquestra:
    1. Carregamento / geração dos dados
    2. Inferência com o modelo semente
    3. Execução do Algoritmo Genético
    4. Cálculo do ganho linearizado G
    5. Inserção da coluna "X[n] * G = Y[n]" no DataFrame
    6. Geração dos gráficos de análise

Execute com:
    python main_pipeline.py

Dependências:
    pip install deap numpy pandas matplotlib scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Módulos do projeto
from core.model_config  import NARXModelConfig
from core.narx_engine   import NARXEngine, compute_gain
from ga.genetic_algorithm import PALinearizationGA, GAConfig
from visualization.plots  import (
    plot_constellation,
    plot_am_am_pm,
    plot_ga_convergence,
    plot_full_dashboard,
)

# --- Utilitários de Dados
def generate_data(
    csv_path: str | None = None,
    n_samples: int = 4096,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega dados de um CSV.

    Para dados reais, o CSV deve ter as colunas:
        Xreal, Ximg, Yreal, Yimg

    :return
    X : np.ndarray complex (N,) — sinal de entrada
    Y : np.ndarray complex (N,) — sinal de saída medido
    """
    if csv_path and Path(csv_path).exists():
        print(f"Carregando dados de: {csv_path}")
        df = pd.read_csv(csv_path)
        X = (df["Xreal"] + 1j * df["Ximg"]).to_numpy()
        Y = (df["Yreal"] + 1j * df["Yimg"]).to_numpy()
        print(f"   {len(X)} amostras carregadas.")
        return X, Y
    else:
        raise FileNotFoundError(
            f"CSV não encontrado: {csv_path}. "
            f"Verifique se o caminho relativo está correto a partir de onde você executa o script."
        )

def build_dataset(
    X: np.ndarray,
    Y_true: np.ndarray,
    Y_pred_seed: np.ndarray,
    Y_pred_opt:  np.ndarray,
    G_linearized: np.ndarray,
) -> pd.DataFrame:
    """
    Constrói o DataFrame final com todas as colunas relevantes.

    Colunas geradas:
        Xreal, Ximg          → entrada complexa
        Yreal, Yimg          → saída medida (ground truth)
        Ypred_seed_real/img  → predição do modelo semente
        Ypred_opt_real/img   → predição do modelo otimizado
        G_real, G_img        → ganho complexo G[n] = Y_opt[n]/X[n]
        G_magnitude_dB       → |G[n]| em dB
        G_phase_deg          → ∠G[n] em graus
        X_G_Yreal            → Re(X[n] * G_media) — coluna "X*G=Y"
        X_G_Yimg             → Im(X[n] * G_media)
    """
    # Ganho médio (escalar complexo) para a coluna X[n]*G=Y[n]
    G_mean = np.nanmean(G_linearized[np.abs(G_linearized) > 0])

    df = pd.DataFrame({
        # Entrada
        "Xreal":              X.real,
        "Ximg":               X.imag,
        "X_magnitude":        np.abs(X),
        # Saída medida
        "Yreal":              Y_true.real,
        "Yimg":               Y_true.imag,
        # Predição semente (modelo original)
        "Ypred_seed_real":    Y_pred_seed.real,
        "Ypred_seed_img":     Y_pred_seed.imag,
        "RMSE_seed_point":    np.abs(Y_true - Y_pred_seed),
        # Predição otimizada (GA)
        "Ypred_opt_real":     Y_pred_opt.real,
        "Ypred_opt_img":      Y_pred_opt.imag,
        "RMSE_opt_point":     np.abs(Y_true - Y_pred_opt),
        # Ganho complexo por amostra
        "G_real":             G_linearized.real,
        "G_img":              G_linearized.imag,
        "G_magnitude":        np.abs(G_linearized),
        "G_magnitude_dB":     20 * np.log10(np.abs(G_linearized) + 1e-15),
        "G_phase_deg":        np.angle(G_linearized, deg=True),
        # ── Coluna principal: X[n] * G = Y[n] ────────────────
        # Multiplica cada amostra pelo ganho médio linearizado
        "X_times_G_real":     (X * G_mean).real,
        "X_times_G_img":      (X * G_mean).imag,
        "X_times_G_magnitude": np.abs(X * G_mean),
    })

    return df

# --- Pipeline Principal
def run_pipeline(
    csv_path:       str | None = None,
    n_samples:      int   = 4096,
    n_generations:  int   = 100,
    population_size:int   = 50,
    output_dir:     str   = "./output",
    show_plots:     bool  = True,
) -> pd.DataFrame:
    """
    Executa o pipeline completo de linearização do PA.

    Parâmetros
    ----------
    csv_path       : caminho para CSV com dados reais
    n_samples      : amostras para dados sintéticos
    n_generations  : gerações do GA
    population_size: população do GA
    output_dir     : pasta para salvar outputs
    show_plots     : exibe os gráficos ao final

    Retorna
    -------
    df : pd.DataFrame com todas as colunas incluindo X[n]*G=Y[n]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PA LINEARIZATION PIPELINE")
    print("=" * 60)

    # ── 1. Dados ──────────────────────────────────────────────
    X, Y_true = generate_data(csv_path, n_samples)

    # ── 2. Modelo semente ──────────────────────────────────────
    print("\nAvaliando modelo semente...")
    seed_config = NARXModelConfig()   # carrega os coeficientes do model_config.py
    seed_engine = NARXEngine(seed_config)
    Y_pred_seed = seed_engine.predict(X)

    rmse_seed = NARXEngine.rmse(Y_true, Y_pred_seed)
    nmse_seed = NARXEngine.nmse(Y_true, Y_pred_seed)
    print(f"   RMSE semente : {rmse_seed:.6f}")
    print(f"   NMSE semente : {nmse_seed:.2f} dB")
    print(f"   Ganho linear : {seed_config.gain_linear_dB:.2f} dB")

    # ── 3. Alwgoritmo Genético ─────────────────────────────────
    print(f"\nIniciando GA ({population_size} indivíduos, {n_generations} gerações)...")

    ga_config = GAConfig(
        population_size    = population_size,
        n_generations      = n_generations,
        crossover_prob     = 0.70,
        mutation_prob      = 0.20,
        mutation_sigma_rel = 0.05,
        tournament_size    = 3,
        n_elites           = 2,
        random_seed        = 42,
        init_perturbation  = 0.10,
    )

    ga = PALinearizationGA(
        seed_config  = seed_config,
        X_input      = X,
        Y_true       = Y_true,
        ga_config    = ga_config,
    )

    best_config, logbook = ga.run()

    # ── 4. Predição com modelo otimizado ──────────────────────
    print("\nPredizendo com modelo otimizado...")
    opt_engine  = NARXEngine(best_config)
    Y_pred_opt  = opt_engine.predict(X)

    rmse_opt = NARXEngine.rmse(Y_true, Y_pred_opt)
    nmse_opt = NARXEngine.nmse(Y_true, Y_pred_opt)
    print(f"   RMSE otimizado : {rmse_opt:.6f}  (redução: {rmse_seed - rmse_opt:.6f})")
    print(f"   NMSE otimizado : {nmse_opt:.2f} dB")
    print(f"   Ganho linear   : {best_config.gain_linear_dB:.2f} dB")

    # ── 5. Ganho linearizado G[n] ─────────────────────────────
    G_linearized = compute_gain(X, Y_pred_opt)
    G_mean       = np.nanmean(G_linearized[np.abs(G_linearized) > 0])
    print(f"\nGanho médio linearizado:")
    print(f"   |G| = {abs(G_mean):.4f}  ({20*np.log10(abs(G_mean)):.2f} dB)")
    print(f"   ∠G  = {np.angle(G_mean, deg=True):.2f}°")

    # ── 6. DataFrame final ────────────────────────────────────
    print("\nConstruindo DataFrame...")
    df = build_dataset(X, Y_true, Y_pred_seed, Y_pred_opt, G_linearized)

    csv_out = out / "pa_linearization_results.csv"
    df.to_csv(csv_out, index=False)
    print(f"   Salvo em: {csv_out}")
    print(f"   Shape: {df.shape}")
    print(f"\n   Colunas: {list(df.columns)}")

    # ── 7. Resumo: X * G = Y ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  EQUAÇÃO X[n] * G = Y[n]")
    print("=" * 60)
    print(f"  G (complexo) = {G_mean:.4f}")
    print(f"  |G|          = {abs(G_mean):.4f}  →  {20*np.log10(abs(G_mean)):.2f} dB")
    print(f"  ∠G           = {np.angle(G_mean, deg=True):.4f}°")
    print()
    print(f"  Verificação na amostra 100:")
    n = 100
    Y_check = X[n] * G_mean
    print(f"    X[100]         = {X[n]:.6f}")
    print(f"    X[100] * G     = {Y_check:.6f}")
    print(f"    Y_pred_opt[100]= {Y_pred_opt[n]:.6f}")
    print(f"    Y_true[100]    = {Y_true[n]:.6f}")

    # ── 8. Gráficos ───────────────────────────────────────────
    print("\nGerando gráficos...")

    fig_const = plot_constellation(
        X, Y_pred_seed, Y_pred_opt,
        title_suffix="— Linearização GA",
        save_path=str(out / "constellation.png"),
    )
    fig_amam = plot_am_am_pm(
        X, Y_pred_seed, Y_pred_opt,
        save_path=str(out / "am_am_pm.png"),
    )
    fig_ga = plot_ga_convergence(
        logbook,
        save_path=str(out / "ga_convergence.png"),
    )
    fig_dash = plot_full_dashboard(
        X, Y_pred_seed, Y_pred_opt,
        logbook=logbook,
        save_path=str(out / "dashboard.png"),
    )

    if show_plots:
        plt.show()

    print("\nPipeline concluído com sucesso!")
    return df

# --- Entrada Principal
if __name__ == "__main__":
    df = run_pipeline(
        csv_path        = "../../datasets/dadosIniciais.csv",
        n_samples       = 4096,          # usado apenas para dados sintéticos
        n_generations   = 100,
        population_size = 50,
        output_dir      = "./output",
        show_plots      = True,
    )

    # Inspeciona as primeiras linhas
    print("\nPrimeiras 5 linhas do DataFrame:")
    print(df[["Xreal", "Ximg", "Yreal", "Yimg",
              "Ypred_opt_real", "Ypred_opt_img",
              "G_magnitude_dB", "G_phase_deg",
              "X_times_G_real", "X_times_G_img"]].head())

