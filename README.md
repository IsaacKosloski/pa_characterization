# Sessão 1 — Identificação do PA com NARX + Algoritmo Genético

> **Propósito:** identificar o comportamento não-linear de um Power Amplifier (PA) de RF usando um modelo de série temporal multivariável (NARX — Nonlinear AutoRegressive with eXogenous inputs) cujos coeficientes são otimizados por um Algoritmo Genético (GA). O resultado é um modelo calibrado capaz de prever a saída complexa Y[n] do PA dado o sinal de entrada X[n], fornecendo o ganho linearizado `X[n] * G = Y[n]`.

---

## Sumário

- [Contexto do problema](#contexto-do-problema)
- [Arquitetura do projeto](#arquitetura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Formato dos dados](#formato-dos-dados)
- [Como executar](#como-executar)
- [Parâmetros configuráveis](#parâmetros-configuráveis)
- [Saídas geradas](#saídas-geradas)
- [Testes e validação](#testes-e-validação)
- [Referências](#referências)

---

## Contexto do problema

Um PA de RF opera idealmente como amplificador linear:

```
Y[n] = G · X[n]       (G constante para qualquer amplitude)
```

Na prática, especialmente próximo da saturação, o PA exibe:

| Fenômeno | Causa física | Efeito no sinal |
|---|---|---|
| **AM-AM** | Compressão do transistor | Ganho cai nos picos de amplitude |
| **AM-PM** | Capacitâncias não-lineares | Fase gira com a amplitude |
| **IM3** | Termos cúbicos | Produtos de intermodulação na banda adjacente |
| **Memória** | Térmicos, traps, parasitas | Ganho atual depende de instantes passados |

O modelo NARX captura todos esses efeitos numa única equação:

```
Y(t) = c  +  A₁·Y(t-1) + A₂·Y(t-2)  +  Σ βₖ · Φₖ(X(t), X(t-1), X(t-2))
        ↑         ↑                              ↑
    offset DC  memória AR               regressores polinomiais
```

O GA encontra os 36 coeficientes β que minimizam o RMSE entre Y[n] previsto e Y[n] medido.

---

## Arquitetura do projeto

```
pa_linearization/
│
├── core/
│   ├── __init__.py
│   ├── model_config.py       # Dataclasses: Intercepts, AutoregressiveMatrix,
│   │                         # ExogenousBetas, NARXModelConfig
│   └── narx_engine.py        # Motor de inferência: predict(), rmse(), nmse(),
│                             # compute_gain()
│
├── ga/
│   ├── __init__.py
│   └── genetic_algorithm.py  # GAConfig, PALinearizationGA
│
├── visualization/
│   ├── __init__.py
│   └── plots.py              # plot_constellation(), plot_am_am_pm(),
│                             # plot_ga_convergence(), plot_full_dashboard()
│
├── main_pipeline.py          # Ponto de entrada: orquestra todo o fluxo
└── requirements.txt
```

### Responsabilidade de cada módulo

| Módulo | Responsabilidade | Depende de |
|---|---|---|
| `model_config.py` | Estrutura de dados do modelo; serialização/desserialização do cromossomo | `numpy` |
| `narx_engine.py` | Executa a equação NARX amostra-a-amostra; calcula métricas | `model_config` |
| `genetic_algorithm.py` | Define e executa o GA (DEAP); avalia fitness; evolui betas | `narx_engine`, `deap` |
| `plots.py` | Visualizações de diagnóstico | `matplotlib`, `deap` |
| `main_pipeline.py` | Orquestração; carregamento de dados; geração do DataFrame final | todos acima |

---

## Pré-requisitos

- Python **3.10** ou superior (usa `str | None` — union types nativos)
- Sistema operacional: Linux, macOS ou Windows
- Memória RAM: mínimo 4 GB (recomendado 8 GB para datasets > 50k amostras)

---

## Instalação

```bash
# 1. Clone ou copie a pasta pa_linearization para seu projeto
# 2. Crie e ative um ambiente virtual
python -m venv venv_pa
source venv_pa/bin/activate          # Linux / macOS
# venv_pa\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

**Conteúdo de `requirements.txt`:**

```
deap>=1.4.1
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.11.0
```

---

## Formato dos dados

O pipeline aceita um arquivo **CSV com cabeçalho** contendo exatamente 4 colunas:

| Coluna | Tipo | Descrição |
|---|---|---|
| `Xreal` | `float64` | Parte real do sinal de entrada (canal I) |
| `Ximg` | `float64` | Parte imaginária do sinal de entrada (canal Q) |
| `Yreal` | `float64` | Parte real do sinal de saída medido |
| `Yimg` | `float64` | Parte imaginária do sinal de saída medido |

**Exemplo das primeiras linhas:**

```
Xreal,Ximg,Yreal,Yimg
1.137157,-0.061341,26.208588,-1.858761
1.137499,0.082685,26.244322,1.471061
1.117805,0.225745,25.818823,4.754794
```

> **Origem dos dados:** `dadosIniciais.xlsx` convertido — 48.384 amostras, ganho medido ≈ 23.43 (27.40 dB), consistente com os coeficientes do modelo semente.

Se você tiver os dados em `.xlsx`, converta com:

```python
import pandas as pd
df = pd.read_excel("dadosIniciais.xlsx", header=None)
df.columns = ["Xreal", "Ximg", "Yreal", "Yimg"]
df.to_csv("dadosIniciais.csv", index=False)
```

---

## Como executar

### Execução básica com dados reais

Edite o bloco `if __name__ == "__main__":` em `main_pipeline.py`:

```python
df = run_pipeline(
    csv_path        = "dadosIniciais.csv",   # caminho para o CSV
    n_generations   = 100,
    population_size = 50,
    output_dir      = "./output",
    show_plots      = True,
)
```

Em seguida, dentro da pasta `pa_linearization/`:

```bash
python main_pipeline.py
```

### Execução rápida (teste de sanidade)

Para verificar se o ambiente está correto sem esperar a evolução completa:

```python
df = run_pipeline(
    csv_path        = "dadosIniciais.csv",
    n_generations   = 5,         # apenas 5 gerações
    population_size = 10,        # população pequena
    output_dir      = "./output_test",
    show_plots      = False,
)
```

### Execução com dados sintéticos (sem CSV)

```python
df = run_pipeline(
    csv_path        = None,      # gera dados OFDM sintéticos
    n_samples       = 4096,
    n_generations   = 100,
    population_size = 50,
    output_dir      = "./output_synth",
    show_plots      = True,
)
```

### Uso programático dos módulos

```python
from core.model_config import NARXModelConfig
from core.narx_engine  import NARXEngine, compute_gain
import numpy as np

# Instancia o modelo com os coeficientes padrão
config = NARXModelConfig()
engine = NARXEngine(config)

# Prediz para qualquer sinal complexo
X = np.random.randn(1000) + 1j * np.random.randn(1000)
Y_pred = engine.predict(X)

# Métricas
print(f"Ganho linear: {config.gain_linear_dB:.2f} dB")
print(f"RMSE: {NARXEngine.rmse(Y_true, Y_pred):.6f}")
print(f"NMSE: {NARXEngine.nmse(Y_true, Y_pred):.2f} dB")
```

---

## Parâmetros configuráveis

### `run_pipeline()` — parâmetros principais

| Parâmetro | Padrão | Descrição | Quando alterar |
|---|---|---|---|
| `csv_path` | `None` | Caminho para o CSV com dados reais | Sempre que tiver dados medidos |
| `n_samples` | `4096` | Amostras sintéticas (ignorado se csv_path definido) | Apenas modo sintético |
| `n_generations` | `100` | Máximo de gerações do GA | Aumentar se RMSE ainda cai na última geração |
| `population_size` | `50` | Indivíduos por geração | Aumentar para melhor exploração (mais lento) |
| `output_dir` | `"./output"` | Pasta para CSV, PNGs e logs | Separar experimentos por pasta |
| `show_plots` | `True` | Exibir janelas matplotlib | `False` em servidores sem display |

### `GAConfig` — hiperparâmetros do algoritmo genético

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `crossover_prob` | `0.70` | Probabilidade de crossover por par de pais |
| `mutation_prob` | `0.20` | Probabilidade de mutação por indivíduo |
| `mutation_sigma_rel` | `0.05` | Desvio relativo da mutação gaussiana (5% do valor do gene) |
| `tournament_size` | `3` | Tamanho do torneio de seleção |
| `n_elites` | `2` | Número de melhores indivíduos preservados sem modificação |
| `random_seed` | `42` | Semente para reprodutibilidade |
| `init_perturbation` | `0.10` | Perturbação inicial: ±10% dos valores da semente |

---

## Saídas geradas

Após execução, a pasta `output/` conterá:

```
output/
├── pa_linearization_results.csv   # DataFrame completo com todos os sinais
├── constellation.png              # Diagrama de constelação IQ (3 painéis)
├── am_am_pm.png                   # Curvas AM-AM e AM-PM
├── ga_convergence.png             # Evolução do RMSE por geração
└── dashboard.png                  # Painel unificado com todos os gráficos
```

### Colunas do CSV de saída

| Coluna | Descrição |
|---|---|
| `Xreal`, `Ximg` | Sinal de entrada original |
| `X_magnitude` | `\|X[n]\|` — amplitude de entrada |
| `Yreal`, `Yimg` | Saída medida (ground truth) |
| `Ypred_seed_real/img` | Predição do modelo semente (antes do GA) |
| `Ypred_opt_real/img` | Predição do modelo otimizado (após GA) |
| `RMSE_seed_point` | Erro ponto-a-ponto do modelo semente |
| `RMSE_opt_point` | Erro ponto-a-ponto do modelo otimizado |
| `G_real`, `G_img` | Ganho complexo G[n] = Y_opt[n] / X[n] |
| `G_magnitude_dB` | `\|G[n]\|` em dB por amostra |
| `G_phase_deg` | Fase de G[n] em graus por amostra |
| `X_times_G_real/img` | **`X[n] * G_médio = Y[n]`** — coluna principal de linearização |

---

## Testes e validação

### Checklist de sanidade após execução

Execute no terminal após rodar o pipeline:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("output/pa_linearization_results.csv")

# 1. Ganho médio deve ser ~27.4 dB
g_dB = df["G_magnitude_dB"].mean()
assert 25 < g_dB < 30, f"Ganho fora do esperado: {g_dB:.2f} dB"

# 2. RMSE otimizado deve ser menor que o semente
rmse_seed = df["RMSE_seed_point"].mean()
rmse_opt  = df["RMSE_opt_point"].mean()
assert rmse_opt < rmse_seed, "GA não melhorou o modelo!"

# 3. X * G deve ser próximo de Y_pred
erro_xg = np.abs(df["X_times_G_real"] + 1j*df["X_times_G_img"]
                 - df["Ypred_opt_real"] - 1j*df["Ypred_opt_img"])
assert erro_xg.mean() < 1.0, "Coluna X*G inconsistente com Y_pred"

print("Todos os checks passaram")
```

### Interpretação das métricas

| Métrica | Valor típico (bons dados) | Ação se fora do range |
|---|---|---|
| NMSE (modelo otimizado) | < −30 dB | Aumentar `n_generations` ou `population_size` |
| Variação de ganho (std) | < 0.5 dB | Verificar se dados cobrem faixa dinâmica completa |
| Variação de fase (std) | < 2° | PA pode ter memória forte; revisar A₁, A₂ |
| EVM após GA | < 5% | Verificar qualidade dos dados de entrada |

---

## Referências

- Schetzen, M. — *The Volterra and Wiener Theories of Nonlinear Systems* (1980)
- Morgan, D. et al. — *A Generalized Memory Polynomial Model for Digital Predistortion of RF PAs* — IEEE Trans. Signal Processing, 2006
- DEAP Documentation — https://deap.readthedocs.io
- 3GPP TS 38.104 — NR Base Station Radio Transmission and Reception
