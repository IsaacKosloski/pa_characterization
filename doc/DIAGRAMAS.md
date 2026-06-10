# Diagramas de Comportamento e Fluxo — Sessão 1

---

## Diagrama 1 — Arquitetura de Módulos (Estrutural)

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#EEEDFE", "primaryBorderColor": "#534AB7", "primaryTextColor": "#26215C", "lineColor": "#888780", "fontSize": "13px"}}}%%
flowchart TD

    classDef gray   fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef coral  fill:#FAECE7,stroke:#993C1D,color:#4A1B0C
    classDef purple fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef teal   fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef green  fill:#EAF3DE,stroke:#3B6D11,color:#173404
    classDef red    fill:#FCEBEB,stroke:#A32D2D,color:#501313

    subgraph ENTRADA["Entrada"]
        XN["X[n] — sinal complexo I/Q"]:::gray
    end

    subgraph NARX["NARXModelConfig — estrutura do modelo"]
        IC["Interceptos\nc.Yreal = −0.00053 · c.Yimg = +0.00006\nOffset DC do ponto de operação Q"]:::coral
        AR["Matrizes A₁ e A₂\nMemória autorregressiva da saída\nValores ≈ 10⁻⁵  — memória fraca"]:::coral
        BE["Betas β — 36 coeficientes\ndeg 1 → ganho ~27.4 dB\ndeg 2 → HD2 / IM2\ndeg 3 → AM-AM · AM-PM · IM3"]:::coral
    end

    subgraph ENGINE["NARXEngine — inferência amostra a amostra"]
        EQ["Y(t) = c + A₁·Y(t-1) + A₂·Y(t-2) + β·Φ(X)\nΦ = regressores: linear + quadrático + cúbico\ncom lags 0, 1 e 2"]:::coral
    end

    subgraph GA1["GA₁ — identificação do modelo PA"]
        CROM["Cromossomo: vetor de 36 floats\n(todos os betas serializados via to_array())"]:::purple
        POP["População inicial: 50 indivíduos\n±10% da semente analítica"]:::purple
        FIT1["Fitness = RMSE(Y_pred, Y_medido)\nMinimização — avalia apenas inválidos"]:::red
        OPS["Seleção: torneio k=3\nCrossover: Blend α=0.3\nMutação gaussiana adaptativa\nElitismo: 2 melhores preservados"]:::purple
        CONV["100 gerações\nConvergência do RMSE mínimo"]:::purple
        CROM --> POP --> FIT1 --> OPS --> CONV
    end

    subgraph OUT1["Saídas — modelo PA identificado"]
        C1["NARXModelConfig\nbetas otimizados"]:::green
        C2["RMSE / NMSE em dB\nGanho linear em dB"]:::green
        C3["Constelação IQ\nAM-AM · AM-PM · Convergência GA"]:::green
    end

    XN --> NARX --> ENGINE --> GA1
    GA1 -->|"melhor indivíduo"| OUT1
```

---

## Diagrama 2 — Fluxo de Execução do Pipeline (Sequencial)

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#EEEDFE", "primaryBorderColor": "#534AB7", "primaryTextColor": "#26215C", "lineColor": "#888780", "fontSize": "13px"}}}%%
sequenceDiagram
    autonumber
    actor U as Usuário
    participant P as main_pipeline.py
    participant D as load_or_generate_data()
    participant M as NARXModelConfig
    participant E as NARXEngine
    participant G as PALinearizationGA
    participant V as plots.py

    U->>P: python main_pipeline.py
    P->>D: csv_path="dadosIniciais.csv"
    D-->>P: X[n] complex, Y[n] complex (48384 amostras)

    P->>M: NARXModelConfig() — semente analítica
    P->>E: NARXEngine(seed_config)
    E-->>P: Y_pred_seed[n]
    P->>E: rmse(Y_true, Y_pred_seed)
    E-->>P: RMSE_seed, NMSE_seed

    P->>G: PALinearizationGA(seed_config, X, Y, GAConfig)
    note over G: Geração 0: avalia 50 indivíduos
    loop Para cada geração 1..100
        G->>G: selecionar pais (torneio k=3)
        G->>G: crossover Blend α=0.3
        G->>G: mutação gaussiana adaptativa
        G->>G: avaliar apenas indivíduos inválidos
        G->>G: atualizar HallOfFame
    end
    G-->>P: best_config (NARXModelConfig), logbook

    P->>E: NARXEngine(best_config).predict(X)
    E-->>P: Y_pred_opt[n]
    P->>E: compute_gain(X, Y_pred_opt)
    E-->>P: G_linearized[n]

    P->>P: build_dataset(X, Y, Y_seed, Y_opt, G)
    P-->>U: pa_linearization_results.csv

    P->>V: plot_constellation(X, Y_seed, Y_opt)
    P->>V: plot_am_am_pm(X, Y_seed, Y_opt)
    P->>V: plot_ga_convergence(logbook)
    P->>V: plot_full_dashboard(...)
    V-->>U: constellation.png, am_am_pm.png,\nga_convergence.png, dashboard.png
```

---

## Diagrama 3 — Loop Evolutivo Interno do GA (Fluxo de Controle)

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#EEEDFE", "primaryBorderColor": "#534AB7", "primaryTextColor": "#26215C", "lineColor": "#888780", "fontSize": "13px"}}}%%
flowchart TD

    classDef gray   fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef coral  fill:#FAECE7,stroke:#993C1D,color:#4A1B0C
    classDef purple fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef teal   fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef green  fill:#EAF3DE,stroke:#3B6D11,color:#173404
    classDef red    fill:#FCEBEB,stroke:#A32D2D,color:#501313
    classDef amber  fill:#FAEEDA,stroke:#854F0B,color:#412402

    START(["GALinearizationGA.run()"]):::gray

    INIT["Gerar população inicial\n50 indivíduos ±10% da semente\npop[0] = semente analítica original"]:::purple

    EVAL0["Avaliar todos os 50 indivíduos\nfitness = RMSE(Y_pred, Y_true)"]:::red

    HOF0["Atualizar HallOfFame\n(melhor indivíduo global)"]:::green

    CHECK{"gen ≤ n_generations?"}:::gray

    ELITE["Copiar n_elites=2 melhores\nsem modificação"]:::green

    SELECT["Seleção por torneio k=3\nselecionar pop − 2 pais"]:::purple

    CROSS{"random() < 0.70?"}:::gray
    CROSSOP["cxBlend(α=0.3)\nfilhos entre e além dos pais\ninvalidar fitness dos filhos"]:::purple
    SKIP1["Manter pais sem crossover"]:::gray

    MUT{"random() < 0.20\npor indivíduo?"}:::gray
    MUTOP["Mutação gaussiana adaptativa\nσ = |gene| × 0.05 + ε\ninvalidar fitness"]:::purple
    SKIP2["Manter indivíduo sem mutação"]:::gray

    REGROUP["Reagrupar: elites + offspring"]:::purple

    EVALINV["Avaliar apenas indivíduos inválidos\n(lazy evaluation — economiza fitness calls)"]:::red

    HOF["Atualizar HallOfFame"]:::green

    LOG["Registrar no Logbook\nmin / avg / max / std do RMSE"]:::teal

    CB{"on_generation\ncallback definido?"}:::gray
    CBCALL["Chamar on_generation(gen, logbook)"]:::teal

    INC["gen += 1"]:::gray

    END(["Retornar best_config, logbook"]):::green

    START --> INIT --> EVAL0 --> HOF0 --> CHECK
    CHECK -->|"Sim"| ELITE
    CHECK -->|"Não"| END
    ELITE --> SELECT --> CROSS
    CROSS -->|"Sim"| CROSSOP --> MUT
    CROSS -->|"Não"| SKIP1 --> MUT
    MUT -->|"Sim"| MUTOP --> REGROUP
    MUT -->|"Não"| SKIP2 --> REGROUP
    REGROUP --> EVALINV --> HOF --> LOG --> CB
    CB -->|"Sim"| CBCALL --> INC --> CHECK
    CB -->|"Não"| INC --> CHECK
```

---

## Diagrama 4 — Modelo de Dados e Relações entre Classes

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#EEEDFE", "primaryBorderColor": "#534AB7", "primaryTextColor": "#26215C", "lineColor": "#888780", "fontSize": "13px"}}}%%
classDiagram
    direction TB

    class NARXModelConfig {
        +Intercepts intercepts
        +AutoregressiveMatrix A1
        +AutoregressiveMatrix A2
        +ExogenousBetas betas
        +gain_linear_dB() float
    }

    class Intercepts {
        +float Yreal = -0.000530
        +float Yimg  = +0.000063
        +to_array() ndarray
    }

    class AutoregressiveMatrix {
        +float Yreal_to_Yreal
        +float Yimg_to_Yreal
        +float Yreal_to_Yimg
        +float Yimg_to_Yimg
        +to_matrix() ndarray_2x2
    }

    class ExogenousBetas {
        +float Xreal_Yreal = 23.4565
        +float Ximg_Yreal  = -1.4743
        +float[34] outros_betas
        +to_dict() Dict
        +to_array() ndarray_36
        +from_array(arr) ExogenousBetas
    }

    class NARXEngine {
        -NARXModelConfig config
        -ndarray _c
        -ndarray _A1
        -ndarray _A2
        -ExogenousBetas _b
        +predict(X) ndarray_complex
        +rmse(Y_true, Y_pred) float
        +nmse(Y_true, Y_pred) float
        -_build_regressors() ndarray_18
        -_beta_vector_yreal() ndarray_18
        -_beta_vector_yimg() ndarray_18
    }

    class GAConfig {
        +int population_size = 50
        +int n_generations = 100
        +float crossover_prob = 0.70
        +float mutation_prob = 0.20
        +float mutation_sigma_rel = 0.05
        +int tournament_size = 3
        +int n_elites = 2
        +int random_seed = 42
        +float init_perturbation = 0.10
    }

    class PALinearizationGA {
        -NARXModelConfig seed_config
        -ndarray X_input
        -ndarray Y_true
        -GAConfig ga_config
        -ndarray _seed_array
        +run() NARXModelConfig, Logbook
        -_fitness_function(individual) tuple
        -_adaptive_mutate(individual) tuple
        -_setup_deap()
        -_eval_seed() float
    }

    NARXModelConfig "1" *-- "1" Intercepts
    NARXModelConfig "1" *-- "2" AutoregressiveMatrix
    NARXModelConfig "1" *-- "1" ExogenousBetas
    NARXEngine "1" --> "1" NARXModelConfig : usa
    PALinearizationGA "1" --> "1" NARXEngine : instancia por fitness call
    PALinearizationGA "1" --> "1" GAConfig : configurado por
    PALinearizationGA "1" --> "1" NARXModelConfig : semente + retorna melhor
```

---

## Diagrama 5 — Estrutura dos Regressores Φ(X)

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#EEEDFE", "primaryBorderColor": "#534AB7", "primaryTextColor": "#26215C", "lineColor": "#888780", "fontSize": "13px"}}}%%
flowchart LR

    classDef gray   fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef coral  fill:#FAECE7,stroke:#993C1D,color:#4A1B0C
    classDef purple fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef teal   fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef amber  fill:#FAEEDA,stroke:#854F0B,color:#412402
    classDef red    fill:#FCEBEB,stroke:#A32D2D,color:#501313

    X["X[n] complexo\n= Xreal + j·Ximg"]:::gray

    subgraph LAGS["Janela de memória da entrada"]
        L0["X(t)\nlag 0"]:::teal
        L1["X(t-1)\nlag 1"]:::teal
        L2["X(t-2)\nlag 2"]:::teal
    end

    subgraph DEG1["deg = 1  (linear)\n6 regressores"]
        D1["Xreal, Ximg\nXreal_l1, Ximg_l1\nXreal_l2, Ximg_l2"]:::teal
    end

    subgraph DEG2["deg = 2  (quadrático)\n6 regressores"]
        D2["Xreal², Xreal_l1², Xreal_l2²\nXimg², Ximg_l1², Ximg_l2²"]:::amber
    end

    subgraph DEG3["deg = 3  (cúbico)\n6 regressores"]
        D3["Xreal³, Xreal_l1³, Xreal_l2³\nXimg³, Ximg_l1³, Ximg_l2³"]:::red
    end

    PHI["Φ(X) — vetor de 18 regressores"]:::purple

    subgraph SAIDA["Produto com betas → contribuição exógena"]
        YR["β_Yreal · Φ(X)\n→ Re(Y(t))"]:::coral
        YI["β_Yimg · Φ(X)\n→ Im(Y(t))"]:::coral
    end

    X --> LAGS
    LAGS --> DEG1 --> PHI
    LAGS --> DEG2 --> PHI
    LAGS --> DEG3 --> PHI
    PHI --> YR
    PHI --> YI
```
