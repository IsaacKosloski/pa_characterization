# ADRs — Sessão 1: Identificação do PA com NARX + GA

> Architecture Decision Records — registro das decisões técnicas relevantes, com contexto, alternativas avaliadas e justificativa da escolha.

---

## ADR-001 — Modelo NARX com polinômio de memória em vez de rede neural

**Status:** Aceito  
**Data:** Sessão 1  
**Contexto:** Precisamos de um modelo caixa-cinza capaz de prever a saída complexa Y[n] do PA dado X[n], capturando não-linearidades e efeitos de memória.

**Alternativas avaliadas:**

| Opção | Vantagens | Desvantagens |
|---|---|---|
| **NARX polinomial** (escolhido) | Interpretável; coeficientes têm significado físico direto (ganho, IM3, memória); inversão analítica de 1ª ordem possível para seed do DPD | Expressividade limitada a grau 3 com lags fixos |
| Rede Neural (LSTM/RNN) | Alta capacidade de representação | Caixa-preta; sem inversão analítica; requer muito mais dados; latência de inferência maior |
| Volterra truncado | Base teórica sólida | Explosão combinatorial de termos com memória > 2 |
| Wiener-Hammerstein | Simples de identificar | Não captura memória distribuída nos termos cruzados I/Q |

**Decisão:** NARX polinomial com graus 1, 2, 3 e lags 0, 1, 2. O número de parâmetros (36 betas + interceptos + matrizes AR = 42 escalares) é gerenciável por GA e fornece interpretabilidade física imediata dos coeficientes.

**Consequência:** Modelo não captura não-linearidades de ordem > 3. Se o PA operar em compressão severa (P_in muito acima de P_1dB), considerar extensão para grau 5.

---

## ADR-002 — Algoritmo Genético (DEAP) em vez de otimização por gradiente

**Status:** Aceito  
**Data:** Sessão 1  
**Contexto:** Com 36 parâmetros, a superfície de erro do NARX é não-convexa (múltiplos mínimos locais gerados pelos termos cúbicos cruzados I/Q). Precisamos de um otimizador robusto a mínimos locais.

**Alternativas avaliadas:**

| Opção | Vantagens | Desvantagens |
|---|---|---|
| **GA com DEAP** (escolhido) | Robusto a mínimos locais; não requer gradiente; fácil impor restrições; semente analítica acelera convergência | Mais lento por avaliação; não garante ótimo global |
| OLS / Regressão linear | Solução fechada; extremamente rápido | Requer reformulação matricial; não explora não-convexidade |
| Adam / L-BFGS | Rápido; bem estabelecido | Sensível ao ponto inicial; preso em mínimos locais |
| PSO (Particle Swarm) | Boa exploração global | Convergência prematura em alta dimensão sem mecanismo de elitismo |
| Simulated Annealing | Fuga de mínimos locais garantida | Convergência muito lenta; sem paralelismo natural |

**Decisão:** DEAP foi escolhido por: (1) API declarativa limpa que separa representação, operadores e fitness; (2) suporte nativo a elitismo (HallOfFame); (3) Blend crossover adequado para espaços contínuos; (4) comunidade ativa e documentação sólida.

**Consequência:** Tempo de execução proporcional a `population_size × n_generations × N_amostras`. Com 48k amostras, 50 indivíduos e 100 gerações: ~15–40 minutos. Mitigação: avaliação lazy (só re-avalia indivíduos modificados).

---

## ADR-003 — Cromossomo como vetor plano de 36 floats

**Status:** Aceito  
**Data:** Sessão 1  
**Contexto:** O DEAP representa indivíduos como listas de Python. Precisamos mapear os 36 betas do `ExogenousBetas` em/out desse formato de forma determinística.

**Decisão:** Os métodos `ExogenousBetas.to_array()` e `from_array()` são o contrato único entre o modelo e o GA. A serialização usa `dataclasses.asdict()`, que garante ordem determinística dos campos pela ordem de declaração da dataclass.

```
Cromossomo[0..17]  → betas que afetam Yreal (18 genes)
Cromossomo[18..35] → betas que afetam Yimg  (18 genes)

Dentro de cada metade, ordem: linear_lag0, linear_lag1, linear_lag2,
quad_lag0..2, cubic_lag0..2 (para Xreal e Ximg separadamente)
```

**Alternativa rejeitada:** Dicionário como cromossomo — incompatível com operadores do DEAP (cxBlend, mutGaussian) que operam sobre sequências indexadas.

**Consequência:** Adicionar novos termos (ex: deg=5) exige atualizar `ExogenousBetas`, `to_array()`, `from_array()`, `_beta_vector_yreal()` e `_beta_vector_yimg()` de forma consistente. Os índices `_LINEAR_IDX` e `_NONLINEAR_IDX` da Sessão 2 também precisarão de atualização.

---

## ADR-004 — Mutação gaussiana adaptativa por magnitude do gene

**Status:** Aceito  
**Data:** Sessão 1  
**Contexto:** Os betas variam em 6 ordens de magnitude (`Xreal_Yreal ≈ 23.45` vs `Xreal_lag1 ≈ 0.0006`). Mutação com sigma fixo destruiria os betas pequenos ou seria insignificante para os grandes.

**Decisão:**

```python
sigma = abs(gene) * mutation_sigma_rel + epsilon
# mutation_sigma_rel = 0.05 → 5% do valor absoluto do gene
# epsilon = 1e-6  → garante mutação mesmo para genes ≈ 0
```

**Consequência:** Cada gene explora sua própria escala. Gene `= 23.45` → sigma `≈ 1.17`. Gene `= 0.0006` → sigma `≈ 3e-5`. Melhora significativamente a convergência em relação a sigma global fixo.

---

## ADR-005 — Matrizes AR (A₁, A₂) e interceptos mantidos fixos durante o GA

**Status:** Aceito  
**Data:** Sessão 1  
**Contexto:** O GA otimiza os 36 betas exógenos. Os interceptos e matrizes A₁, A₂ poderiam também ser incluídos no cromossomo.

**Decisão:** Manter interceptos e matrizes AR fixos nos valores da semente analítica. Razões:

1. **Estabilidade:** A₁ e A₂ com valores ≈ 10⁻⁵ garantem que o sistema AR seja BIBO-estável. Deixar o GA perturbar livremente poderia gerar sistemas instáveis (autovalores fora do círculo unitário).
2. **Separação de responsabilidades:** Os termos AR capturam memória de longo prazo (identificada por regressão antes do GA). Os betas capturam o comportamento entrada-saída, que é o alvo da linearização.
3. **Redução do espaço de busca:** O GA busca em ℝ³⁶ em vez de ℝ⁴² (+ 8 da AR + 2 interceptos), reduzindo o problema e acelerando a convergência.

**Consequência:** Se o modelo PA tiver memória expressiva (A₁ ou A₂ com valores > 0.01), considerar incluir esses parâmetros no cromossomo com restrição de estabilidade via penalidade no fitness.

---

## ADR-006 — Elitismo explícito em vez de usar `eaSimple` do DEAP

**Status:** Aceito  
**Data:** Sessão 1  
**Contexto:** O DEAP oferece `algorithms.eaSimple()` como loop evolutivo pronto. Optamos por implementar o loop manualmente.

**Decisão:** Loop manual com elitismo explícito:

```python
elites   = tools.selBest(pop, n_elites)           # copia os melhores
offspring = toolbox.select(pop, len(pop) - n_elites)  # seleciona o restante
# ... crossover + mutação em offspring ...
pop[:] = elites + offspring                        # reagrupa
```

**Vantagens sobre `eaSimple`:** (1) garante que RMSE mínimo nunca piora entre gerações; (2) permite callback `on_generation` para early-stopping ou logging externo; (3) avaliação lazy (`invalid` only) reduz o número de chamadas ao fitness.

**Consequência:** Código mais verboso (≈30 linhas extras), mas comportamento determinístico e auditável por geração.

---

## ADR-007 — `dataclasses` para estrutura do modelo em vez de dicionários ou YAML

**Status:** Aceito  
**Data:** Sessão 1  
**Contexto:** Os coeficientes do modelo precisam ser armazenados, passados entre funções, serializados para o GA e desserializados do cromossomo.

**Decisão:** `@dataclass` com valores padrão hard-coded como semente analítica.

**Vantagens:** (1) type hints nativos; (2) `asdict()` para serialização; (3) valores padrão documentam a semente no próprio código; (4) IDE autocomplete nos nomes dos betas; (5) imutável (se `frozen=True`) para o modelo PA na Sessão 2.

**Alternativas rejeitadas:**
- Dicionário Python: sem type hints, sem autocomplete, sem validação de campos.
- Arquivo YAML/JSON: adiciona dependência de parsing e separa a semente do código.
- Arquivo `.npy`: não é legível por humanos, dificulta inspeção dos coeficientes.
