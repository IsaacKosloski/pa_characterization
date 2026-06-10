"""
Algoritmo Genético (GA) para otimização dos coeficientes
Beta do modelo NARX do PA.

Objetivo: encontrar o vetor de betas que minimiza o RMSE
entre o sinal predito pelo modelo e o sinal medido Y[n],
com a perspectiva de descobrir o G ótimo/linearizado.

Estrutura do GA:
    Representação : vetor real (float64) de tamanho N_betas (36)
    Seleção       : Torneio (k=3)
    Cruzamento    : Blend crossover (cxBlend, alpha=0.3)
    Mutação       : Gaussiana (mu=0, sigma adaptativo por magnitude)
    Elitismo      : copia os 2 melhores indivíduos sem mutação
    Aptidão       : minimizar RMSE(Y_true, Y_pred)
"""

import random
import warnings
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from deap import base, creator, tools, algorithms

from core.model_config import NARXModelConfig, ExogenousBetas, Intercepts, AutoregressiveMatrix
from core.narx_engine import NARXEngine

# --- Configuração do GA
@dataclass
class GAConfig:
    """
    Hiperparâmetros do Algoritmo Genético. Concentrar aqui facilita experimentação (tuning).
    """
    population_size:    int   = 50           # Número de indivíduos na população
    n_generations:      int   = 100          # Máximo de gerações
    crossover_prob:     float = 0.7     # P(crossover) por par de pais
    mutation_prob:      float = 0.2     # P(mutação) por indivíduo
    mutation_sigma_rel: float = 0.05    # Desvio relativo da mutação gaussiana
    tournament_size:    int   = 3            # k do torneio de seleção
    n_elites:           int   = 2            # Elitismo: melhores preservados
    random_seed:        int   = 42           # Reprodutibilidade
    # Perturbação inicial em torno da semente (range relativo)
    init_perturbation:  float = 0.10    # ±10% dos valores originais

# --- Algoritmo Genético Principal
class PALinearizationGA:
    """
    GA para linearização de PA via otimização dos betas NARX.

    :parameters
    seed_config : NARXModelConfig
        Modelo base (semente). A população inicial é gerada perturbando os betas desse modelo.
    X_input : np.ndarray complex
        Sinal de entrada U[n] / X[n] do PA.
    Y_true : np.ndarray complex
        Sinal de saída medido Y[n] do PA (ground truth).
    ga_config : GAConfig
        Hiperparâmetros do GA.
    on_generation : Callable, optional
        Callback chamado a cada geração com (gen, logbook). Útil para logging externo ou early-stopping.
    """

    def __init__(
            self,
            seed_config: NARXModelConfig,
            X_input: np.ndarray,
            Y_true: np.ndarray,
            ga_config: GAConfig = GAConfig(),
            on_generation: Optional[Callable] = None,
    ):
        self.seed_config = seed_config
        self.X_input = X_input
        self.Y_true = Y_true
        self.ga_config = ga_config
        self.on_generation = on_generation

        # Vetor semente dos betas (cromossomo base)
        self._seed_array = seed_config.betas.to_array()
        self._n_genes = len(self._seed_array)  # = 36 betas

        # Configura DEAP (evita redeclarar em chamadas repetidas)
        self._setup_deap()

    # --- Setup do DEAP
    def _setup_deap(self):
        """
        Registra tipos e operadores no toolbox do DEAP.

        DEAP usa um sistema de "criadores" para definir:
            - FitnessMin: minimiza uma função escalar (weights=(-1.0,))
            - Individual: lista de floats com atributo .fitness
        """
        # Limpa registros anteriores (evita conflito em re-execuções)
        for name in ["FitnessMin", "Individual"]:
            if hasattr(creator, name):
                delattr(creator, name)
        # -- Tipos
        # weights=(-1.0,) → minimização (RMSE)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # -- Toolbox
        self.toolbox = base.Toolbox()

        # Geração de gene: perturbação gaussiana em torno da semente
        cfg = self.ga_config

        def _make_gene(seed_val: float) -> float:
            """Gera um beta inicial perturbado em torno do valor semente."""
            sigma = abs(seed_val) * cfg.init_perturbation + 1e-6
            return random.gauss(seed_val, sigma)

        def _init_individual():
            """Cria um indivíduo (cromossomo) com betas perturbados."""
            genes = [_make_gene(v) for v in self._seed_array]
            return creator.Individual(genes)

        self.toolbox.register("individual", _init_individual)
        self.toolbox.register("population", tools.initRepeat,list, self.toolbox.individual)

        # -- Operadores Genéticos
        # Avaliação
        self.toolbox.register("evaluate", self._fitness_function)

        # Seleção por torneio (pressão seletiva moderada)
        self.toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_size)

        # Blend crossover: offspring entre e além dos pais (alpha=0.3)
        # Preserva boas regiões, mas permite exploração
        self.toolbox.register("mate", tools.cxBlend, alpha=0.3)

        # Mutação gaussiana adaptativa: sigma proporcional à magnitude
        # betas grandes podem variar mais; betas pequenos, menos
        self.toolbox.register("mutate", self._adaptive_mutate)

    # --- Função de Aptidão
    def _fitness_function(self, individual: list) -> Tuple[float]:
        """
        Avalia a aptidão de um indivíduo.

        1. Reconstrói ExogenousBetas a partir do cromossomo.
        2. Cria uma cópia do NARXModelConfig com os novos betas.
        3. Executa a predição com NARXEngine.
        4. Calcula e retorna o RMSE.

        O DEAP exige que a função retorne uma TUPLA.
        """
        try:
            arr = np.array(individual, dtype=float)

            # Reconstrói o modelo com os novos betas
            new_betas = ExogenousBetas.from_array(arr)
            new_config = NARXModelConfig(
                intercepts=self.seed_config.intercepts,
                A1=self.seed_config.A1,
                A2=self.seed_config.A2,
                betas=new_betas,
            )

            engine = NARXEngine(new_config)
            Y_pred = engine.predict(self.X_input)
            rmse = NARXEngine.rmse(self.Y_true, Y_pred)

            # Penalidade por NaN/Inf (indivíduo inválido)
            if not np.isfinite(rmse):
                return (1e9,)

            return (rmse,)

        except Exception:
            return (1e9,)

    # --- Mutação Adaptativa
    def _adaptive_mutate(self, individual: list) -> Tuple[list]:
        """
        Mutação gaussiana onde sigma é proporcional à magnitude
        de cada gene (beta).

        Vantagem: betas grandes (ex: Xreal_Yreal ≈ 23.45) podem
        explorar mais; betas minúsculos (ex: lag terms ≈ 0.001)
        recebem perturbações menores → mais estabilidade.
        """
        cfg = self.ga_config
        for i, gene in enumerate(individual):
            if random.random() < cfg.mutation_prob:
                sigma = abs(gene) * cfg.mutation_sigma_rel + 1e-6
                individual[i] += random.gauss(0, sigma)
        return (individual,)

    # --- Execução Principal do GA
    def run(self) -> Tuple[NARXModelConfig, tools.Logbook]:
        """
        Executa o loop evolutivo completo.

        Retorna
        -------
        best_config : NARXModelConfig
            Configuração do modelo com os betas otimizados.
        logbook : tools.Logbook
            Histórico de estatísticas por geração
            (min/avg/max do RMSE).
        """
        cfg = self.ga_config
        random.seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)

        # ── 1. Gerar população inicial ────────────────────────
        pop = self.toolbox.population(n=cfg.population_size)

        # Inserir a semente original como o primeiro indivíduo
        # (garante que o GA começa de um ponto conhecido)
        seed_individual = creator.Individual(self._seed_array.tolist())
        pop[0] = seed_individual

        # ── 2. Estatísticas ───────────────────────────────────
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("std", np.std)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "min", "avg", "max", "std"]

        # Hall of Fame: preserva o melhor indivíduo de todas as gerações
        hof = tools.HallOfFame(1)

        # ── 3. Avaliação inicial ──────────────────────────────
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)
        print(f"Gen 0 | RMSE min: {record['min']:.6f} | avg: {record['avg']:.6f}")

        # ── 4. Loop evolutivo ─────────────────────────────────
        for gen in range(1, cfg.n_generations + 1):

            # Elitismo: preserva os N melhores sem modificação
            elites = tools.selBest(pop, cfg.n_elites)
            elites = [deepcopy(e) for e in elites]

            # Seleção dos pais (tamanho da população - elites)
            offspring = self.toolbox.select(pop, len(pop) - cfg.n_elites)
            offspring = list(map(deepcopy, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cfg.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutação
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                if not mutant.fitness.valid:
                    del mutant.fitness.values

            # Reagrupa: elites + offspring mutados
            pop[:] = elites + offspring

            # Avalia apenas os inválidos (eficiência computacional)
            invalid = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid), **record)

            if gen % 10 == 0:
                print(
                    f"Gen {gen:4d} | "
                    f"RMSE min: {record['min']:.6f} | "
                    f"avg: {record['avg']:.6f} | "
                    f"std: {record['std']:.6f}"
                )

            # Callback externo
            if self.on_generation:
                self.on_generation(gen, logbook)

        # ── 5. Extrair melhor solução ─────────────────────────
        best_array = np.array(hof[0], dtype=float)
        best_betas = ExogenousBetas.from_array(best_array)
        best_config = NARXModelConfig(
            intercepts=self.seed_config.intercepts,
            A1=self.seed_config.A1,
            A2=self.seed_config.A2,
            betas=best_betas,
        )

        print(f"\nEvolução concluída")
        print(f"   RMSE inicial (semente): {self._eval_seed():.6f}")
        print(f"   RMSE final   (melhor):  {hof[0].fitness.values[0]:.6f}")
        print(f"   Ganho linear estimado: {best_config.gain_linear_dB:.2f} dB")

        return best_config, logbook

    def _eval_seed(self) -> float:
        """Avalia o RMSE do modelo semente (antes da evolução)."""
        engine = NARXEngine(self.seed_config)
        Y_pred = engine.predict(self.X_input)
        return NARXEngine.rmse(self.Y_true, Y_pred)