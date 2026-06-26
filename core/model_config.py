"""
Configuração central do modelo NARX (VARMAX sem MA com não-linearidades) para o PA.

O modelo represena mateaticamente como o PA transforma o sinal de entrada X[n] (complexo) no sinal de Saída Y[n] (complexo), capturando:
    1. Memória Linear (das matrizes autorregressivas A1 e A2): efeitos de realimentação da saída passada.
    2. Não-linearidade (mônico de betas com deg > 1): a compreessão de ganho e geração de harmônicos do PA.
    3. Efeitos de memória da entrada (mônicos de betas com lags): dispersão temporal causada por capacitâncias parasitas, efeitos térmicos e de armadilhas do transistor.

    Estrutura da equação:
    Y(t) = intercepto
         + A_1 * Y(t-1) + A_2 * Y(t-2)      := Parte AR
         + sum(beta_k * phi_k(X(t),lags))   := Parte Exógena
Onde phi_k são os regressores: Xreal^deg, Ximg^deg, com lags 0, 1 e 2
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import numpy as np

# --- Interceptos
@dataclass
class Intercepts:
    """
    Bias do modelo (offset DC).
    Fisicamente é o desbalanceamento de tensão no ponto de operação do PA.
    Valores próximos de zero indicam um PA bem polarizado.
    @:return Interceptos como vetor coluna [Yreal, Yimg]
    """
    Yreal: float = -0.000530
    Yimg : float =  0.000063

    def to_array(self) -> np.ndarray:
        return np.array([self.Yreal, self.Yimg])

# --- Matrizes Autorregressivas (A_1, A_2)
@dataclass
class AutoregressiveMatrix:
    """
    Matriz 2x2 que captura a dependência da saída Y(t) em relação a uma saída passada Y(t-lag).

    Layout da matriz:
        [[Yreal(t-k)->Yreal(t),  Yimg(t-k)->Yreal(t)],
         [Yreal(t-k)->Yimg(t),   Yimg(t-k)->Yimg(t) ]]
    Fisicamente modela a memória do PA causada por efeitos de self-heating, armadilhas de carga (traps) e impedâncias parasitas que fazem o ganho atual depender de amplificações anteriores.

    Valores pequenos (<<1) indicam que a memória é fraca, ou seja, o PA não tem muita histerese. Esse comportamento é típico em PAs GaN (nitreto de gálio) bem projetados.
    """
    # Linha 01: Contribuição para Yreal(t)
    Yreal_to_Yreal: float = 0.0
    Yimg_to_Yreal : float = 0.0
    #Linha 02: Contribuição para Yimg(t)
    Yreal_to_Yimg : float = 0.0
    Yimg_to_Yimg  : float = 0.0

    def to_matrix(self) -> np.ndarray:
        """
        :return: Matriz 2x2: A * Y(t-k)
        """
        return np.array([[self.Yreal_to_Yreal, self.Yimg_to_Yreal],[self.Yreal_to_Yimg, self.Yimg_to_Yimg]])

# --- Coeficientes Beta
@dataclass
class ExogenousBetas:
    """
    Coeficientes que mapeiam a entrada X[n] para a saída Y[n].
    Conveção de nomes:
        beta.X{componente}_lag{k}_deg{d}_Y{saída}

    Onde:
        componente : 'real' ou 'img'
        lag k      : atraso temporal (0 = instante atual, 1 = t-1, ...)
        deg d      : grau do polinômio (1=linear, 2=quadrático, 3=cúbico)
        saída      : 'Yreal' ou 'Yimg'

    Fisicamente:
        - deg=1 (linear)    : ganho nominal do PA (~23.45 = 27.4 dB) (onde pode-se verificar a transmissão direta da entrada pra a saída).
        - deg=2 (quadrático): distorção de 2º harmônico (HD2), IM2
        - deg=3 (cúbico)    : compressão de ganho (AM-AM), rotação (AM-PM), IM3 (críticos para linearidade)
        - lags              : memória de curto prazo da entrada
    """
    # -- Betas para Yreal
    # Termos lineares (deg=1)
    Xreal_Yreal             : float =  23.4565211336
    Ximg_Yreal              : float = -1.4743037880
    Xreal_lag1_Yreal        : float =  0.0006146895
    Ximg_lag1_Yreal         : float = -0.0020095820
    Xreal_lag2_Yreal        : float = -0.0011283101
    Ximg_lag2_Yreal         : float = -0.0033082169

    # Termos quadráticos (deg=2)
    Xreal_deg2_Yreal        : float = -0.0013038188
    Xreal_lag1_deg2_Yreal   : float =  0.0048655646
    Xreal_lag2_deg2_Yreal   : float = -0.0044189839
    Ximg_deg2_Yreal         : float = -0.0078042871
    Ximg_lag1_deg2_Yreal    : float =  0.0007715753
    Ximg_lag2_deg2_Yreal    : float = -0.0047246018

    # Termos cúbicos (deg=3) — principais responsáveis pelo IM3
    Xreal_deg3_Yreal        : float = -0.4633073724
    Xreal_lag1_deg3_Yreal   : float = -0.0017976227
    Xreal_lag2_deg3_Yreal   : float =  0.0034993903
    Ximg_deg3_Yreal         : float =  2.0417623072
    Ximg_lag1_deg3_Yreal    : float =  0.0059627874
    Ximg_lag2_deg3_Yreal    : float =  0.0050378128

    # -- Betas para Yimg
    # Termos lineares (deg=1)
    Xreal_Yimg              : float =  1.4749853770
    Ximg_Yimg               : float =  23.4571530218
    Xreal_lag1_Yimg         : float =  0.0039931260
    Ximg_lag1_Yimg          : float =  0.0078289845
    Xreal_lag2_Yimg         : float = -0.0001370732
    Ximg_lag2_Yimg          : float = -0.0016653415

    # Termos quadráticos (deg=2)
    Xreal_deg2_Yimg         : float =  0.0019282380
    Xreal_lag1_deg2_Yimg    : float = -0.0033525384
    Xreal_lag2_deg2_Yimg    : float = -0.0047643685
    Ximg_deg2_Yimg          : float =  0.0041005096
    Ximg_lag1_deg2_Yimg     : float = -0.0007919437
    Ximg_lag2_deg2_Yimg     : float = -0.0003759096

    # Termos cúbicos (deg=3)
    Xreal_deg3_Yimg         : float = -2.0427333587
    Xreal_lag1_deg3_Yimg    : float = -0.0059792395
    Xreal_lag2_deg3_Yimg    : float = -0.0028998505
    Ximg_deg3_Yimg          : float = -0.4671041951
    Ximg_lag1_deg3_Yimg     : float = -0.0067674106
    Ximg_lag2_deg3_Yimg     : float =  0.0026973011

    def to_dict(self) -> Dict[str, float]:
        """Serializa todos os betas em dicionario chave->valor."""
        return asdict(self)

    def to_array(self) -> np.ndarray:
        """
        Vetoriza os betas em array numpy.
        Ordem: primeiro todos os betas de Yreal, depois de Yimg.
        Usado pelo Algoritmo Genético como cromossomo.
        """
        return np.array(list(self.to_dict().values()), dtype=float)
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ExogenousBetas":
        """
        Reconstrói Exogenous Betas a partir de um array (cromossomo do GA).
        Exige que a ordem corresponda exatamente à de to_array().
        """
        fields = list(cls.__dataclass_fields__.keys())
        if len(arr) != len(fields):
            raise ValueError(f"Array tem {len(arr)} elementos, mas o modelo exige {len(fields)} elementos.")
        return cls(**dict(zip(fields, arr)))


# --- Modelo NARX completo
@dataclass
class NARXModelConfig:
    """
    Contêiner completo do modelo NARX.
    Agrega interceptos, matrizes AR e coeficientes beta.
    """
    intercepts: Intercepts   = field(default_factory=Intercepts)

    A1: AutoregressiveMatrix = field(default_factory=lambda: AutoregressiveMatrix(
        Yreal_to_Yreal       =  0.0000028678,
        Yimg_to_Yreal        = -0.0000035027,
        Yreal_to_Yimg        =  0.0000041053,
        Yimg_to_Yimg         = -0.0000452185,
    ))

    A2: AutoregressiveMatrix = field(default_factory=lambda: AutoregressiveMatrix(
        Yreal_to_Yreal       = -0.0000115606,
        Yimg_to_Yreal        = -0.0000131692,
        Yreal_to_Yimg        =  0.0000304813,
        Yimg_to_Yimg         = -0.0000203335,
    ))

    betas: ExogenousBetas = field(default_factory=ExogenousBetas)

    @property
    def gain_linear_dB(self) -> float:
        """
        Ganho linear aproximado em dB.
        Deriva os betas de primeiro grau (Xreal->Yreal e Ximg->Yimg), assumindo simetria do canal (caso ideal: beta_cross = 0).
        """
        g = (self.betas.Xreal_Yreal + self.betas.Ximg_Yimg) / 2.0
        return 20.0 * np.log10(abs(g))