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

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

# --- Interceptos
@dataclass
class Intercepts:
    """
    Bias do modelo (offset DC).
    Fisicamente é o desbalanceamento de tensão no ponto de operação do PA.
    Valores próximos de zero indicam um PA bem polarizado.
    """
    Yreal: float = -0.000530
    Yimg : float =  0.000063
    