from src.core.base_model import BaseModel
import numpy as np

class GeneralizedMemoryPolynomial(BaseModel):

    def __init__(self,memory, r_deg, i_deg, lag_depth, lead_depth, **kw):
        super().__init__(memory=memory, **kw)
        self.r_deg = r_deg
        self.i_deg = i_deg
        self.lag_depth = lag_depth
        self.lead_depth = lead_depth

        self.trim_front = memory + lag_depth
        self.trim_back = lead_depth

        self.term_names = None

    def build_regressors(self, x, y=None):
        xreal, ximag = x.real, x.imag
        M = self.memory
        columns, names = [], []
        channels = [(xreal, self.r_deg, "xr"), (ximag, self.i_deg, "xi")]

        for channel, deg, ch_lbl in channels:
            for m in range(M + 1):
                sample = self._shift(channel, m)
                s_lbl = self._lbl(ch_lbl, m)  # rótulo da amostra

                c, n = self._termos(sample, sample, 0, deg, s_lbl, s_lbl)  # alinhado
                columns.extend(c);
                names.extend(n)

                for l in range(1, self.lag_depth + 1):  # lagging
                    env = self._shift(channel, m + l)
                    e_lbl = self._lbl(ch_lbl, m + l)
                    c, n = self._termos(sample, env, 1, deg, s_lbl, e_lbl)
                    columns.extend(c);
                    names.extend(n)

                for l in range(1, self.lead_depth + 1):  # leading
                    env = self._shift(channel, m - l)
                    e_lbl = self._lbl(ch_lbl, m - l)  # m-l pode ser negativo
                    c, n = self._termos(sample, env, 1, deg, s_lbl, e_lbl)
                    columns.extend(c);
                    names.extend(n)

        self.term_names = names
        return np.column_stack(columns)


    def _termos(self, sample, envelope, p_ini, p_end, sample_lbl, env_lbl):
        columns, names = [], []
        for p in range(p_ini, p_end):
            columns.append(sample * np.abs(envelope) ** p)
            names.append(f"{sample_lbl}*|{env_lbl}|^{p}")
        return columns, names

    @staticmethod
    def _lbl(ch, d):
        # d = atraso; d>=0 vira "[n-d]", d<0 vira "[n+|d|]" (futuro)
        return f"{ch}[n-{d}]" if d >= 0 else f"{ch}[n+{-d}]"