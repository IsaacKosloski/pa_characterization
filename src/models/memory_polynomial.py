from src.core.base_model import BaseModel
import numpy as np

class MemoryPolynomial(BaseModel):
    """Polinômio de memória clássico de telecomunicações,com termos todos ímpar-simétricos que elevam o ACLR e ACPR."""
    def __init__(self, memory=2, r_deg=3, i_deg=3, **kw):
        super().__init__(memory=memory, **kw)
        self.r_deg = r_deg
        self.i_deg = i_deg
        self.term_names = None

    def build_regressors(self,x, y=None):
        xreal = x.real
        ximag = x.imag
        N = len(x)
        M = self.memory

        columns = []
        names   = []
        for m in range(M + 1):
            xreal_delayed = self._shift(xreal, m)
            ximag_delayed = self._shift(ximag, m)
            for p in range(self.r_deg):
                column_real = xreal_delayed * (np.abs(xreal_delayed) ** p)
                columns.append(column_real)
                names.append(f"xr[n-{m}]*|xr[n-{m}]|^{p}")
            for p in range(self.i_deg):
                column_imag = ximag_delayed * (np.abs(ximag_delayed) ** p)
                columns.append(column_imag)
                names.append(f"xi[n-{m}]*|xi[n-{m}]|^{p}")
        Phi = np.column_stack(columns)
        self.term_names = names
        return Phi
