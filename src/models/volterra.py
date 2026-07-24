from src.core.base_model import BaseModel
from itertools import combinations_with_replacement
import numpy as np

class Volterra(BaseModel):

    def __init__(self, memory=2, order_r=3, order_i=3, **kw):
        super().__init__(memory=memory, **kw)
        self.order_r = order_r
        self.order_i = order_i
        self.term_names = None

    def build_regressors(self,x, y=None):
        xr, xi = x.real, x.imag
        M = self.memory
        columns, names = [], []
        channels = [(xr, self.order_r, "xr"), (xi, self.order_i, "xi")]

        for channel, deg, ch_lbl in channels:
            for p in range(1, deg + 1):
                for combo in combinations_with_replacement(range(M + 1), p):
                    column = np.ones(len(channel), dtype=channel.dtype)
                    for d in combo:
                        column = column * self._shift(channel, d)
                    name = "*".join(f"{ch_lbl}[n-{d}]" for d in combo)

                    columns.append(column)
                    names.append(name)
        self.term_names = names
        return np.column_stack(columns)




