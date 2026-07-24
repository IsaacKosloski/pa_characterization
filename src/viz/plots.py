import numpy as np
import matplotlib.pyplot as plt
from src.metrics.metrics import RMSE, EVM

_C = {"true": "#455A64", "pred": "#4CAF50", "grid": "#E0E0E0", "bg": "#FAFAFA"}


class Plotter:
    @staticmethod
    def _evm(yt, yp):
        return 100 * np.sqrt(np.mean(np.abs(yt - yp) ** 2)) / np.sqrt(np.mean(np.abs(yt) ** 2))

    @staticmethod
    def _sub(sig, n):
        return np.linspace(0, len(sig) - 1, min(n, len(sig)), dtype=int)

    @staticmethod
    def am_am(x, y_true, y_pred, path=None, ax=None):
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(6, 5), facecolor=_C["bg"])
        a = np.abs(x)
        ax.scatter(a, np.abs(y_true), c=_C["true"], alpha=.3, s=3, label="Medido")
        ax.scatter(a, np.abs(y_pred), c=_C["pred"], alpha=.3, s=3, label="Predito")
        ax.set_xlabel("|X[n]|"); ax.set_ylabel("|Y[n]|")
        ax.set_title("AM-AM", fontweight="bold")
        ax.legend(markerscale=4); ax.grid(True, color=_C["grid"]); ax.set_facecolor(_C["bg"])
        if own and path:
            fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

    @staticmethod
    def am_pm(x, y_true, y_pred, path=None, ax=None):
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(6, 5), facecolor=_C["bg"])
        eps = 1e-12; a = np.abs(x)
        pt = np.angle(y_true / (x + eps), deg=True)
        pp = np.angle(y_pred / (x + eps), deg=True)
        ax.scatter(a, pt, c=_C["true"], alpha=.3, s=3, label="Medido")
        ax.scatter(a, pp, c=_C["pred"], alpha=.3, s=3, label="Predito")
        ax.set_xlabel("|X[n]|"); ax.set_ylabel("∠Y - ∠X (graus)")
        ax.set_title("AM-PM", fontweight="bold")
        ax.legend(markerscale=4); ax.grid(True, color=_C["grid"]); ax.set_facecolor(_C["bg"])
        if own and path:
            fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

    @staticmethod
    def constellation(y_true, y_pred, path=None, ax=None, n=2000):
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(6, 6), facecolor=_C["bg"])
        i = Plotter._sub(y_true, n)
        ax.scatter(y_true[i].real, y_true[i].imag, c=_C["true"], alpha=.4, s=6, label="Medido")
        ax.scatter(y_pred[i].real, y_pred[i].imag, c=_C["pred"], alpha=.4, s=6, label="Predito")
        lim = np.abs(y_true).max() * 1.1
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_title("Constelação", fontweight="bold")
        ax.legend(markerscale=3); ax.grid(True, color=_C["grid"]); ax.set_facecolor(_C["bg"])
        ax.text(.05, .95, f"EVM: {Plotter._evm(y_true, y_pred):.2f}%",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.8))
        if own and path:
            fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

    @staticmethod
    def dashboard(x, y_true, y_pred, title="", path=None):
        """Painel 1x3: constelação + AM-AM + AM-PM, com RMSE/EVM no título."""
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=_C["bg"])
        Plotter.constellation(y_true, y_pred, ax=ax[0])
        Plotter.am_am(x, y_true, y_pred, ax=ax[1])
        Plotter.am_pm(x, y_true, y_pred, ax=ax[2])
        r = RMSE().compute(y_true, y_pred); e = EVM().compute(y_true, y_pred)
        fig.suptitle(f"{title}   —   RMSE={r:.4f}   EVM={e:.3f}%", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)