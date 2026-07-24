
import re, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from scipy.signal import welch
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

_C = {"true": "#455A64", "pred": "#4CAF50", "aic": "#FF9800", "bic": "#9C27B0",
      "rmse": "#2196F3", "grid": "#E0E0E0"}

# ---------- diretório numerado por rodada ----------
def create_run_directory(base="relatorios", prefix="relatorio"):
    base = Path(base); base.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    nums = [int(m.group(1)) for p in base.iterdir() if p.is_dir() and (m := pat.match(p.name))]
    d = base / f"{prefix}{max(nums, default=0)+1:02d}"; d.mkdir(); return d

# ---------- métricas TELECOM ----------
def metricas(X, Yt, Yp):
    err = Yt - Yp
    mse = np.mean(np.abs(err)**2); pwr = np.mean(np.abs(Yt)**2)
    amp = np.abs(X) + 1e-12
    g_true = np.abs(Yt) / amp
    fase = ((np.angle(Yt, deg=True) - np.angle(X, deg=True) + 180) % 360) - 180
    return {
        "n": len(X),
        "rmse": float(np.sqrt(mse)),
        "nmse_dB": float(10*np.log10(mse/(pwr+1e-15))),
        "evm_pct": float(np.sqrt(mse/(pwr+1e-15))*100),
        "ganho_dB": float(20*np.log10(np.mean(g_true))),
        "gain_flat_std": float(np.std(20*np.log10(g_true+1e-15))),
        "phase_flat_std": float(np.std(fase)),
        "papr_dB": float(10*np.log10(np.max(np.abs(X)**2)/np.mean(np.abs(X)**2))),
    }

def classificar(m):
    def v(x, ot, bo): return "EXCELENTE" if x <= ot else ("BOM" if x <= bo else "A MELHORAR")
    mag = -m["nmse_dB"]
    nmse = "EXCELENTE" if mag >= 40 else ("BOM" if mag >= 30 else "A MELHORAR")
    return {"nmse": nmse, "evm": v(m["evm_pct"], 1.5, 3.5),
            "gain": v(m["gain_flat_std"], 0.1, 0.5), "phase": v(m["phase_flat_std"], 0.5, 2.0)}

# ---------- seleção (AIC / BIC) ----------
def _n2ll(rmse, N): return N*np.log(rmse**2)
def aic(rmse, k, N, ppc=2): return _n2ll(rmse, N) + 2*(ppc*k)
def bic(rmse, k, N, ppc=2): return _n2ll(rmse, N) + (ppc*k)*np.log(N)

def pareto(df):
    d = df.sort_values("n_coef"); best = np.inf; keep = []
    for _, r in d.iterrows():
        if r["score"] < best: keep.append(r); best = r["score"]
    return pd.DataFrame(keep)

def selecoes(df):
    df = df.copy()
    df["aic"] = [aic(r.score, r.n_coef, r.N) for r in df.itertuples()]
    df["bic"] = [bic(r.score, r.n_coef, r.N) for r in df.itertuples()]
    return {"RMSE": df.loc[df.score.idxmin()], "AIC": df.loc[df.aic.idxmin()],
            "BIC": df.loc[df.bic.idxmin()]}, df

# ---------- equação (termos dominantes) ----------
def equacao(term_names, cr, ci, top=12):
    mag = np.sqrt(np.asarray(cr)**2 + np.asarray(ci)**2)
    ordem = np.argsort(mag)[::-1][:top]
    linhas = [f"  ({cr[k]:+.4g}{ci[k]:+.4g}j) · {term_names[k]}" for k in ordem]
    return "\n".join(linhas)

# ---------- carga de uma rodada ----------
def load_run(run_dir):
    run_dir = Path(run_dir)
    rep = json.load(open(run_dir/"report.json"))
    hist = pd.read_csv(run_dir/"gridsearch.csv")
    sim = pd.read_csv(run_dir/"simulation.csv")
    npz = np.load(run_dir/"coefficients.npz")
    X = sim["Xreal"].to_numpy() + 1j*sim["Ximg"].to_numpy()
    Yt = sim["Yreal"].to_numpy() + 1j*sim["Yimg"].to_numpy()
    Yp = sim["Ypred_real"].to_numpy() + 1j*sim["Ypred_img"].to_numpy()
    return dict(dir=run_dir, rep=rep, hist=hist, X=X, Yt=Yt, Yp=Yp,
                cr=npz["real"], ci=npz["imag"])

# ---------- painéis ----------
def painel_modelo(d, dest):
    X, Yt, Yp = d["X"], d["Yt"], d["Yp"]
    idx = np.linspace(0, len(X)-1, min(3000, len(X)), dtype=int)
    fig, ax = plt.subplots(2, 2, figsize=(14, 11), facecolor="white")
    fig.suptitle(f"{d['rep']['model']} — painel de caracterização", fontsize=15, fontweight="bold")
    # constelação
    a = ax[0,0]
    a.scatter(Yt[idx].real, Yt[idx].imag, c=_C["true"], s=5, alpha=.35, label="Medido")
    a.scatter(Yp[idx].real, Yp[idx].imag, c=_C["pred"], s=5, alpha=.35, label="Predito")
    a.set_title("Constelação IQ", fontweight="bold"); a.set_xlabel("I"); a.set_ylabel("Q")
    a.legend(markerscale=3); a.grid(True, alpha=.3); a.set_aspect("equal")
    # AM-AM
    a = ax[0,1]; amp = np.abs(X[idx])
    a.scatter(amp, np.abs(Yt[idx]), c=_C["true"], s=4, alpha=.35, label="Medido")
    a.scatter(amp, np.abs(Yp[idx]), c=_C["pred"], s=4, alpha=.35, label="Predito")
    a.set_title("AM-AM", fontweight="bold"); a.set_xlabel("|X[n]|"); a.set_ylabel("|Y[n]|")
    a.legend(markerscale=3); a.grid(True, alpha=.3)
    # AM-PM
    a = ax[1,0]; eps=1e-12
    a.scatter(amp, np.angle(Yt[idx]/(X[idx]+eps), deg=True), c=_C["true"], s=4, alpha=.35, label="Medido")
    a.scatter(amp, np.angle(Yp[idx]/(X[idx]+eps), deg=True), c=_C["pred"], s=4, alpha=.35, label="Predito")
    a.set_title("AM-PM", fontweight="bold"); a.set_xlabel("|X[n]|"); a.set_ylabel("∠Y-∠X (graus)")
    a.legend(markerscale=3); a.grid(True, alpha=.3)
    # PSD (regrowth espectral) — padrão telecom (contexto de ACPR)
    a = ax[1,1]
    if _HAS_SCIPY:
        f1,p1 = welch(Yt, nperseg=1024, return_onesided=False)
        f2,p2 = welch(Yp, nperseg=1024, return_onesided=False)
        o1=np.argsort(f1); o2=np.argsort(f2)
        a.plot(f1[o1], 10*np.log10(p1[o1]+1e-15), c=_C["true"], lw=1, label="Medido")
        a.plot(f2[o2], 10*np.log10(p2[o2]+1e-15), c=_C["pred"], lw=1, label="Predito")
        a.set_xlabel("Frequência normalizada")
    else:
        a.plot(20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(Yt)))+1e-9), c=_C["true"], lw=.5, label="Medido")
        a.plot(20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(Yp)))+1e-9), c=_C["pred"], lw=.5, label="Predito")
    a.set_title("PSD — espectro de saída", fontweight="bold"); a.set_ylabel("dB")
    a.legend(); a.grid(True, alpha=.3)
    plt.tight_layout(); fig.savefig(dest, dpi=150, bbox_inches="tight"); plt.close(fig)

def painel_fronteira(hist, sels, dest, nome):
    pf = pareto(hist).sort_values("n_coef")
    _, dfic = selecoes(hist)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle(f"{nome} — fronteira e seleção", fontsize=14, fontweight="bold")
    ax[0].plot(pf["n_coef"], pf["score"], "-o", c=_C["rmse"], label="fronteira de Pareto")
    for nm, cor in [("RMSE",_C["rmse"]),("AIC",_C["aic"]),("BIC",_C["bic"])]:
        s = sels[nm]; ax[0].scatter([s["n_coef"]],[s["score"]], c=cor, s=140, zorder=5,
                                    edgecolors="black", label=f"escolha {nm}")
    ax[0].set_xlabel("nº de coeficientes"); ax[0].set_ylabel("RMSE (val)")
    ax[0].set_title("RMSE × complexidade"); ax[0].legend(); ax[0].grid(True, alpha=.3)
    dfic_s = dfic.sort_values("n_coef")
    ax[1].plot(dfic_s["n_coef"], dfic_s["aic"], ".", c=_C["aic"], alpha=.5, label="AIC")
    ax[1].plot(dfic_s["n_coef"], dfic_s["bic"], ".", c=_C["bic"], alpha=.5, label="BIC")
    ax[1].set_xlabel("nº de coeficientes"); ax[1].set_ylabel("critério (menor=melhor)")
    ax[1].set_title("AIC / BIC × complexidade"); ax[1].legend(); ax[1].grid(True, alpha=.3)
    plt.tight_layout(); fig.savefig(dest, dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- montagem do relatório ----------
def gerar(root="output", base_rel="relatorios"):
    root = Path(root)
    runs = []
    for mdir in sorted(root.iterdir()):
        if not mdir.is_dir(): continue
        rr = sorted([p for p in mdir.glob("run*") if p.is_dir()])
        if rr: runs.append(rr[-1])                      # última rodada de cada modelo
    if not runs:
        print("nenhuma rodada encontrada em", root); return
    out = create_run_directory(base_rel)
    L = [f"# Relatório de Caracterização de PA", "",
         f"*Gerado em {pd.Timestamp.now():%Y-%m-%d %H:%M}*  ·  fonte: `{root}/`", "",
         "## 1. Comparação entre modelos", "",
         "| Modelo | nº coef | RMSE | NMSE (dB) | EVM (%) | Ganho (dB) | Veredito EVM |",
         "|---|---:|---:|---:|---:|---:|---|"]
    dados = []
    for rd in runs:
        d = load_run(rd); m = metricas(d["X"], d["Yt"], d["Yp"]); c = classificar(m)
        d["m"], d["c"] = m, c; dados.append(d)
        L.append(f"| {d['rep']['model']} | {d['rep']['n_coef']} | {m['rmse']:.4f} | "
                 f"{m['nmse_dB']:.2f} | {m['evm_pct']:.3f} | {m['ganho_dB']:.2f} | {c['evm']} |")
    melhor = min(dados, key=lambda d: d["m"]["rmse"])
    L += ["", f"**Melhor por RMSE:** {melhor['rep']['model']} "
          f"(RMSE={melhor['m']['rmse']:.4f}, EVM={melhor['m']['evm_pct']:.3f}%, "
          f"{melhor['rep']['n_coef']} coeficientes).", ""]
    # seções por modelo
    for d in dados:
        nome = d["rep"]["model"]; m = d["m"]; c = d["c"]
        painel_modelo(d, out/f"{nome}_painel.png")
        sels, _ = selecoes(d["hist"])
        painel_fronteira(d["hist"], sels, out/f"{nome}_fronteira.png", nome)
        L += [f"## Modelo: {nome}", "",
              f"- **Hiperparâmetros:** `{d['rep']['params']}`",
              f"- **Amostras (teste):** {m['n']:,}  ·  **PAPR entrada:** {m['papr_dB']:.2f} dB", "",
              "### Métricas (padrão TELECOM)", "",
              "| Métrica | Valor | Veredito |", "|---|---:|---|",
              f"| RMSE | {m['rmse']:.5f} | — |",
              f"| NMSE | {m['nmse_dB']:.2f} dB | {c['nmse']} |",
              f"| EVM | {m['evm_pct']:.3f} % | {c['evm']} |",
              f"| Ganho médio | {m['ganho_dB']:.2f} dB | — |",
              f"| Planicidade de ganho (σ) | {m['gain_flat_std']:.3f} dB | {c['gain']} |",
              f"| Planicidade de fase (σ) | {m['phase_flat_std']:.3f} ° | {c['phase']} |", "",
              "### Seleção de modelo (histórico do GridSearch)", "",
              f"O GridSearch avaliou **{len(d['hist'])}** combinações. Escolha por critério:", "",
              "| Critério | nº coef | RMSE |", "|---|---:|---:|"]
        for nm in ["RMSE", "AIC", "BIC"]:
            s = sels[nm]; L.append(f"| {nm} | {int(s['n_coef'])} | {s['score']:.4f} |")
        L += ["", f"![fronteira]({nome}_fronteira.png)", "",
              "### Equação do modelo (12 termos dominantes)", "", "```",
              equacao(d["rep"]["term_names"], d["cr"], d["ci"]), "```", "",
              f"![painel]({nome}_painel.png)", ""]
    (out/"relatorio.md").write_text("\n".join(L), encoding="utf-8")
    print("relatório em:", out)
    return out

if __name__ == "__main__":
    gerar(sys.argv[1] if len(sys.argv) > 1 else "output")
