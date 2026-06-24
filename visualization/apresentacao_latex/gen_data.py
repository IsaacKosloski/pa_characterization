"""Gera CSVs enxutos para o pgfplots a partir de pa_linearization_results.csv."""
import numpy as np, pandas as pd

src = "/mnt/user-data/uploads/pa_linearization_results.csv"
out = "/home/claude/beamer/data/"
df = pd.read_csv(src)

X  = df.Xreal.values + 1j*df.Ximg.values
Ym = df.Yreal.values + 1j*df.Yimg.values
Ys = df.Ypred_seed_real.values + 1j*df.Ypred_seed_img.values
Yo = df.Ypred_opt_real.values + 1j*df.Ypred_opt_img.values
Xmag = np.abs(X)

def wrap(d):
    return (d + 180) % 360 - 180

gain_m = 20*np.log10(np.abs(Ym)/Xmag)
gain_o = 20*np.log10(np.abs(Yo)/Xmag)
ph_m = wrap(np.degrees(np.angle(Ym) - np.angle(X)))
ph_o = wrap(np.degrees(np.angle(Yo) - np.angle(X)))

# ---- downsample para scatter (~1300 pts) ----
n = len(df); stride = max(1, n//1300)
idx = np.arange(0, n, stride)

pd.DataFrame({"xmag": Xmag[idx], "gm": gain_m[idx], "go": gain_o[idx]}
             ).to_csv(out+"amam.csv", index=False)
pd.DataFrame({"xmag": Xmag[idx], "pm": ph_m[idx], "po": ph_o[idx]}
             ).to_csv(out+"ampm.csv", index=False)
pd.DataFrame({"i": Ym[idx].real, "q": Ym[idx].imag}).to_csv(out+"const_meas.csv", index=False)
pd.DataFrame({"i": Yo[idx].real, "q": Yo[idx].imag}).to_csv(out+"const_pred.csv", index=False)
pd.DataFrame({"i": X[idx].real, "q": X[idx].imag}).to_csv(out+"const_in.csv", index=False)

# ---- erro por amostra (primeiras 2000) ----
err = np.abs(Ym - Yo)
k = np.arange(2000)
pd.DataFrame({"n": k, "e": err[:2000]}).to_csv(out+"error.csv", index=False)

# ---- RMSE semente vs otimizado ----
rmse_s = np.sqrt(np.mean(np.abs(Ym-Ys)**2))
rmse_o = np.sqrt(np.mean(np.abs(Ym-Yo)**2))
with open(out+"rmse.csv","w") as f:
    f.write("modelo,rmse\n")
    f.write(f"Semente (VARMAX),{rmse_s:.5f}\n")
    f.write(f"GA otimizado,{rmse_o:.5f}\n")

# ---- simetria rotacional (do modelo, termos instantaneos) ----
def model_out(Xr, Xi):
    Yr = (23.4565*Xr - 1.4743*Xi - 0.0013038*Xr**2 - 0.0078043*Xi**2
          - 0.463307*Xr**3 + 2.04176*Xi**3)
    Yi = (1.47499*Xr + 23.4572*Xi + 0.0019282*Xr**2 + 0.0041005*Xi**2
          - 2.04273*Xr**3 - 0.467104*Xi**3)
    return Yr + 1j*Yi
ang = np.linspace(0, 360, 361)
rad = np.radians(ang)
cols = {"ang": ang}
for A, tag in [(0.3,"03"),(0.6,"06"),(1.0,"10"),(1.15,"115")]:
    Xr, Xi = A*np.cos(rad), A*np.sin(rad)
    Xc = Xr+1j*Xi; Yc = model_out(Xr, Xi)
    cols["g"+tag] = 20*np.log10(np.abs(Yc)/np.abs(Xc))
    cols["p"+tag] = wrap(np.degrees(np.angle(Yc)-np.angle(Xc)))
pd.DataFrame(cols).to_csv(out+"rot.csv", index=False)

print("RMSE semente =", round(rmse_s,5), " | RMSE otim =", round(rmse_o,5),
      " | melhora =", round(100*(rmse_s-rmse_o)/rmse_s,2), "%")
print("pts scatter =", len(idx))
print("CSVs gerados em", out)
