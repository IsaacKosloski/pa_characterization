import numpy as np
from src.core.estimators import OLS

rng = np.random.default_rng(0)
Phi = rng.standard_normal((200,3)) # 200 amostras, 3 features
theta_true = np.array([2.0, -1.0, 0.5])
y = Phi @ theta_true               # y sem ruído

est = OLS().fit(Phi, y)
print("theta estimado: ", est.theta)
assert np.allclose(est.theta, theta_true, atol=1e-8), "OLS não recuperou theta!"

y_hat = est.predict(Phi)
assert np.allclose(y_hat, y, atol=1e-8), "predict errado!"
print("Etapa 2 ok")