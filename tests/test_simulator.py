from src.engine.simulator import Simulator

sim = Simulator(model)
sim.to_csv("teste_sim.csv", x=test.x, y_true=test.y)
# abra o teste_sim.csv: deve ter as 6 colunas e len = len(test) - trim