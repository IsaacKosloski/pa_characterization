#!/bin/bash

# Arquitetura de diretórios desejado
# (...)

ROOT_DIR="pa_characterization"

# Diretórios
mkdir -p "$ROOT_DIR"/{data,src,experiments,results,tests}
mkdir -p "$ROOT_DIR"/data/{raw,processed,generated}
mkdir -p "$ROOT_DIR"/src/{data_engine,core,models,metrics,search,viz}
mkdir -p "$ROOT_DIR"/experiments/configs

# Arquivos Raiz, Dados e Configurações
touch "$ROOT_DIR"/main.py

# Python Files
touch "$ROOT_DIR"/src/data_engine/{waveform.py,pa_simulator.py,generate.py}
touch "$ROOT_DIR"/src/core/{base_model.py,estimators.py,dataset.py,registry.py}
touch "$ROOT_DIR"/src/models/{mp.py,gmp.py,volterra.py,narmax.py,fractional_mp.py,lstm.py}
touch "$ROOT_DIR"/src/metrics/{rmse.py,evm.py}
touch "$ROOT_DIR"/src/search/grid_search.py
touch "$ROOT_DIR"/src/viz/{am_am_pm.py,constellation.py}

# __init__.py files
touch "$ROOT_DIR"/src/__init__.py

for pasta in data_engine core models metrics search viz; do
  touch "$ROOT_DIR"/src/"$pasta"/__init__.py
done