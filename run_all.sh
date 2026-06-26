#!/usr/bin/env bash
# =============================================================
#  run_all.sh  —  Orquestrador do projeto pa_linearization
# =============================================================
#  O que faz, em ordem:
#    1. Cria o ambiente virtual (.venv) se ainda não existir
#    2. Ativa o ambiente virtual
#    3. Instala as dependências do requirements.txt
#    4. Detecta o arquivo de dados em datasets/ (xlsx ou csv)
#    5. Roda o pipeline principal (main_pipeline.py)
#    6. Roda o gerador de relatório (gerar_relatorio.py)
#
#  Estrutura esperada (conforme o repositório):
#    pa_linearization/
#      ├── core/  ga/  visualization/
#      ├── datasets/           <- dados de entrada (xlsx/csv) aqui
#      ├── output/             <- resultados do pipeline
#      ├── relatorio/          <- relatório final
#      ├── main_pipeline.py
#      ├── gerar_relatorio.py
#      └── requirements.txt
#
#  Uso:
#    chmod +x run_all.sh      (só na primeira vez, no Linux/Mac)
#    ./run_all.sh             (roda tudo)
#    ./run_all.sh --gens 200 --pop 80   (sobrescreve hiperparâmetros)
# =============================================================

# 'set -e' faz o script parar no primeiro erro.
# 'set -u' trata variáveis não definidas como erro.
# 'set -o pipefail' faz um pipe falhar se qualquer etapa falhar.
set -euo pipefail

# ─────────────────────────────────────────────────────────────
#  Cores para deixar a saída legível
# ─────────────────────────────────────────────────────────────
VERDE="\033[0;32m"; AMARELO="\033[1;33m"; VERMELHO="\033[0;31m"; AZUL="\033[0;34m"; SEM="\033[0m"
info()  { echo -e "${AZUL}[INFO]${SEM} $1"; }
ok()    { echo -e "${VERDE}[OK]${SEM} $1"; }
aviso() { echo -e "${AMARELO}[AVISO]${SEM} $1"; }
erro()  { echo -e "${VERMELHO}[ERRO]${SEM} $1"; }

# ─────────────────────────────────────────────────────────────
#  0. Garante que rodamos a partir da pasta do script
# ─────────────────────────────────────────────────────────────
# "${BASH_SOURCE[0]}" é o caminho deste próprio script.
# Mudamos o diretório de trabalho para a pasta dele, assim os
# caminhos relativos (datasets/, output/...) sempre funcionam,
# não importa de onde o usuário chame o script.
cd "$(dirname "${BASH_SOURCE[0]}")"
info "Diretório de trabalho: $(pwd)"

# ─────────────────────────────────────────────────────────────
#  Parâmetros opcionais (com valores padrão)
# ─────────────────────────────────────────────────────────────
GENS=100
POP=50
# Lê argumentos no formato --gens N --pop N
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gens) GENS="$2"; shift 2 ;;
    --pop)  POP="$2";  shift 2 ;;
    *) aviso "Argumento desconhecido ignorado: $1"; shift ;;
  esac
done
info "Hiperparâmetros: gerações=${GENS}, população=${POP}"

# ─────────────────────────────────────────────────────────────
#  1. Descobrir o interpretador Python
# ─────────────────────────────────────────────────────────────
# Tenta python3 primeiro, depois python. Se nenhum existir, aborta.
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  erro "Python não encontrado. Instale o Python 3.10+ e tente de novo."
  exit 1
fi
info "Usando interpretador: $($PY --version)"

# ─────────────────────────────────────────────────────────────
#  2. Criar o ambiente virtual se não existir
# ─────────────────────────────────────────────────────────────
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
  info "Ambiente virtual não encontrado. Criando em ./${VENV_DIR} ..."
  $PY -m venv "$VENV_DIR"
  ok "Ambiente virtual criado."
else
  ok "Ambiente virtual já existe em ./${VENV_DIR} (reutilizando)."
fi

# ─────────────────────────────────────────────────────────────
#  3. Ativar o ambiente virtual
# ─────────────────────────────────────────────────────────────
# No Windows (Git Bash), o ativador fica em Scripts/activate.
# No Linux/Mac, fica em bin/activate. Detectamos os dois casos.
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  erro "Não encontrei o ativador do venv. Abortando."
  exit 1
fi
ok "Ambiente virtual ativado: $(which python)"

# ─────────────────────────────────────────────────────────────
#  4. Instalar dependências
# ─────────────────────────────────────────────────────────────
if [[ ! -f "requirements.txt" ]]; then
  erro "requirements.txt não encontrado na pasta do projeto."
  exit 1
fi
info "Atualizando pip e instalando dependências..."
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt
ok "Dependências instaladas."

# ─────────────────────────────────────────────────────────────
#  5. Detectar o arquivo de dados
# ─────────────────────────────────────────────────────────────
# Procura primeiro por .xlsx, depois por .csv, dentro de datasets/.
# 'find ... | head -1' pega o primeiro encontrado.
DATA_FILE=""
if [[ -d "datasets" ]]; then
  DATA_FILE="$(find datasets -maxdepth 1 -type f \( -iname '*.xlsx' -o -iname '*.csv' \) 2>/dev/null | head -1 || true)"
fi

if [[ -n "$DATA_FILE" ]]; then
  ok "Arquivo de dados detectado: $DATA_FILE"
else
  aviso "Nenhum dado encontrado em datasets/. O pipeline rodará com dados SINTÉTICOS."
fi

# Garante que as pastas de saída existam
mkdir -p output relatorio

# ─────────────────────────────────────────────────────────────
#  6. Preparar o CSV (converter xlsx -> csv se necessário)
# ─────────────────────────────────────────────────────────────
# O main_pipeline.py lê CSV. Se o dado for xlsx, convertemos aqui
# usando um pequeno trecho Python embutido (here-doc).
CSV_PARA_PIPELINE=""
if [[ -n "$DATA_FILE" ]]; then
  case "$DATA_FILE" in
    *.xlsx|*.XLSX|*.xls|*.XLS)
      info "Convertendo planilha para CSV..."
      CSV_PARA_PIPELINE="datasets/_dados_convertidos.csv"
      python - "$DATA_FILE" "$CSV_PARA_PIPELINE" <<'PYEOF'
import sys, pandas as pd
origem, destino = sys.argv[1], sys.argv[2]
df = pd.read_excel(origem, header=None).iloc[:, :4]
df.columns = ["Xreal", "Ximg", "Yreal", "Yimg"]
df.to_csv(destino, index=False)
print(f"  {len(df)} amostras convertidas -> {destino}")
PYEOF
      ok "Conversão concluída."
      ;;
    *)
      CSV_PARA_PIPELINE="$DATA_FILE"
      ;;
  esac
fi

# ─────────────────────────────────────────────────────────────
#  7. Rodar o pipeline principal
# ─────────────────────────────────────────────────────────────
# O main_pipeline.py tem csv_path e show_plots hard-coded no bloco
# __main__. Em vez de editar o arquivo, chamamos a função run_pipeline
# diretamente via Python, passando os parâmetros corretos. Assim
# evitamos travar em show_plots=True (que abriria janelas e pararia
# o script num servidor sem tela).
info "Executando o pipeline principal..."
python - "$CSV_PARA_PIPELINE" "$GENS" "$POP" <<'PYEOF'
import sys
from main_pipeline import run_pipeline

csv_arg = sys.argv[1] if sys.argv[1] else None
gens    = int(sys.argv[2])
pop     = int(sys.argv[3])

run_pipeline(
    csv_path        = csv_arg,
    n_generations   = gens,
    population_size = pop,
    output_dir      = "./output",
    show_plots      = False,   # nunca abrir janelas em modo script
)
print("\n>>> Pipeline finalizado. Resultados em ./output")
PYEOF
ok "Pipeline principal concluído."

# ─────────────────────────────────────────────────────────────
#  8. Rodar o relatório de métricas
# ─────────────────────────────────────────────────────────────
# O gerar_relatorio.py aceita o caminho do CSV de resultados.
RESULTS_CSV="output/pa_linearization_results.csv"
if [[ -f "$RESULTS_CSV" ]]; then
  info "Gerando relatório de métricas a partir de $RESULTS_CSV ..."
  python gerar_relatorio.py "$RESULTS_CSV"
  ok "Relatório gerado em ./relatorio (relatorio_metricas.txt + painel_metricas.png)"
else
  aviso "CSV de resultados não encontrado em $RESULTS_CSV; pulando o relatório."
fi

# ─────────────────────────────────────────────────────────────
#  9. Resumo final
# ─────────────────────────────────────────────────────────────
echo ""
ok "TUDO PRONTO."
echo -e "${VERDE}Saídas geradas:${SEM}"
echo "  • ./output/      → CSV de resultados + PNGs do pipeline"
echo "  • ./relatorio/   → relatorio_metricas.txt + painel_metricas.png"
echo ""
info "Para desativar o ambiente virtual: digite 'deactivate'"
