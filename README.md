# RATING CURVE EXPLORER

Ambiente simples para **carregar**, **visualizar** e **ajustar** curvas‑chave de estações fluviométricas.

## Estrutura

### Organização de pastas
```graphql
RATING_CURVE_EXPLORER/
│
├── data/                         # Dados (não versionados)
│
├── scripts/
│   ├── __init__.py
│   ├── db_load.py                # ETL → SQLite (usa schema.sql)
│   ├── plot_things.py            # exemplos de uso
│   └── hydrodb.py                # classe HydroDB (timeseries, vertical profile, rating curve)
│
├── docs/
│   ├── rating_curve_method.md    # modelo, segmentação, ajustes, boas práticas
│   └── data_layout.md            # convenções de arquivos/pastas em data/
│
├── schema.sql
├── requirements.txt
├── .gitignore
└── README.md                     # guia rápido
```

### Ambiente

- numpy>=1.25
- pandas>=2.0
- matplotlib>=3.7
- scipy>=1.11

### Dados de entrada

CSVs por estação em `data/<station_id>/` (ex.: `data/86720000/_Estacao_.csv`, `_Cotas_.csv`, etc).

## Uso

### ETL
```bash
python scripts/db_load.py
```

### Visualizações

```python
from scripts.viz import HydroDB
db = HydroDB("data/hydrodata.sqlite")

# Séries
db.plot_timeseries(station=86720000, variable="level", start_date="2024-01-01", end_date="2025-01-01", rolling=7)

# Perfil vertical
db.plot_vertical_profile(survey_id=3875)

# Curva‑chave (H no eixo X)
db.plot_rating_curve(station=86720000, level_breaks=[100, 300, 600], fit=True)
```

#### Parâmetros úteis

* `plot_timeseries`: `variable` ∈ {`level`, `discharge`}, `rolling` = janela em dias.
* `plot_rating_curve`:

  * `level_breaks=[b1, b2, ...]` → segmentos `[minH, b1]`, `(b1, b2]`, …, `(b_n, maxH]`
  * Modelo: `Q = a * (H - h0)^b` (ajuste SciPy).


# Documentação mais detalhada

- [Curva‑chave: método](.docs/rating_curve_method.md) — modelo, segmentação, ajustes
- [Layout de dados](.docs/data_layout.md) — convenções para os dados de entrada e organização do banco de dados para análise

## Colaboração (Git)

* **Branch curto por tarefa**; PR com *squash & merge* para `main`.
* Nomes: `feat/<topico>-<iniciais>`, `fix/...`, `docs/...`.
* Antes de começar:

  ```bash
  git pull origin main
  git switch -c feat/my-task-rb
  ```
* Para atualizar seu branch:

  ```bash
  git fetch origin
  git rebase origin/main      # mantém histórico limpo
  # ou: git merge origin/main
  ```

## Próximos passos

- Plots de múltiplas seções transversais.
- Ajustar nodata nas séries temporais.
- Aprimorar métodos de análise de curva chave.
