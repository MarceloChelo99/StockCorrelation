# StockCorrelation

Research pipeline for learning stock/company embeddings from SEC filings,
events, fundamentals, graph relationships, and market behavior.

## Pipeline Scripts

The canonical script layout follows the project flow:

```text
scripts/pipeline/stage_01_ingest/
scripts/pipeline/stage_02_features/
scripts/pipeline/stage_03_assembly/
scripts/pipeline/stage_04_model/
scripts/pipeline/stage_05_evaluation/
```

Common commands still have compatibility wrappers at the old root paths, so both
styles work while the repo settles:

```bash
.venv/bin/python scripts/02_compute_features.py historical_text_business
.venv/bin/python scripts/03_assemble_dataset.py historical_text_business
.venv/bin/python scripts/04_train.py historical_text_business
.venv/bin/python scripts/05_evaluate.py historical_text_business experiments/<run>/embeddings.parquet
```

The new direct module-style equivalents are:

```bash
.venv/bin/python -m scripts.pipeline.stage_02_features.compute_features historical_text_business
.venv/bin/python -m scripts.pipeline.stage_03_assembly.assemble_dataset historical_text_business
.venv/bin/python -m scripts.pipeline.stage_04_model.train historical_text_business
.venv/bin/python -m scripts.pipeline.stage_05_evaluation.evaluate historical_text_business experiments/<run>/embeddings.parquet
```

Historical filing text streaming now lives at:

```bash
.venv/bin/python -m scripts.historical_text.stream_features --since 2010-01-01 --forms "10-K,10-Q" --output-dir data/processed/historical_text_10k_10q
```
