PYTHON ?= python
MODELS := baserate elo poisson dixon_coles market

.PHONY: all data market backtest backtests report test clean-outputs

# Everything: data -> market benchmark -> every model -> report -> tests
all: data market backtests report test

# M1: download + cache raw CSVs, then build data/clean/matches.parquet
data:
	$(PYTHON) src/ingest.py
	$(PYTHON) src/clean.py

# M2: market benchmark metrics (both de-vig methods) + its calibration figure
market: data
	$(PYTHON) src/evaluate.py market

# M3+: walk-forward backtest of one model (one of $(MODELS)), e.g. make backtest MODEL=elo
MODEL ?= baserate
backtest: data
	$(PYTHON) src/backtest.py --model $(MODEL)

# every model (Elo, Poisson and Dixon-Coles also sweep and plot their tuning grid)
backtests: data
	for m in $(MODELS); do $(PYTHON) src/backtest.py --model $$m || exit 1; done

# M7: calibration curves, model-vs-market tables and figures, isotonic recalibration test
report: backtests
	$(PYTHON) src/evaluate.py models

test:
	$(PYTHON) -m pytest tests -q

# remove generated data, keeping raw downloads; reports/ is committed so is left alone
clean-outputs:
	rm -rf data/clean data/predictions
