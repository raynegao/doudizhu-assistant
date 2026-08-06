PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: test coverage lint typecheck quality demo benchmark docker-demo web-demo demo-gif holdout-seal holdout-evaluate live-calibrate live-assistant live-finalize live-annotate-prepare live-annotate-seal live-diagnostics phase6-acceptance

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest -q --cov=src --cov=scripts --cov-report=term-missing --cov-fail-under=65

lint:
	$(PYTHON) -m ruff check src scripts tests

typecheck:
	$(PYTHON) -m mypy src/state src/logic src/config/live_layout.py src/capture/recording_integrity.py src/reporting
	$(PYTHON) -m mypy --follow-imports=skip src/capture/macos_stream.py src/capture/recorded_window.py

quality: lint typecheck coverage

demo:
	$(PYTHON) -m scripts.run_phase5_showcase --output-dir runs/showcase --repeats 1 --simulations 8 --max-depth 12

benchmark:
	$(PYTHON) -m scripts.run_phase5_showcase --output-dir runs/showcase-benchmark --repeats 3 --simulations 20 --max-depth 20

docker-demo:
	docker build -t doudizhu-assistant:phase5 .
	docker run --rm doudizhu-assistant:phase5

web-demo:
	$(PYTHON) -m src.ui.web

demo-gif:
	$(PYTHON) -m scripts.generate_phase5b_demo_gif

holdout-seal:
	$(PYTHON) -m scripts.seal_real_window_holdout

holdout-evaluate:
	$(PYTHON) -m scripts.evaluate_real_window_holdout \
		--model models/card_cnn.pt \
		--manifest data/real_window_holdout/manifest.jsonl \
		--training-manifest data/cards_cls/manifest.jsonl \
		--output-dir runs/real-window-holdout \
		--require-seal

live-calibrate:
	$(PYTHON) -m scripts.calibrate_live_game \
		--save-config configs/live_game.local.json

live-assistant:
	$(PYTHON) -m scripts.run_live_assistant \
		--config configs/live_game.local.json

live-finalize:
	@test -n "$(SESSION)" || (echo "usage: make live-finalize SESSION=game-001"; exit 2)
	$(PYTHON) -m scripts.finalize_live_recording --session "$(SESSION)"

live-annotate-prepare:
	@test -n "$(SESSION)" || (echo "usage: make live-annotate-prepare SESSION=acceptance-001"; exit 2)
	$(PYTHON) -m scripts.annotate_live_session \
		--session-dir "data/live_game/recordings/$(SESSION)" \
		--prepare

live-annotate-seal:
	@test -n "$(SESSION)" || (echo "usage: make live-annotate-seal SESSION=acceptance-001"; exit 2)
	$(PYTHON) -m scripts.annotate_live_session \
		--session-dir "data/live_game/recordings/$(SESSION)" \
		--seal

live-diagnostics:
	$(PYTHON) -m scripts.analyze_live_log

phase6-acceptance:
	$(PYTHON) -m scripts.audit_phase6_acceptance --require-thresholds
