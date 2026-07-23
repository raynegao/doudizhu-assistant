PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: test demo benchmark docker-demo web-demo demo-gif holdout-evaluate live-calibrate live-assistant

test:
	$(PYTHON) -m pytest -q

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

holdout-evaluate:
	$(PYTHON) -m scripts.evaluate_real_window_holdout \
		--model models/card_cnn.pt \
		--manifest data/real_window_holdout/manifest.jsonl \
		--training-manifest data/cards_cls/manifest.jsonl \
		--output-dir runs/real-window-holdout

live-calibrate:
	$(PYTHON) -m scripts.calibrate_live_game \
		--save-config configs/live_game.local.json

live-assistant:
	$(PYTHON) -m scripts.run_live_assistant \
		--config configs/live_game.local.json
