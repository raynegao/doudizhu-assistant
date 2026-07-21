PYTHON ?= python

.PHONY: test demo benchmark docker-demo web-demo demo-gif

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
