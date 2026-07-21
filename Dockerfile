FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt

COPY src ./src
COPY scripts ./scripts
COPY examples ./examples
COPY docs/evidence ./docs/evidence

CMD ["python", "-m", "scripts.run_phase5_showcase", "--output-dir", "/tmp/showcase", "--repeats", "1", "--simulations", "8", "--max-depth", "12"]
