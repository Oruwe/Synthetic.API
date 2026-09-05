.PHONY: up down logs test lint demo

up:
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f agents-orchestrator agents-synthesizer

test:
	uv run pytest -q

demo:
	bash scripts/send_sample_transcript.sh
