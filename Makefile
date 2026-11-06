.PHONY: backend frontend test

PYTHON ?= python

backend:
	$(PYTHON) -m uvicorn songcraft.main:app --reload --app-dir backend --host 127.0.0.1 --port 8000

frontend:
	cd frontend && npm run dev

test:
	$(PYTHON) -m pytest
	cd frontend && npm test
