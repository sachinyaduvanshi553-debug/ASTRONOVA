# ============================================================
# AstroNova Makefile
# ============================================================
.DEFAULT_GOAL := help
SHELL         := /bin/bash
PYTHON        := python3.12
PIP           := pip
SERVICES      := ingestion processing features forecasting xai earth-impact satellite-risk rag copilot notifications gateway

# Colours
BOLD   := $(shell tput bold 2>/dev/null)
RED    := $(shell tput setaf 1 2>/dev/null)
GREEN  := $(shell tput setaf 2 2>/dev/null)
YELLOW := $(shell tput setaf 3 2>/dev/null)
RESET  := $(shell tput sgr0 2>/dev/null)

.PHONY: help install install-dev dev docker-build docker-up docker-down docker-logs \
        test test-unit test-integration test-ml test-coverage test-service \
        lint format typecheck migrate seed-data generate-data \
        train k8s-apply k8s-delete docs clean

# ────────────────────────────────────────────────────────────
# Help
# ────────────────────────────────────────────────────────────
help: ## Show this help message
	@echo "$(BOLD)AstroNova — Available Make Targets$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ────────────────────────────────────────────────────────────
# Installation
# ────────────────────────────────────────────────────────────
install: ## Install production dependencies for all services
	@echo "$(BOLD)Installing shared library...$(RESET)"
	cd shared && $(PIP) install -e .
	@for svc in $(SERVICES); do \
	  echo "$(BOLD)Installing $$svc...$(RESET)"; \
	  if [ -f services/$$svc/requirements.txt ]; then \
	    $(PIP) install -r services/$$svc/requirements.txt; \
	  fi; \
	done
	@echo "$(GREEN)Installation complete.$(RESET)"

install-dev: install ## Install dev + test dependencies
	$(PIP) install \
	  pytest pytest-asyncio pytest-cov pytest-mock httpx \
	  ruff black mypy isort pre-commit \
	  locust hypothesis faker
	pre-commit install
	@echo "$(GREEN)Dev installation complete.$(RESET)"

install-ml: ## Install ML-specific dependencies
	$(PIP) install \
	  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
	  xgboost lightgbm scikit-learn optuna shap \
	  mlflow langchain langchain-community chromadb \
	  transformers accelerate
	@echo "$(GREEN)ML installation complete.$(RESET)"

# ────────────────────────────────────────────────────────────
# Development
# ────────────────────────────────────────────────────────────
dev: ## Start all services in development mode (hot-reload)
	@echo "$(BOLD)Starting all services in dev mode...$(RESET)"
	@for svc in $(SERVICES); do \
	  echo "Starting $$svc..."; \
	  cd services/$$svc && \
	  uvicorn app.main:app --host 0.0.0.0 --port $$(cat .port 2>/dev/null || echo 8000) \
	    --reload --log-level debug & \
	  cd ../..; \
	done
	@echo "$(GREEN)All services started. Use 'make dev-stop' to stop.$(RESET)"

dev-ingestion: ## Start only ingestion service in dev mode
	cd services/ingestion && \
	  uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload --log-level debug

dev-processing: ## Start only processing service in dev mode
	cd services/processing && \
	  uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload --log-level debug

dev-features: ## Start only feature service in dev mode
	cd services/features && \
	  uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload --log-level debug

dev-forecasting: ## Start only forecasting service in dev mode
	cd services/forecasting && \
	  uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload --log-level debug

dev-gateway: ## Start only API gateway in dev mode
	cd services/gateway && \
	  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

dev-stop: ## Stop all background development services
	@pkill -f "uvicorn app.main:app" || true
	@echo "$(GREEN)All dev services stopped.$(RESET)"

# ────────────────────────────────────────────────────────────
# Docker
# ────────────────────────────────────────────────────────────
docker-build: ## Build all Docker images
	@echo "$(BOLD)Building Docker images...$(RESET)"
	docker compose -f docker/docker-compose.yml build --parallel
	@echo "$(GREEN)Build complete.$(RESET)"

docker-build-service: ## Build a single service image (SERVICE=<name>)
	@test -n "$(SERVICE)" || (echo "$(RED)ERROR: Set SERVICE=<name>$(RESET)" && exit 1)
	docker compose -f docker/docker-compose.yml build $(SERVICE)

docker-up: ## Start all infrastructure and services
	@echo "$(BOLD)Starting AstroNova stack...$(RESET)"
	docker compose -f docker/docker-compose.yml up -d
	@echo "$(GREEN)Stack is up. Waiting for health checks...$(RESET)"
	@sleep 10
	@docker compose -f docker/docker-compose.yml ps

docker-up-infra: ## Start only infrastructure services (DB, Kafka, Redis, etc.)
	@echo "$(BOLD)Starting infrastructure services...$(RESET)"
	docker compose -f docker/docker-compose.yml up -d \
	  postgres redis kafka zookeeper kafka-ui mlflow chromadb ollama prometheus grafana
	@echo "$(GREEN)Infrastructure up.$(RESET)"

docker-down: ## Stop all services and remove containers
	@echo "$(BOLD)Stopping AstroNova stack...$(RESET)"
	docker compose -f docker/docker-compose.yml down
	@echo "$(GREEN)Stack stopped.$(RESET)"

docker-down-volumes: ## Stop all services AND remove volumes (WARNING: deletes data)
	@echo "$(RED)$(BOLD)WARNING: This will delete all data volumes!$(RESET)"
	@read -p "Are you sure? [y/N] " ans && [ "$$ans" = "y" ]
	docker compose -f docker/docker-compose.yml down -v
	@echo "$(GREEN)Stack and volumes removed.$(RESET)"

docker-logs: ## Tail logs from all services
	docker compose -f docker/docker-compose.yml logs -f

docker-logs-service: ## Tail logs from a specific service (SERVICE=<name>)
	@test -n "$(SERVICE)" || (echo "$(RED)ERROR: Set SERVICE=<name>$(RESET)" && exit 1)
	docker compose -f docker/docker-compose.yml logs -f $(SERVICE)

docker-restart: ## Restart all services
	docker compose -f docker/docker-compose.yml restart

docker-ps: ## Show status of all containers
	docker compose -f docker/docker-compose.yml ps

docker-pull: ## Pull latest base images
	docker compose -f docker/docker-compose.yml pull

# ────────────────────────────────────────────────────────────
# Testing
# ────────────────────────────────────────────────────────────
test: ## Run all tests
	@echo "$(BOLD)Running all tests...$(RESET)"
	pytest services/ ml/ shared/ -v --tb=short
	@echo "$(GREEN)All tests complete.$(RESET)"

test-unit: ## Run unit tests only
	pytest services/ shared/ -v --tb=short -m "not integration and not slow"

test-integration: ## Run integration tests (requires running infra)
	pytest tests/integration/ -v --tb=short -m "integration"

test-ml: ## Run ML model tests
	pytest ml/tests/ -v --tb=short

test-coverage: ## Run tests with coverage report
	pytest services/ ml/ shared/ \
	  --cov=. \
	  --cov-report=html:htmlcov \
	  --cov-report=term-missing \
	  --cov-fail-under=70 \
	  -v
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(RESET)"

test-service: ## Run tests for a specific service (SERVICE=<name>)
	@test -n "$(SERVICE)" || (echo "$(RED)ERROR: Set SERVICE=<name>$(RESET)" && exit 1)
	pytest services/$(SERVICE)/tests/ -v --tb=short

test-load: ## Run load tests with Locust
	locust -f tests/load/locustfile.py \
	  --host=http://localhost:8000 \
	  --users 50 --spawn-rate 5 --run-time 60s --headless

test-watch: ## Run tests in watch mode (re-run on file changes)
	ptw services/ shared/ -- -v --tb=short

# ────────────────────────────────────────────────────────────
# Code Quality
# ────────────────────────────────────────────────────────────
lint: ## Run ruff linter on all source files
	@echo "$(BOLD)Running ruff linter...$(RESET)"
	ruff check services/ ml/ shared/ --fix
	@echo "$(GREEN)Linting complete.$(RESET)"

format: ## Run black formatter on all source files
	@echo "$(BOLD)Running black formatter...$(RESET)"
	black services/ ml/ shared/ --line-length 100
	@echo "$(GREEN)Formatting complete.$(RESET)"

typecheck: ## Run mypy type checker
	@echo "$(BOLD)Running mypy...$(RESET)"
	mypy services/ shared/ --ignore-missing-imports
	@echo "$(GREEN)Type check complete.$(RESET)"

check: lint typecheck ## Run lint + typecheck (no format changes)

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

isort: ## Sort imports with isort
	isort services/ ml/ shared/ --profile black --line-length 100

# ────────────────────────────────────────────────────────────
# Database
# ────────────────────────────────────────────────────────────
migrate: ## Run Alembic database migrations
	@echo "$(BOLD)Running database migrations...$(RESET)"
	cd shared && alembic upgrade head
	@echo "$(GREEN)Migrations applied.$(RESET)"

migrate-create: ## Create a new migration (MSG="description")
	@test -n "$(MSG)" || (echo "$(RED)ERROR: Set MSG='description'$(RESET)" && exit 1)
	cd shared && alembic revision --autogenerate -m "$(MSG)"

migrate-rollback: ## Rollback last migration
	cd shared && alembic downgrade -1

migrate-history: ## Show migration history
	cd shared && alembic history --verbose

migrate-current: ## Show current migration revision
	cd shared && alembic current

db-shell: ## Connect to PostgreSQL shell
	docker compose -f docker/docker-compose.yml exec postgres \
	  psql -U astronova -d astronova

db-backup: ## Backup database to file
	@mkdir -p backups
	docker compose -f docker/docker-compose.yml exec postgres \
	  pg_dump -U astronova astronova > backups/astronova-$(shell date +%Y%m%d-%H%M%S).sql
	@echo "$(GREEN)Database backed up.$(RESET)"

db-restore: ## Restore database from backup (FILE=<path>)
	@test -n "$(FILE)" || (echo "$(RED)ERROR: Set FILE=<path>$(RESET)" && exit 1)
	docker compose -f docker/docker-compose.yml exec -T postgres \
	  psql -U astronova astronova < $(FILE)
	@echo "$(GREEN)Database restored.$(RESET)"

# ────────────────────────────────────────────────────────────
# Data Generation
# ────────────────────────────────────────────────────────────
generate-data: ## Generate synthetic SOLEXS telemetry data for testing
	@echo "$(BOLD)Generating synthetic solar data...$(RESET)"
	$(PYTHON) ml/data/generators.py \
	  --output data/synthetic \
	  --days 30 \
	  --flare-rate 0.1 \
	  --include-noise
	@echo "$(GREEN)Synthetic data generated in data/synthetic/$(RESET)"

seed-data: ## Seed database with sample events and users
	@echo "$(BOLD)Seeding database...$(RESET)"
	$(PYTHON) scripts/seed_database.py
	@echo "$(GREEN)Database seeded.$(RESET)"

ingest-sample: ## Ingest sample FITS files from data/samples/
	@echo "$(BOLD)Ingesting sample FITS files...$(RESET)"
	$(PYTHON) scripts/ingest_sample_data.py --dir data/samples/
	@echo "$(GREEN)Sample data ingested.$(RESET)"

# ────────────────────────────────────────────────────────────
# ML Training
# ────────────────────────────────────────────────────────────
train: ## Train a model (MODEL=<name>, e.g. lstm_forecaster)
	@test -n "$(MODEL)" || (echo "$(RED)ERROR: Set MODEL=<name>$(RESET)" && exit 1)
	@echo "$(BOLD)Training model: $(MODEL)$(RESET)"
	$(PYTHON) ml/training/trainer.py \
	  --model $(MODEL) \
	  --experiment astronova-forecasting \
	  --data-dir data/processed
	@echo "$(GREEN)Training complete. View at http://localhost:5000$(RESET)"

train-all: ## Train all models sequentially
	@for model in lstm_forecaster cnn_detector xgboost_classifier transformer_model; do \
	  echo "$(BOLD)Training $$model...$(RESET)"; \
	  $(PYTHON) ml/training/trainer.py --model $$model --experiment astronova-forecasting; \
	done
	@echo "$(GREEN)All models trained.$(RESET)"

hyperopt: ## Run hyperparameter optimization (MODEL=<name>)
	@test -n "$(MODEL)" || (echo "$(RED)ERROR: Set MODEL=<name>$(RESET)" && exit 1)
	$(PYTHON) ml/training/hyperopt.py --model $(MODEL) --n-trials 50

evaluate: ## Evaluate models on test data
	$(PYTHON) ml/evaluation/evaluate.py --data-dir data/test/

promote-model: ## Promote a model to production (MODEL=<name> VERSION=<v>)
	@test -n "$(MODEL)" && test -n "$(VERSION)" || \
	  (echo "$(RED)ERROR: Set MODEL= and VERSION=$(RESET)" && exit 1)
	$(PYTHON) scripts/promote_model.py --model $(MODEL) --version $(VERSION)

# ────────────────────────────────────────────────────────────
# Kubernetes
# ────────────────────────────────────────────────────────────
k8s-apply: ## Apply all Kubernetes manifests
	@echo "$(BOLD)Applying Kubernetes manifests...$(RESET)"
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmaps/
	kubectl apply -f k8s/secrets/
	kubectl apply -f k8s/deployments/
	kubectl apply -f k8s/services/
	kubectl apply -f k8s/ingress/
	kubectl apply -f k8s/hpa/
	@echo "$(GREEN)All manifests applied.$(RESET)"

k8s-delete: ## Delete all Kubernetes resources
	@echo "$(RED)$(BOLD)WARNING: This deletes all K8s resources!$(RESET)"
	@read -p "Are you sure? [y/N] " ans && [ "$$ans" = "y" ]
	kubectl delete -f k8s/ --recursive

k8s-status: ## Show status of all pods and services
	kubectl get pods,services,ingress -n astronova

k8s-logs: ## Tail logs from a K8s pod (SERVICE=<name>)
	@test -n "$(SERVICE)" || (echo "$(RED)ERROR: Set SERVICE=<name>$(RESET)" && exit 1)
	kubectl logs -n astronova -l app=$(SERVICE) -f --tail=100

k8s-scale: ## Scale a deployment (SERVICE=<name> REPLICAS=<n>)
	@test -n "$(SERVICE)" && test -n "$(REPLICAS)" || \
	  (echo "$(RED)ERROR: Set SERVICE= and REPLICAS=$(RESET)" && exit 1)
	kubectl scale deployment $(SERVICE) --replicas=$(REPLICAS) -n astronova

helm-install: ## Install with Helm chart
	helm install astronova ./helm/astronova \
	  --namespace astronova \
	  --create-namespace \
	  --values helm/astronova/values.yaml

helm-upgrade: ## Upgrade existing Helm release
	helm upgrade astronova ./helm/astronova \
	  --namespace astronova \
	  --values helm/astronova/values.yaml

helm-uninstall: ## Uninstall Helm release
	helm uninstall astronova --namespace astronova

# ────────────────────────────────────────────────────────────
# Documentation
# ────────────────────────────────────────────────────────────
docs: ## Build API documentation
	@echo "$(BOLD)Building documentation...$(RESET)"
	mkdocs build --site-dir site/
	@echo "$(GREEN)Docs built in site/$(RESET)"

docs-serve: ## Serve documentation locally
	mkdocs serve --dev-addr 0.0.0.0:8888

docs-deploy: ## Deploy documentation to GitHub Pages
	mkdocs gh-deploy --force

openapi-export: ## Export OpenAPI schema from gateway
	curl -s http://localhost:8000/openapi.json | \
	  python3 -m json.tool > docs/api/openapi.json
	@echo "$(GREEN)OpenAPI schema exported to docs/api/openapi.json$(RESET)"

# ────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────
clean: ## Remove build artifacts, caches, and temp files
	@echo "$(BOLD)Cleaning up...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete.$(RESET)"

health-check: ## Check health of all running services
	@echo "$(BOLD)Checking service health...$(RESET)"
	@for port in 8000 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010; do \
	  status=$$(curl -s -o /dev/null -w "%{http_code}" \
	    http://localhost:$$port/health 2>/dev/null || echo "UNREACHABLE"); \
	  echo "  Port $$port: $$status"; \
	done

kafka-topics: ## List all Kafka topics
	docker compose -f docker/docker-compose.yml exec kafka \
	  kafka-topics.sh --list --bootstrap-server localhost:9092

kafka-create-topics: ## Create required Kafka topics
	@for topic in astronova.raw.solexs astronova.processed astronova.features \
	              astronova.alerts astronova.predictions; do \
	  docker compose -f docker/docker-compose.yml exec kafka \
	    kafka-topics.sh --create \
	    --bootstrap-server localhost:9092 \
	    --replication-factor 1 \
	    --partitions 6 \
	    --topic $$topic \
	    --if-not-exists; \
	done
	@echo "$(GREEN)Kafka topics created.$(RESET)"

redis-shell: ## Connect to Redis CLI
	docker compose -f docker/docker-compose.yml exec redis redis-cli

ollama-pull: ## Pull the configured Ollama model
	@echo "$(BOLD)Pulling Ollama model: llama3.2:3b...$(RESET)"
	docker compose -f docker/docker-compose.yml exec ollama \
	  ollama pull llama3.2:3b
	docker compose -f docker/docker-compose.yml exec ollama \
	  ollama pull nomic-embed-text
	@echo "$(GREEN)Ollama models ready.$(RESET)"

env-check: ## Validate that all required env variables are set
	@$(PYTHON) scripts/check_env.py
	@echo "$(GREEN)Environment validation complete.$(RESET)"

version: ## Show versions of key dependencies
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Docker: $$(docker --version)"
	@echo "kubectl: $$(kubectl version --client --short 2>/dev/null || echo 'not installed')"
	@echo "Helm: $$(helm version --short 2>/dev/null || echo 'not installed')"
