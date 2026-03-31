.PHONY: build up down test logs clean

# Copy env template if .env doesn't exist
env:
	@if [ ! -f .env ]; then cp .env.template .env && echo "Created .env from template — fill in your keys"; fi

# Build all Docker images
build:
	docker compose build

# Start all services
up:
	docker compose up -d

# Stop all services
down:
	docker compose down

# Show logs (usage: make logs s=embedding)
logs:
	docker compose logs -f $(s)

# Run all tests
test:
	docker compose run --rm embedding pytest tests/ -v
	docker compose run --rm rag pytest tests/ -v

# Remove containers, volumes, images
clean:
	docker compose down -v --rmi local

# Shortcut: rebuild and start fresh
restart: down build up
