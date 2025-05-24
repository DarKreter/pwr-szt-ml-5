IMAGE_NAME = ml4
NETWORK_NAME = $(IMAGE_NAME)-network

# Do it once
create_network:
	docker network create $(NETWORK_NAME)

build:
	docker build --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -t $(IMAGE_NAME) .

format:
	docker run --rm -v $$(pwd):/app $(IMAGE_NAME) ruff format .

run_docker:
	docker run --rm -v $$(pwd):/app --network $(NETWORK_NAME) -it $(IMAGE_NAME) bash

run_mlflow_server:
	docker run -d --name mlflow-server --rm -v $$(pwd):/app --network $(NETWORK_NAME) -p 5000:5000 -it $(IMAGE_NAME) mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0
