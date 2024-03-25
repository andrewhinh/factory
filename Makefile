# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST))

# Install exact Python and CUDA versions
env:
	conda env update --prune -f environment.yml

# Compile and install exact pip packages
install:
	pip install uv==0.1.22
	uv pip compile requirements/requirements.in -o requirements/requirements.txt
	uv pip sync requirements/requirements.txt

# Setup
setup:
	pre-commit install
	export PYTHONPATH=.
	mkcert -install
	mkcert localhost 127.0.0.1 ::1
	mkdir -p certificates
	mv localhost+2-key.pem certificates/localhost+2-key.pem
	mv localhost+2.pem certificates/localhost+2.pem

# Compile the model
compile:
	python app/compile.py

# Run the app
run:
	python app/main.py

# Serve the app
serve:
	ngrok http https://localhost:8000

# Lint and format
fix:
	pre-commit run --all-files

# Bump versions of transitive dependencies
upgrade:
	pip install uv==0.1.22
	uv pip compile --upgrade requirements/requirements.in -o requirements/requirements.txt
	uv pip sync requirements/requirements.txt
