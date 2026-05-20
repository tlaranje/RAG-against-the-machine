# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tlaranje <tlaranje@src.42porto.com>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/02/26 12:06:51 by tlaranje          #+#    #+#              #
#    Updated: 2026/04/29 10:42:53 by tlaranje         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

RM					:= rm -rf
FIND				:= find

MODEL_DIR  := $(HF_HOME)/
MODEL_URL  := https://huggingface.co/enacimie/Qwen3-0.6B-Q4_K_M-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf
MODEL_NAME := Qwen

# === DIRENV SETUP ===
DIRENV_BIN := $(HOME)/.local/bin/direnv

define SETUP_DIRENV
if ! command -v direnv >/dev/null 2>&1; then \
    echo "Installing direnv..."; \
    mkdir -p $(MODEL_DIR); \
    curl -sfL https://direnv.net/install.sh | bash; \
    export PATH="$(HOME)/.local/bin:$$PATH"; \
fi; \
direnv allow >/dev/null 2>&1 || true
endef

CLEAR := $(SETUP_DIRENV) && clear

# === BUILD TARGETS ===
install:
	@$(CLEAR)
	@echo "Checking LLM directory..."
	@mkdir -p $(MODEL_DIR)
	@if [ ! -f "$(MODEL_DIR)/$(MODEL_NAME)" ]; then \
		echo "Downloading $(MODEL_NAME)..."; \
		curl -L $(MODEL_URL) -o $(MODEL_DIR)/$(MODEL_NAME); \
	fi
	@uv venv --python 3.12
	@uv sync

run:
	@$(CLEAR) && uv run python -m student

debug:
	@$(CLEAR) && uv run python -m pdb -m student

clean:
	@clear
	@echo "Cleaning project cache..."
	@$(RM) -r data/raw/vllm-0.10.1
	@$(RM) -r .direnv
	@$(FIND) . -type d -name "__pycache__" -exec $(RM) {} +
	@$(FIND) . -type d -name ".mypy_cache" -exec $(RM) {} +
	@$(FIND) . -type d -name ".pytest_cache" -exec $(RM) {} +
	@$(FIND) . -type f -name "*.pyc" -delete
	@$(FIND) . -type f -name "*.pyo" -delete

fclean: clean
	@echo "Removing environment and caches..."
	@if [ -d "$(UV_PROJECT_ENVIRONMENT)" ]; then \
		echo "Deleting $(UV_PROJECT_ENVIRONMENT)..."; \
		$(RM) "$(UV_PROJECT_ENVIRONMENT)"; \
	fi
	@uv cache clean --force
	@if [ -d "$(HF_HOME)" ]; then \
		echo "Deleting HF_HOME..."; \
		$(RM) "$(HF_HOME)"; \
	fi

lint:
	@clear && uv run flake8 student
	@uv run mypy student --warn-return-any \
	    --warn-unused-ignores \
	    --ignore-missing-imports \
	    --disallow-untyped-defs \
	    --check-untyped-defs

lint-strict:
	@clear && uv run flake8 student
	@uv run mypy student --strict

.PHONY: install run debug clean lint lint-strict