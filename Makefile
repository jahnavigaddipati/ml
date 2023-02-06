SHELL := $(shell which bash)
CUR_DIR := $(CURDIR)

CONDA_BIN = $(shell which conda)
CONDA_ROOT = $(shell $(CONDA_BIN) info --base)

ENV_NAME := deep-learning-scene-recognition
ENV_FILE := environment.yml

CONDA_ENV_PREFIX = $(shell conda env list | grep $(ENV_NAME) | sort | awk '{$$1=""; print $$0}' | tr -d '*\| ')
CONDA_ACTIVATE := source $(CONDA_ROOT)/etc/profile.d/conda.sh ; conda activate $(ENV_NAME) && PATH=${CONDA_ENV_PREFIX}/bin:${PATH};

setup:
	$(CONDA_BIN) env create -f $(ENV_FILE)

update:
	$(CONDA_BIN) env update -n $(ENV_NAME) -f $(ENV_FILE) --prune