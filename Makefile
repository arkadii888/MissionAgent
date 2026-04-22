# MissionAgent — local llama.cpp build and server helpers
#
# Uses scripts/build-llama.sh and agent/scripts/run-llama-server.sh

ROOT := $(abspath .)

.DEFAULT_GOAL := help

.PHONY: help build-llama run-llama-server clean-llama

help:
	@echo "Targets:"
	@echo "  build-llama      Configure and build llama.cpp (GGML_METAL=ON) via scripts/build-llama.sh"
	@echo "  run-llama-server  Start llama-server (reads agent/.env.orchestrator if present)"
	@echo "  clean-llama      Remove agent/llama.cpp/build"
	@echo ""
	@echo "Default model paths are set in agent/scripts/run-llama-server.sh; override with MODEL_PATH, PROJ_PATH, PORT, etc."

build-llama:
	bash "$(ROOT)/scripts/build-llama.sh"

run-llama-server:
	bash "$(ROOT)/agent/scripts/run-llama-server.sh"

clean-llama:
	rm -rf "$(ROOT)/agent/llama.cpp/build"
