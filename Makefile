SHELL := /bin/bash

.PHONY: setup
setup:
	pyenv local 3.9.8
	python -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install --no-binary=h5py h5py
	.venv/bin/python -m pip install -r requirements_dev.txt


.PHONY: setup_tf28

	pyenv local 3.9.8
	python -m venv .venv_dev_tf2.8p
	.venv_dev_tf2.8p/bin/python -m pip install --upgrade pip
	.venv_dev_tf2.8p/bin/python -m pip install --no-binary=h5py h5py
	.venv_dev_tf2.8p/bin/python -m pip install -r requirements_dev-tf2.8.txt

