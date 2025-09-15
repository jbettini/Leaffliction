#!/bin/bash

set -eu

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

