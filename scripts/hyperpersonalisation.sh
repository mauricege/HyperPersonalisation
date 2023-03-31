#!/usr/bin/env bash

direnv allow . && eval "\$(direnv export bash)"

python -m hyperpersonalisation.trainer configs/hyperpersonalisation_all.json
