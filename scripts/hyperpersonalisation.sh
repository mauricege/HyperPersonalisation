#!/usr/bin/env bash

direnv allow . && eval "\$(direnv export bash)"

python -m hyperpersonalisation.trainer config/hyperpersonalisation_all.json