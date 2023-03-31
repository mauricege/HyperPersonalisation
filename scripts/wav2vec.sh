#!/usr/bin/env bash

direnv allow . && eval "\$(direnv export bash)"

python -m hyperpersonalisation.trainer configs/wav2vec.json
