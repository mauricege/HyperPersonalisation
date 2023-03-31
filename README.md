# HyperPersonalisation

This repository contains code for zero-shot, meta-data conditioned personalisation of wav2vec speech models. Personalisation is achieved through an adaptation of [hyperformers](https://github.com/rabeehk/hyperformer) - relevant reused code is assembled in `./hyperformer`.



### Install dependencies
You can use either poetry (`poetry install`) of pip (`pip install -r requirements.txt`).

## Usage
The training code is configured via .json files (you can find templates for the wav2vec baseline and personalisation in `./configs`). You have to adapt the paths to your data directory (`data_base`), your label directory (`label_base` - containing `train.csv`, `dev.csv` and `test.csv`) and the csv file containing metadata for every subject (`metadata_file`).

### Structure of label and metadata csv files
The label csvs should have this header:
```csv
filename,[subject_column],[target]
```
where `filename` is the relative path to each audio file from `data_base`. `[subject_column]` and `[target]` are defined through the config file and should contain the key used for personalisation (e.g., a participant code) and the label for the audio file.

The metadata csv should follow the format:
```csv
[subject_column],[m_1],[m_2],...,[m_N]
```
where `[subject_column]` is defined in the config file and should have matching keys with the subjects defined for the audio files in the labels. `[m_i]` are metadata columns containing numeric values. The config files can be adjusted to define which columns will be utilised for personalisation.


### Fine-tuning wav2vec
Run
```console
python -m hyperpersonalisation.trainer configs/wav2vec.json
```
to fine-tune the wav2vec encoder.

### Personalisation
After fine-tuning, run
```console
python -m hyperpersonalisation.trainer configs/hyperpersonalisation_all.json
```
to train the personalisation components.


## Contact
Please direct any questions or requests to Maurice Gercuk (maurice.gerczuk at informatik.uni-augsburg.de).