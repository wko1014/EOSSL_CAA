# S<sup>4</sup>TRL: A Novel Deep Learning Framework with Self-Supervision for Uncovering Spatio-Spectro-Temporal Patterns of Electroencephalogram Signals
Self-supervised spatio-spectro-temporal represenation learning for EEG analysis

This repository provides a tensorflow implementation of a submitted paper:
> **S<sup>4</sup>TRL: A Novel Deep Learning Framework with Self-Supervision for Uncovering Spatio-Spectro-Temporal Patterns of Electroencephalogram Signals**<br>
> Anonymous Authors, Anonymous Institutions<br>
> **Abstract:** *Recently, deep learning has demonstrated its caliber for representing various data types, vision, natural language, and signal. Among them, deep learning-based electroencephalogram (EEG) analysis has gained widespread attention to monitor a user's clinical condition or identify their intention. Nevertheless, existing methods are often not satisfactory to represent EEG signals because the data analysis is performed in limited viewpoints. In this work, we propose novel self-supervised learning methods to recognize complex patterns of spatio-spectral characteristics and spatio-temporal dynamics for EEG analysis. Moreover, our proposed deep learning architecture as well as feature normalization method suppresses inter-subject variability of the complex patterns. Our proposed framework obtains more empowered representation even with inter-subject calibration samples, thereby shedding light on the practical use of EEG in the real world. We demonstrated the validity of the proposed framework on three publicly available datasets with state-of-the-art comparison methods.*

## Dependencies
* [Python 3.8+](https://www.continuum.io/downloads)
* [TensorFlow 2.7.0+](https://www.tensorflow.org/)

## Datasets
To download Sleep-EDF database
* https://physionet.org/content/sleep-edf/1.0.0/

To download KU-MI database
* http://gigadb.org/dataset/100542/

To download TUH abnormal corpus database
* https://isip.piconepress.com/projects/tuh_eeg/

## Usage
`network.py` contains the proposed deep learning architectures, `utils.py` contains functions used for experimental procedures, and `experiment.py` contains the main experimental functions.

## Acknowledgements
Anonymous
