# EEG-Oriented Self-Supervised Learning and Cluster-Aware Adaptation
<p align="center"><img width="100%" src="files/framework.png" /></p>

This repository provides a tensorflow implementation of a submitted paper:
> **EEG-Oriented Self-Supervised Learning and Cluster-Aware Adaptation**<br>
> Anonymous Authors, Anonymous Institutions<br>
> **Abstract:** *Recently, deep learning-based electroencephalogram (EEG) analysis and decoding have gained widespread attention to monitor a user's clinical condition or identify his/her intention/emotion. Nevertheless, the existing methods mostly model EEG signals with limited viewpoints or restricted concerns about the characteristics of the EEG signals, thus suffering from representing complex spatio-spectro-temporal patterns as well as inter-subject variability. In this work, we propose novel EEG-oriented self-supervised learning methods to discover complex and diverse patterns of spatio-spectral characteristics and spatio-temporal dynamics. Combined with the proposed self-supervised representation learning, we also devise a feature normalization strategy to resolve an inter-subject variability problem via clustering. We demonstrated the validity of the proposed framework on three publicly available datasets by comparing with state-of-the-art methods. It is noteworthy that the same network architecture was applied to three different tasks and outperformed the competing methods, thus relieving the challenge of task-dependent network architecture engineering.*

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
`network.py` contains the proposed deep learning architectures, `utils.py` contains functions used for experimental procedures, and `main.py` contains the main experimental functions.

## Acknowledgements
Anonymous institutes, Anonymous fundings
