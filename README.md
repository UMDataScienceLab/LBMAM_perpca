# LBMAM_perpca

This is the implementation of the paper "Personalized Feature Extraction for Manufacturing Process Signature Characterization and Anomaly Detection".

We propose personalized feature extraction (PFE), which separates the features shared across a manufacturing process from the transient features *unique* to each image, and then uses the unique features to detect anomalies.

## Data
- Download data (of one layer) from [this link](https://doi.org/10.18434/M31935)

- Then unzip it.

- Use the matlab file proprocess.m to preprocess the data. Change the name variable on the second line to the .mat to be processed.

- copy every .pkl file into images\ folder.

- Upload everything to a google drive.

## Personalized PCA and t-SNE

Double click the main.ipynb file in google drive, it will automatically launch in google Colab. Follow the instructions there to perform analysis.

## Citation

[Paper](https://doi.org/10.1016/j.jmsy.2024.04.002) (Journal of Manufacturing Systems, 2024). If you use this code, please cite:

```bibtex
@article{shi2024personalized,
  title={Personalized Feature Extraction for Manufacturing Process Signature Characterization and Anomaly Detection},
  author={Shi, Naichen and Guo, Shenghan and Al Kontar, Raed},
  journal={Journal of Manufacturing Systems},
  volume={74},
  pages={435--448},
  year={2024},
  doi={10.1016/j.jmsy.2024.04.002}
}
```
