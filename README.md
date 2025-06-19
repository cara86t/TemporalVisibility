[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/cara86t/TemporalVisibility)

# Perceptual Visibility Model for Temporal Contrast Changes in Periphery

## Description

This repository provides the authors' implementation of the following paper

Cara Tursun and Piotr Didyk. "Perceptual Visibility Model for Temporal Contrast Changes in Periphery." ACM Transactions on Graphics 42.2 (2022): 1-16.

Project website: https://visualcomputing.nl/cara/projects/temporal_visibility

## Requirements

The implementation is tested on Ubuntu 20.04 with Python 3.6.10 and on macOS 13.1 with Python 3.8.13. The project depends on following Python libraries:

```
decord>=0.4.0
external>=0.0.1
h5py>=2.10.0
matplotlib>=3.2.2
numpy>=1.19.3
opencv_python>=4.5.1.48
PsychoPy>=2020.2.5
pypng>=0.20220715.0
scipy>=1.5.0
```

The libraries are also provided in "requirements.txt" file. The recommended way of installing the requirements is using a Python package and environment manager (e.g., Conda) and issuing the following command in your Python environment:

```
pip install -r requirements.txt
```

## Usage

Some sample videos are provided in the directory "videos_taa". If the Python environment is set up correctly, executing the Bash script "test_run_vid.sh" should run the predictor for a sample video and save the results under "test_run" directory.

The entry point to the implementation is "process_vid.py" file. This file requires an input file name and an output directory name as arguments. There are other optional arguments. Please run it with "-h" flag to get help:

```
python process_vid.py -h
```

## How to add/change display parameters

The metric uses display parameters such as physical width/height (m), horizontal/vertical pixel count (resolution), refresh rate (Hz), peak luminance (cd/m^2), and viewing distance (m) of the observer to compute the output. These are defined at the beginning of the file "lib/display_tools.py" as a dict (may be used to add new displays). The "Name" tag of the used display is set as the "DISPLAY" variable at the beginning of "process_vid.py".

## Known issues

cv2.cvtColor is very slow on macOS with Python 3.6.10 and visualization of the outputs take a very long time. If you experience this, you may try running with Python 3.8.

## How to cite using BibTeX

```
@article{Cara2023,
  author = {Tursun, Cara and Didyk, Piotr},
  title = {Perceptual Visibility Model for Temporal Contrast Changes in Periphery},
  year = {2022},
  issue_date = {April 2023},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {42},
  number = {2},
  issn = {0730-0301},
  url = {https://doi.org/10.1145/3564241},
  doi = {10.1145/3564241},
  journal = {ACM Trans. Graph.},
  month = {nov},
  articleno = {20},
  numpages = {16},
  keywords = {Temporal visibility metric, temporal change perception, spatio-temporal sensitivity, imperceptible visual change}
}
```

## Authors

Piotr is an associate professor in Università della Svizzera italiana, Switzerland. His personal webpage:
https://www.pdf.inf.usi.ch/people/piotr

Cara's personal webpage:
https://visualcomputing.nl/cara

