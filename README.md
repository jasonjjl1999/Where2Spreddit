# Where2Spreddit

A collection of self-trained Neural Networks for classifying Reddit posts and Comments.

## Installation Instructions

### 1. Clone this repository to your local workspace

### 2. Install required Python packages specified in `requirements.txt`

From root directory of repo, run `pip3 install -r requirements.txt`. It is highly recommended to use a clean virtual environment to prevent conflicts with other packages.

### 3. Install packages for spaCy tokenizer

Run

```
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm
```

## Usage

For inference, run `python predictor.py` in the virutal environment.

For training, run `python main.py` in the virtual environment.