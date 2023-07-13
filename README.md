# House Price Improvement Recommender
MSCI 436 Decision Support Systems Term Project

Group 17: Annie Yuan, Edward Jeong, Nishesh Jagga, Tian Xing Zhang

## Description

This tool will suggest renovations to a house based on the details a user enters.

### Files

- **model.py** contains all the functions that deal with data processing and the Linear Regression model from SciKit-Learn.
- **app.py** contains is run by Streamlit and describes the UI layout and the user interaction.
- **data/features.yaml** contains information about the features that are used for prediction. This is used throughout the app via the `FeatureInfo` class.
- **data/train.csv** contains the training data of the Ames Housing Dataset retreived from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
- **requirements.txt** lists all the Python packages that are needed for this app   to run.
- **Dockerfile** contains the instructions to build a Docker image.

## Running the program

### Local

1. Install Python 3.11 (might work with 3.9 or newer)
2. Install requirements via the Terminal: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`
4. Access the app at the Network URL shown

### Docker

1. Install [Docker](https://www.docker.com/)
2. Pull the image: `docker pull ghcr.io/invisco/msci436-group17-dss-project:main`
3. Run the container: `docker run --rm -p 8080:8080 msci436-group17-dss-project:main`
4. Access the app at [localhost:8080](http://localhost:8080/)

## Development

1. Follow [this guide](https://docs.streamlit.io/library/get-started/installation#install-pip) to setup Streamlit.
2. Run `pip install -r requirements.txt` instead of the command in those instructions.
3. Start working.