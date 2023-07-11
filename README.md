# House Price Improvement Recommender
## MSCI 436 | Group 17

## Running the program

### Local

1. Install Python 3.11 (might work with 3.9 or newer)
2. Install requirements via the Terminal `pip install -r requirements.txt`
3. Run the Streamlit app `streamlit run app.py`
4. Access the app at the Network URL shown

## Docker

1. Install [Docker](https://www.docker.com/)
2. Pull the image `docker pull ghcr.io/invisco/msci436-group17-dss-project:main`
3. Run the container `docker run --rm -p 8080:8080 msci436-group17-dss-project:main`
4. Access the app at [localhost:8080](http://localhost:8080/)

## Development

1. Follow [this guide](https://docs.streamlit.io/library/get-started/installation#install-pip) to setup Streamlit.
2. Run `pip install -r requirements.txt` instead of the command in those instructions.
3. Start working.