# Python 3.11 image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the relevant contents into the container at /app
COPY *.py /app/
COPY requirements.txt /app/
COPY data/*.yaml /app/data/
COPY data/*.csv /app/data/

# Expose port 8080
EXPOSE 8080

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run Streamlit when the container launches
CMD ["streamlit", "run", "app.py", "--server.port", "8080"]
