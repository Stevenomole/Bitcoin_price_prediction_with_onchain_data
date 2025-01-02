# start from a Python image
FROM python:3.9-slim

# set the working directory in the container
WORKDIR /app

# copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy the entire project into the container's working directory
COPY . .

# set the command that runs the main script
CMD ["python", "src/main.py"]
