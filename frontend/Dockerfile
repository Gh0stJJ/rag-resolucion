FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /streamlit_app
COPY requirements.txt /streamlit_app/requirements.txt
RUN pip install --no-cache-dir -r /streamlit_app/requirements.txt
COPY . /streamlit_app/
EXPOSE 8015
CMD ["streamlit", "run", "app.py"]