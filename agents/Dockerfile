FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -U -r requirements.txt

EXPOSE 8020

CMD [ "uvicorn","main_agent:app","--host","0.0.0.0", "--port", "8020","--reload"]