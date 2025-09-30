#Dockerfile
FROM python:3.11-slim

#actualizar apt, instalar, y luego limpiar en una sola capa.
#necesario para FlagEmbedding ya que necesitoa gcc
#RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requerimientos e instala 
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -U -r requirements.txt
#instalar torch para usar MPS
#RUN pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

#ENV HF_TOKEN=${HF_TOKEN}

# Copia el código de la aplicación
# COPY embed_service.py .

# Expone el puerto y define el comando de inicio
EXPOSE 8010
CMD ["uvicorn", "embed_service:app", "--host", "0.0.0.0", "--port", "8010"]

