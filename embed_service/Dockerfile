FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requerimientos e instálalos
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copia el código de la aplicación
# COPY embed_service.py .

# Expone el puerto y define el comando de inicio
EXPOSE 8010
CMD ["uvicorn", "embed_service:app", "--host", "0.0.0.0", "--port", "8010"]