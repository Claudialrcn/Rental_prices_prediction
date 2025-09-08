FROM python:3.13-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Instalamos herramientas del sistema
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Instalamos uv desde el script oficial
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

# Copiamos archivos de dependencias
COPY pyproject.toml uv.lock ./

# Instalamos dependencias del proyecto
RUN uv sync --frozen --no-dev
RUN uv add dash

# Copiamos el resto del proyecto
COPY . .

# Exponemos puerto de Dash
EXPOSE 8050

ENV APP_ENV=prod

# Ejecutamos directamente con python
CMD ["uv","run", "dash_app/app.py"]
