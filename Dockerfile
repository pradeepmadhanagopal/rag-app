FROM python:3.12-slim

WORKDIR /app

# Dependencies first — this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code — changes often, so it comes after the expensive layer
COPY . .

EXPOSE 8000

CMD ["fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]