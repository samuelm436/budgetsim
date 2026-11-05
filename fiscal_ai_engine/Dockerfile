FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/fiscal_ai_engine

EXPOSE 8000

CMD ["uvicorn", "fiscal_ai_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
