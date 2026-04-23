FROM python:3.11-slim

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY static/ ./static/
COPY netflix_data.csv .
COPY .env .

RUN set -a && . ./.env && set +a && \
    python -c "from recommender import Recommender; Recommender('netflix_data.csv')"

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
