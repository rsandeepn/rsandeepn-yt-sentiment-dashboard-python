# YouTube Sentiment Dashboard API

FastAPI backend with YouTube comment analysis, PostgreSQL user accounts, JWT
authentication, and per-user analysis history. There are currently no usage
limits.

## Local setup

Create a PostgreSQL database and user:

```sql
CREATE USER youtube_app WITH PASSWORD 'choose_a_strong_password';
CREATE DATABASE youtube_analyzer OWNER youtube_app;
```

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Set `DATABASE_URL`, your restricted YouTube API
key, and a unique JWT secret. Generate the JWT secret with:

```bash
openssl rand -hex 32
```

Start the API:

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Database tables are created automatically when the API starts. If
`DATABASE_URL` is omitted, development falls back to a local SQLite file;
configure PostgreSQL for the intended server environment.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Docker Compose

For the production-style local stack with the frontend and PostgreSQL, follow
[DOCKER.md](DOCKER.md). The stack keeps database data in a persistent Docker
volume and does not introduce usage limits.
