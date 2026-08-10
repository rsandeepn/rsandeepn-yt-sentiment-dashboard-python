# Docker Compose setup

This stack runs the React frontend, FastAPI backend, and PostgreSQL database.
PostgreSQL data is stored in the named `postgres_data` volume and survives
container recreation.

## Prerequisites

Install Docker Desktop and ensure `docker compose version` works. The default
frontend build path matches the current Mac folders:

```text
Documents/yt-multilingual-sentiment-agent
Documents/yt-multilingual-react/yt-ui
```

If the frontend is elsewhere, add `FRONTEND_CONTEXT` with its absolute path to
`.env.docker` before running Compose.

## Configure and start

From the backend repository:

```bash
cp .env.docker.example .env.docker
openssl rand -hex 32
openssl rand -hex 24
```

Put the first value in `JWT_SECRET_KEY`. Put the second value in both
`POSTGRES_PASSWORD` and the password portion of `DATABASE_URL`. Also set the
restricted YouTube API key.

### Google sign-in

Create a Google OAuth 2.0 **Web application** client and add the site URL as an
authorized JavaScript origin. For local Docker testing, add:

```text
http://localhost:8080
```

Add the public HTTPS domain later as a second origin. Put the client ID (not a
client secret) in `.env.docker`:

```env
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
```

Docker passes the same client ID to the frontend at build time and to the
backend at runtime. Rebuild both services after changing it:

```bash
docker compose --env-file .env.docker up --build -d backend frontend
```

Validate and start the stack:

```bash
docker compose --env-file .env.docker config
docker compose --env-file .env.docker up --build -d
docker compose --env-file .env.docker ps
```

Open <http://localhost:8080>. The backend health endpoint is available locally
at <http://127.0.0.1:8000/health>.

The first backend build installs the analysis dependencies and can take several
minutes. Later builds reuse Docker's cached layers unless dependencies change.

## Logs and lifecycle

```bash
docker compose --env-file .env.docker logs -f backend
docker compose --env-file .env.docker logs -f frontend
docker compose --env-file .env.docker stop
docker compose --env-file .env.docker start
docker compose --env-file .env.docker down
```

`docker compose down` keeps the database volume. Do not add `--volumes` unless
you intentionally want to delete all application database data.

## Database backup and restore

Create a backup directory outside the repositories, then run:

```bash
mkdir -p "$HOME/youtube-analyzer-backups"
docker compose --env-file .env.docker exec -T db \
  pg_dump -U youtube_app -d youtube_analyzer -Fc \
  > "$HOME/youtube-analyzer-backups/youtube_analyzer.dump"
```

Verify that the dump is non-empty before relying on it:

```bash
ls -lh "$HOME/youtube-analyzer-backups/youtube_analyzer.dump"
```

Restoring overwrites database objects. Stop the backend first and restore only
from a trusted backup:

```bash
docker compose --env-file .env.docker stop backend
docker compose --env-file .env.docker exec -T db \
  pg_restore -U youtube_app -d youtube_analyzer --clean --if-exists \
  < "$HOME/youtube-analyzer-backups/youtube_analyzer.dump"
docker compose --env-file .env.docker start backend
```

## Rebuild after code changes

```bash
docker compose --env-file .env.docker up --build -d
```

Public access, the purchased domain, Cloudflare Tunnel, and HTTPS are handled
in the next deployment milestone. Do not expose PostgreSQL port 5432 publicly.
