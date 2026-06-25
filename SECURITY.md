# Security Policy

## Supported Versions

This project is a demo application. Security fixes are applied to the default
branch.

## Reporting a Vulnerability

Please do not open public issues for secrets, authentication bypasses, data
exposure, or other sensitive vulnerabilities.

Report security issues privately to the project maintainers. If no private
contact is configured for your fork, temporarily keep the repository private
until a maintainer contact is available.

## Production Notes

- Do not commit `.env`, `.dev.env`, API keys, database dumps, Redis dumps, or
  Qdrant storage files.
- Rotate any key that was ever committed or shared.
- Restrict CORS origins before exposing the API publicly.
- Put the API behind HTTPS and authentication for production usage.
- Use strong Redis and Postgres credentials.
- Avoid exposing Redis, Postgres, or Qdrant ports to the public internet.
