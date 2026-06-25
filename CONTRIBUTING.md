# Contributing

Thanks for helping improve this project.

## Development Setup

1. Install Python 3.12 and Poetry.
2. Copy `.env.example` to `.env` and fill in your local keys.
3. Install dependencies:

```bash
poetry install
```

4. Run tests:

```bash
python -m unittest
```

## Pull Requests

- Keep changes focused.
- Do not commit secrets, local vector database storage, SQLite runtime files, or
  generated caches.
- Include tests for behavior changes when practical.
- Update `README.md` when commands, environment variables, or API behavior
  changes.
