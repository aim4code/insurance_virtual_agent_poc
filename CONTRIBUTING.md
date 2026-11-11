# Contributing

## Setup
```bash
uv sync              # creates .venv and installs exact versions from uv.lock
uv run pytest        # run the tests
```

## Workflow (GitHub flow)
1. Branch from an up-to-date `main`: `git switch main && git pull && git switch -c feat/short-name`
2. Commit using [Conventional Commits](https://www.conventionalcommits.org): `type(scope): description`
   - types: `feat`, `fix`, `docs`, `test`, `refactor`, `build`, `ci`, `chore`
   - scopes: `crm`, `kb`, `agent`, `evals`
3. Push and open a pull request. The PR title follows the same format, because it becomes the squash commit on `main`.
4. Squash-merge once checks pass. The branch is deleted automatically.

## Rules
- All data is synthetic. Never commit real customer, personal or financial data.
- Changes to prompts, tools, retrieval or guardrails come with an eval run and its delta in the PR description.
