# GitHub Repository Summarizer

A CLI that summarizes a GitHub repo — from a local folder or an `owner/repo` slug — and writes the
result to JSON, with an optional rich terminal view. Point it at a local clone and it reads the
GitHub remote from `.git` automatically.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env     # add your token, or: export GITHUB_TOKEN=ghp_xxx
```

The token is read from `--token`, then `$GITHUB_TOKEN`, then `.env`. It raises the API rate limit
and lets you read private repos you can access.

## Usage

```bash
python main.py /path/to/repo               # local folder
python main.py owner/repo                  # remote, no clone needed
python main.py /path/to/folder-of-repos    # summarize every repo in a folder
python main.py path/to/repo --ui           # terminal view + JSON
python main.py path/to/repo --output report.json
```

It pulls metadata (stars, forks, topics, license, timestamps), the language breakdown,
community-health signals, structure, and recent activity.

## Notes

- `.env` and report JSON are git-ignored. Rotate any token you've exposed.
