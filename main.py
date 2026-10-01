#!/usr/bin/env python3
"""
GitHub Repository Summarizer  v2
──────────────────────────────────
Comprehensively summarizes GitHub repos via local paths or owner/repo slugs.
Auto-detects the GitHub remote from local .git folders.

╔══════════════════════════════════════════════════════════════╗
║  MODES                                                       ║
╠══════════════════════════════════════════════════════════════╣
║  Single  ─ local git folder or owner/repo slug               ║
║  Multi   ─ parent folder containing multiple git repos       ║
╚══════════════════════════════════════════════════════════════╝

EXAMPLES
────────
  # Summarize one local repo (auto-detects GitHub remote):
  python main.py /path/to/some-repo

  # Summarize all repos inside a folder (multi mode):
  python main.py /path/to/repos-folder

  # Remote-only (no local clone needed):
  python main.py torvalds/linux

  # Show rich terminal UI alongside saving JSON:
  python main.py path/to/repo --ui

  # Custom output file (single) or output folder (multi):
  python main.py path/to/repo --output /reports/arnio.json
  python main.py path/to/parent --output /reports/

  # Use a different .env file:
  python main.py owner/repo --env-file ~/secrets/.env

WHAT GETS FETCHED
─────────────────
  • Repo metadata (stars, forks, topics, license, visibility …)
  • Languages breakdown
  • Community health score
  • README, CONTRIBUTING, LICENSE file contents (previews)
  • All open issues (WIP/assigned highlighted)
  • Open pull requests
  • Recent releases (last 10)
  • Top contributors (up to 30)
  • Labels & milestones
  • Top forks (newest, up to 20)

OUTPUT DEFAULTS
───────────────
  Single local path  → <repo_path>/github_summary.json
  Remote-only slug   → <owner>_<repo>_github_summary.json  (cwd)
  Multi mode         → <each_repo_path>/github_summary.json

REQUIREMENTS
────────────
  pip install requests rich python-dotenv
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

try:
    from dotenv import load_dotenv as _load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
BASE_URL = "https://api.github.com"
console = Console()

WIP_LABELS = {
    "wip",
    "work in progress",
    "in progress",
    "in-progress",
    "doing",
    "working",
    "active",
    "in_progress",
}

SCHEMA_VERSION = "2.0"

# Placeholder owner names that appear in template READMEs — skip these slugs
_PLACEHOLDER_OWNERS = {
    "your-username",
    "your_username",
    "yourusername",
    "YOUR_USERNAME",
    "YOUR-USERNAME",
    "username",
    "owner",
    "OriginalOwner",
    "originalowner",
    "example",
    "user",
    "myusername",
    "yourname",
    "placeholder",
    "TODO",
    "todo",
}


class RepoError(Exception):
    """Raised instead of sys.exit() so worker threads can catch it gracefully."""


# ═════════════════════════════════════════════════════════════════════════════
# GitHub slug detection — works for cloned repos AND zip-extracted folders
# ═════════════════════════════════════════════════════════════════════════════

# Regex that matches a github.com/owner/repo path anywhere in text
import re as _re

_GH_URL_RE = _re.compile(
    r'github\.com[/:]([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+?)(?:\.git|["\'\s,)#]|$)',
    _re.MULTILINE,
)


def parse_github_slug(remote_url: str) -> Optional[str]:
    """Parse 'owner/repo' from any GitHub remote URL (https or ssh)."""
    url = remote_url.strip()
    # SSH: git@github.com:owner/repo.git
    if url.startswith("git@github.com:"):
        return _clean_slug(url[len("git@github.com:") :])
    # HTTPS: https://github.com/owner/repo[.git]
    try:
        parsed = urlparse(url)
        if "github.com" in (parsed.netloc or ""):
            return _clean_slug(parsed.path.lstrip("/"))
    except Exception:
        pass
    return None


def _clean_slug(raw: str) -> Optional[str]:
    slug = raw.removesuffix(".git").strip("/")
    parts = slug.split("/")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return f"{parts[0]}/{parts[1]}"
    return None


def _slug_from_text(text: str) -> Optional[str]:
    """Return the first plausible GitHub owner/repo slug found anywhere in text."""
    for m in _GH_URL_RE.finditer(text):
        owner, repo = m.group(1), m.group(2)
        if owner.lower() in {"features", "about", "login", "pricing", "marketplace",
                              "sponsors", "orgs", "apps", "topics", "explore"}:
            continue
        slug = _clean_slug(f"{owner}/{repo}")
        if slug:
            return slug
    return None


# ── Detection result ──────────────────────────────────────────────────────────

class Detection:
    """Result of slug detection: slug, where it came from, and confidence level."""
    __slots__ = ("slug", "source", "confidence")

    CONFIDENCE_RANK = {"override": 5, "git:upstream": 4, "git:origin": 3,
                       "git:remote": 2, "manifest": 4, "meta": 3,
                       "workflow": 2, "docs": 1, "fallback": 1}

    def __init__(self, slug: str, source: str, confidence: str = "medium"):
        self.slug = slug
        self.source = source        # e.g. "git:upstream", "package.json", "README.md"
        self.confidence = confidence  # "high" | "medium" | "low"

    def __bool__(self):  return bool(self.slug)
    def __str__(self):   return self.slug
    def __repr__(self):  return f"Detection({self.slug!r}, {self.source!r}, {self.confidence!r})"

    @property
    def rank(self) -> int:
        for key, val in self.CONFIDENCE_RANK.items():
            if self.source.startswith(key):
                return val
        return 0


_NONE_DETECTION = Detection("", "none", "low")

# ── .git-based detection ──────────────────────────────────────────────────────

def is_git_repo(path: Path) -> bool:
    """True if *path* has a .git entry (dir or worktree file)."""
    return (path / ".git").exists()


def _resolve_git_dir(path: Path) -> Optional[Path]:
    """Return the real .git directory, resolving worktree/submodule .git files."""
    git_entry = path / ".git"
    if git_entry.is_dir():
        return git_entry
    if git_entry.is_file():
        try:
            text = git_entry.read_text(encoding="utf-8", errors="replace").strip()
            if text.startswith("gitdir:"):
                target = Path(text[len("gitdir:"):].strip())
                if not target.is_absolute():
                    target = (path / target).resolve()
                common = target / "commondir"
                if common.exists():
                    rel = common.read_text(encoding="utf-8", errors="replace").strip()
                    return (target / rel).resolve() if rel != ".." else target.parent
                return target
        except Exception:
            pass
    if (path / "config").exists() and (path / "HEAD").exists():
        return path  # bare repo
    return None


def _read_git_config_remotes(git_dir: Path) -> dict[str, str]:
    """Parse .git/config → {remote_name: url}. Pure Python, no git binary needed."""
    import configparser
    remotes: dict[str, str] = {}
    config_path = git_dir / "config"
    if not config_path.is_file():
        return remotes
    try:
        cfg = configparser.RawConfigParser()
        cfg.read(str(config_path), encoding="utf-8")
        for section in cfg.sections():
            if section.startswith("remote "):
                name = section.split('"')[1] if '"' in section else section.split()[-1]
                url = cfg.get(section, "url", fallback=None)
                if url:
                    remotes[name] = url
    except Exception:
        pass
    return remotes


def detect_from_git(path: Path, preferred_remote: str = "origin") -> Optional[Detection]:
    """
    Detect GitHub slug from .git/config.
    Priority order: upstream > origin > preferred_remote > alphabetical.

    "upstream" is preferred because forked repos point origin → fork,
    upstream → original project, and we want the original.
    """
    git_dir = _resolve_git_dir(path)
    remotes: dict[str, str] = {}
    if git_dir:
        remotes = _read_git_config_remotes(git_dir)

    # Build priority list: upstream first, then origin, then preferred, then rest
    priority: list[str] = []
    for name in ["upstream", "origin", preferred_remote]:
        if name in remotes and name not in priority:
            priority.append(name)
    for name in sorted(remotes):
        if name not in priority:
            priority.append(name)

    for name in priority:
        url = remotes.get(name)
        if not url:
            continue
        slug = parse_github_slug(url)
        if not slug:
            continue
        confidence = "high" if name in ("upstream", "origin") else "medium"
        return Detection(slug, f"git:{name}", confidence)

    # subprocess fallback
    try:
        for name in ["upstream", "origin", preferred_remote]:
            r = subprocess.run(
                ["git", "-C", str(path), "remote", "get-url", name],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                slug = parse_github_slug(r.stdout.strip())
                if slug:
                    return Detection(slug, f"git:{name}", "high")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


# ── Project-file detection (zip-extracted repos, no .git) ─────────────────────

# Files grouped by category.  Each entry: (relative_path, max_bytes, category)
_PROBE_FILES: list[tuple[str, int, str]] = [
    # ── Override (user-specified, highest trust) ────────────────────────────
    (".github-repo",          256, "override"),   # one line: owner/repo
    # ── Package manifests (structured, very reliable) ──────────────────────
    ("package.json",        8_000, "manifest"),
    ("pyproject.toml",      8_000, "manifest"),
    ("setup.cfg",           8_000, "manifest"),
    ("setup.py",            8_000, "manifest"),
    ("Cargo.toml",          8_000, "manifest"),
    ("go.mod",              4_000, "manifest"),
    ("pubspec.yaml",        4_000, "manifest"),
    ("composer.json",       8_000, "manifest"),
    ("pom.xml",            16_000, "manifest"),
    ("build.gradle",       16_000, "manifest"),
    ("build.gradle.kts",   16_000, "manifest"),
    ("mix.exs",             8_000, "manifest"),
    ("deno.json",           4_000, "manifest"),
    ("deno.jsonc",          4_000, "manifest"),
    ("Chart.yaml",          4_000, "manifest"),
    ("conanfile.py",        6_000, "manifest"),
    ("CMakeLists.txt",      8_000, "manifest"),
    # ── Community / meta files ──────────────────────────────────────────────
    (".github/FUNDING.yml", 2_000, "meta"),
    ("CITATION.cff",        4_000, "meta"),
    ("codecov.yml",         2_000, "meta"),
    (".codecov.yml",        2_000, "meta"),
    ("sonar-project.properties", 4_000, "meta"),
    # ── CI/CD workflows (often reference the repo URL) ─────────────────────
    (".github/workflows/ci.yml",        4_000, "workflow"),
    (".github/workflows/release.yml",   4_000, "workflow"),
    (".github/workflows/main.yml",      4_000, "workflow"),
    (".github/workflows/build.yml",     4_000, "workflow"),
    (".github/workflows/test.yml",      4_000, "workflow"),
    ("Makefile",            8_000, "workflow"),
    # ── Docs (free text, lower confidence) ─────────────────────────────────
    ("README.md",           6_000, "docs"),
    ("README.rst",          6_000, "docs"),
    ("README.txt",          4_000, "docs"),
    ("README",              4_000, "docs"),
    ("CONTRIBUTING.md",     4_000, "docs"),
    ("CHANGELOG.md",        4_000, "docs"),
]

# Regex for context-aware extraction: "repository" or "upstream" near a GitHub URL
_CONTEXT_RE = _re.compile(
    r'(?:repository|upstream|source|homepage|url|project.url|clone.url)'
    r'[^"\n]{0,80}'
    r'github\.com[/:]([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)',
    _re.IGNORECASE | _re.MULTILINE,
)


def _read_head(path: Path, max_bytes: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except OSError:
        return ""


def _slug_from_text_with_context(text: str, category: str) -> Optional[tuple[str, str]]:
    """
    Return (slug, confidence) from text, preferring context-aware matches.
    Context matches (near 'repository', 'upstream' keywords) score higher.
    """
    # 1. Try context-aware match first (highest confidence)
    for m in _CONTEXT_RE.finditer(text):
        owner, repo = m.group(1), m.group(2)
        slug = _clean_slug(f"{owner}/{repo}")
        if slug and owner.lower() not in {"features", "about", "login", "pricing",
                                           "marketplace", "sponsors", "orgs", "apps"}:
            conf = "high" if category in ("override", "manifest", "meta") else "medium"
            return slug, conf

    # 2. Fall back to any GitHub URL
    slug = _slug_from_text(text)
    if slug:
        conf = "high" if category in ("override", "manifest") else \
               "medium" if category in ("meta", "workflow") else "low"
        return slug, conf
    return None


def slug_from_project_files(folder: Path) -> Optional[str]:
    """Simplified wrapper that returns just the slug (for backward compat)."""
    det = detect_from_files(folder)
    return det.slug if det else None


def detect_from_files(folder: Path) -> Optional[Detection]:
    """
    Scan well-known project files for a GitHub URL.
    Returns the Detection with the highest confidence found, or None.
    """
    best: Optional[Detection] = None

    # Check .github-repo override first (one line: owner/repo)
    override = folder / ".github-repo"
    if override.is_file():
        raw = _read_head(override, 256).strip().split()[0] if override.stat().st_size else ""
        slug = _clean_slug(raw)
        if slug:
            return Detection(slug, "override", "high")

    for rel, budget, category in _PROBE_FILES:
        if rel == ".github-repo":
            continue  # already handled above
        candidate = folder / rel
        if not candidate.is_file():
            continue
        text = _read_head(candidate, budget)
        result = _slug_from_text_with_context(text, category)
        if result:
            slug, conf = result
            det = Detection(slug, rel, conf)
            if best is None or det.rank > best.rank:
                best = det
                if best.rank >= 4:   # high-confidence manifest hit → stop early
                    return best

    # Scan all workflow files if we haven't hit a good match yet
    if (best is None or best.rank < 3):
        wf_dir = folder / ".github" / "workflows"
        if wf_dir.is_dir():
            for p in sorted(wf_dir.glob("*.yml"))[:10]:
                text = _read_head(p, 4_000)
                result = _slug_from_text_with_context(text, "workflow")
                if result:
                    slug, conf = result
                    det = Detection(slug, f".github/workflows/{p.name}", conf)
                    if best is None or det.rank > best.rank:
                        best = det

    # Last-resort: any remaining file in .github/
    if best is None:
        gh_dir = folder / ".github"
        if gh_dir.is_dir():
            for p in sorted(gh_dir.rglob("*"))[:30]:
                if p.is_file() and p.suffix in {".yml", ".yaml", ".md", ".json", ".txt", ""}:
                    text = _read_head(p, 2_000)
                    result = _slug_from_text_with_context(text, "fallback")
                    if result:
                        slug, conf = result
                        best = Detection(slug, f"{p.relative_to(folder)}", conf)
                        break

    return best or None


def detect_slug(path: Path, remote: str = "origin") -> Optional[str]:
    """Simplified wrapper returning just the slug (backward compat)."""
    det = detect(path, remote=remote)
    return det.slug if det else None


def detect(path: Path, remote: str = "origin") -> Optional[Detection]:
    """
    Master detector. Returns the best Detection available:
      1. git:upstream  (fork → original project)
      2. git:origin    (direct clone)
      3. project files (zip-extracted, context-scored)
    """
    git_det = detect_from_git(path, preferred_remote=remote)
    file_det = detect_from_files(path)

    if git_det and file_det:
        # Prefer whichever has higher rank; git:upstream always wins
        return git_det if git_det.rank >= file_det.rank else file_det
    return git_det or file_det


def find_github_folders(
    root: Path,
    remote: str = "origin",
    verbose: bool = False,
    deep: bool = False,
) -> list[tuple[Path, Detection]]:
    """
    Return (path, Detection) pairs for folders that resolve to a GitHub repo.
    Detects cloned repos (.git) and zip-extracted folders (project files).
    deep=True also scans one level of subdirectories.
    """
    results: list[tuple[Path, Detection]] = []
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except PermissionError:
        if verbose:
            console.print(f"  [red]✗[/]  {root} — permission denied")
        return results

    if verbose and not children:
        console.print("  [yellow]—[/]  (folder is empty)")

    for child in children:
        det = detect(child, remote=remote)
        if det:
            conf_color = {"high": "green", "medium": "yellow", "low": "dim"}.get(det.confidence, "dim")
            if verbose:
                console.print(
                    f"  [green]found[/]  {child.name}  →  [dim]{det.slug}[/]  "
                    f"[[{conf_color}]{det.source}[/]]"
                )
            results.append((child, det))
        else:
            if verbose:
                console.print(
                    f"  [dim]skip[/]   {child.name}  "
                    "[dim](no .git and no GitHub URL found)[/]"
                )
            if deep:
                try:
                    grandchildren = sorted(p for p in child.iterdir() if p.is_dir())
                except PermissionError:
                    grandchildren = []
                for gc in grandchildren:
                    gdet = detect(gc, remote=remote)
                    if gdet:
                        if verbose:
                            console.print(
                                f"    [green]found[/]  {child.name}/{gc.name}  "
                                f"→  [dim]{gdet.slug}[/]  [[dim]{gdet.source}[/]]"
                            )
                        results.append((gc, gdet))

    return results


# Backward-compat alias
def _find_github_folders_paths(root, remote="origin", verbose=False, deep=False):
    return [p for p, _ in find_github_folders(root, remote, verbose, deep)]

find_git_repos = _find_github_folders_paths



# ═════════════════════════════════════════════════════════════════════════════
# GitHub API helpers
# ═════════════════════════════════════════════════════════════════════════════


def make_session(token: Optional[str]) -> requests.Session:
    """Create a requests Session with GitHub headers and optional Bearer token."""
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    if token:
        token = token.strip()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def clone_session(session: requests.Session) -> requests.Session:
    """Return a new independent session with the same headers (safe for threads)."""
    new = requests.Session()
    new.headers.update(dict(session.headers))
    return new


def paginate(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
    limit: int = 0,
) -> list[dict]:
    """Fetch all pages from a GitHub list endpoint (optional hard cap via *limit*)."""
    results: list[dict] = []
    p = {**(params or {}), "per_page": 100, "page": 1}
    while True:
        r = session.get(url, params=p)
        if r.status_code in (404, 403, 409, 451):
            return []
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        results.extend(data)
        if limit and len(results) >= limit:
            return results[:limit]
        if "next" not in r.links:
            break
        p["page"] += 1
    return results


def safe_get(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
) -> Optional[dict]:
    try:
        r = session.get(url, params=params or {})
        if r.status_code in (404, 403, 409, 451):
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Repo core ─────────────────────────────────────────────────────────────────


def validate_token(session: requests.Session) -> tuple[bool, str]:
    """
    Check whether the session's token is accepted by GitHub.
    Returns (is_valid, message).
    Uses GET /rate_limit — costs 0 API calls against the quota.
    """
    try:
        r = session.get(f"{BASE_URL}/rate_limit", timeout=10)
    except Exception as exc:
        return False, f"Network error: {exc}"

    if r.status_code == 200:
        data = r.json()
        limit = data.get("resources", {}).get("core", {}).get("limit", 0)
        used = data.get("resources", {}).get("core", {}).get("used", 0)
        remaining = data.get("resources", {}).get("core", {}).get("remaining", 0)
        authed = limit == 5000  # authenticated limit vs 60 for unauthenticated
        return (
            True,
            f"{'authenticated' if authed else 'unauthenticated'} — {remaining}/{limit} requests remaining (used {used})",
        )
    if r.status_code == 401:
        return False, "Token rejected (401) — expired or revoked."
    return False, f"Unexpected status {r.status_code}."


def get_repo(session: requests.Session, owner: str, repo: str) -> dict:
    """
    Fetch repo metadata.  Raises RepoError (never sys.exit) so callers —
    including worker threads — can handle failures gracefully.
    """
    url = f"{BASE_URL}/repos/{owner}/{repo}"
    r = session.get(url)

    if r.status_code == 401:
        raise RepoError(
            "GitHub token is invalid or expired. "
            "Run  python main.py --check-token  to diagnose, "
            "or generate a new one at https://github.com/settings/tokens"
        )

    if r.status_code == 404:
        raise RepoError(
            f"Repository {owner}/{repo} not found (404). It may be private or deleted."
        )

    if r.status_code == 403:
        # Distinguish rate-limit from access-denied
        msg = (r.json() or {}).get("message", "")
        if "rate limit" in msg.lower():
            reset = r.headers.get("X-RateLimit-Reset", "")
            raise RepoError(
                f"Rate limit exceeded for {owner}/{repo}. "
                f"Resets at: {reset}. Use a valid token for 5000 req/hr."
            )
        raise RepoError(
            f"Access denied for {owner}/{repo} (403) — repo may be private."
        )

    r.raise_for_status()
    return r.json()


# ── File content ──────────────────────────────────────────────────────────────


def get_file_content(
    session: requests.Session,
    owner: str,
    repo: str,
    filename: str,
) -> Optional[str]:
    """
    Fetch a file's decoded text from the repo's default branch.
    Tries the given name plus common variants (lowercase, .rst, .txt).
    """
    candidates = [
        filename,
        filename.lower(),
        filename.upper(),
        filename.replace(".md", ".rst"),
        filename.replace(".md", ".txt"),
        filename.replace(".md", ""),
    ]
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        data = safe_get(session, f"{BASE_URL}/repos/{owner}/{repo}/contents/{name}")
        if not data:
            continue
        # base64-encoded file
        if isinstance(data, dict) and data.get("encoding") == "base64":
            try:
                return base64.b64decode(data["content"]).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                pass
        # fallback: download_url
        if isinstance(data, dict) and data.get("download_url"):
            try:
                dr = session.get(data["download_url"], timeout=10)
                if dr.ok:
                    return dr.text
            except Exception:
                pass
    return None


# ── Data fetchers ─────────────────────────────────────────────────────────────


def get_languages(session: requests.Session, owner: str, repo: str) -> dict[str, int]:
    data = safe_get(session, f"{BASE_URL}/repos/{owner}/{repo}/languages")
    return data or {}


def get_contributors(
    session: requests.Session, owner: str, repo: str, limit: int = 30
) -> list[dict]:
    return paginate(
        session,
        f"{BASE_URL}/repos/{owner}/{repo}/contributors",
        {"anon": "false"},
        limit=limit,
    )


def get_releases(
    session: requests.Session, owner: str, repo: str, limit: int = 10
) -> list[dict]:
    return paginate(
        session,
        f"{BASE_URL}/repos/{owner}/{repo}/releases",
        limit=limit,
    )


def get_labels(session: requests.Session, owner: str, repo: str) -> list[dict]:
    return paginate(session, f"{BASE_URL}/repos/{owner}/{repo}/labels")


def get_milestones(session: requests.Session, owner: str, repo: str) -> list[dict]:
    return paginate(
        session,
        f"{BASE_URL}/repos/{owner}/{repo}/milestones",
        {"state": "open", "sort": "due_on", "direction": "asc"},
    )


def get_all_issues(
    session: requests.Session,
    owner: str,
    repo: str,
    state: str = "open",
) -> list[dict]:
    """
    Fetch issues (PRs excluded).
    state: "open" | "closed" | "all"
    """
    all_items = paginate(
        session,
        f"{BASE_URL}/repos/{owner}/{repo}/issues",
        {"state": state, "sort": "updated", "direction": "desc"},
    )
    return [i for i in all_items if "pull_request" not in i]


def get_pull_requests(
    session: requests.Session,
    owner: str,
    repo: str,
    state: str = "open",
) -> list[dict]:
    """
    Fetch pull requests.
    state: "open" | "closed" | "all"
    Merged PRs are returned as state="closed" with merged_at set.
    """
    return paginate(
        session,
        f"{BASE_URL}/repos/{owner}/{repo}/pulls",
        {"state": state, "sort": "updated", "direction": "desc"},
    )


def get_forks(
    session: requests.Session, owner: str, repo: str, limit: int = 20
) -> list[dict]:
    return paginate(
        session,
        f"{BASE_URL}/repos/{owner}/{repo}/forks",
        {"sort": "newest"},
        limit=limit,
    )


def get_community_profile(
    session: requests.Session, owner: str, repo: str
) -> Optional[dict]:
    return safe_get(session, f"{BASE_URL}/repos/{owner}/{repo}/community/profile")


# ═════════════════════════════════════════════════════════════════════════════
# Utility / formatting helpers
# ═════════════════════════════════════════════════════════════════════════════


def fmt_date(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    days = (now - dt).days
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days}d ago"
    if days < 365:
        return f"{days // 30}mo ago"
    return f"{days // 365}y ago"


def truncate(text: str, width: int = 72) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def assignee_names(issue: dict) -> str:
    assignees = issue.get("assignees") or []
    if not assignees and issue.get("assignee"):
        assignees = [issue["assignee"]]
    return ", ".join(a["login"] for a in assignees) if assignees else "—"


def label_names(item: dict) -> str:
    labels = item.get("labels", [])
    return ", ".join(lbl["name"] for lbl in labels) if labels else "—"


def is_wip(issue: dict) -> bool:
    if issue.get("assignees") or issue.get("assignee"):
        return True
    issue_labels = {lbl["name"].lower() for lbl in issue.get("labels", [])}
    return bool(issue_labels & WIP_LABELS)


def wip_reason(issue: dict) -> str:
    reasons = []
    if issue.get("assignees") or issue.get("assignee"):
        reasons.append("assigned")
    issue_labels = {lbl["name"].lower() for lbl in issue.get("labels", [])}
    if issue_labels & WIP_LABELS:
        reasons.append("WIP label")
    return " + ".join(reasons) if reasons else "—"


def lang_pct_str(languages: dict[str, int], top: int = 6) -> str:
    total = sum(languages.values()) or 1
    parts = [
        f"{lang} {count / total * 100:.1f}%"
        for lang, count in list(languages.items())[:top]
    ]
    return ", ".join(parts) if parts else "—"


# ═════════════════════════════════════════════════════════════════════════════
# Rich display (terminal UI)
# ═════════════════════════════════════════════════════════════════════════════


def display_repo_info(
    repo: dict,
    languages: dict[str, int],
    community: Optional[dict],
) -> None:
    health_line = ""
    if community:
        score = community.get("health_percentage", 0)
        bar = "█" * (score // 10) + "░" * (10 - score // 10)
        health_line = f"\n[bold]Community Health:[/] [green]{bar}[/] {score}%"

    lang_line = (
        f"\n[bold]Languages:[/]     {lang_pct_str(languages)}" if languages else ""
    )

    lines = [
        f"[bold]Description:[/]    {repo.get('description') or '—'}",
        f"[bold]Homepage:[/]       {repo.get('homepage') or '—'}",
        f"[bold]Primary lang:[/]   {repo.get('language') or '—'}",
        f"[bold]License:[/]        {(repo.get('license') or {}).get('spdx_id') or '—'}",
        f"[bold]Visibility:[/]     {'Private 🔒' if repo.get('private') else 'Public 🌐'}",
        f"[bold]Default branch:[/] {repo.get('default_branch', 'main')}",
        f"[bold]Topics:[/]         {', '.join(repo.get('topics', [])) or '—'}",
        lang_line,
        health_line,
        "",
        (
            f"⭐ Stars: [yellow]{repo['stargazers_count']:,}[/]   "
            f"🍴 Forks: [cyan]{repo['forks_count']:,}[/]   "
            f"👁 Watchers: [green]{repo['watchers_count']:,}[/]   "
            f"🐛 Open issues: [red]{repo['open_issues_count']:,}[/]"
        ),
        "",
        (
            f"[bold]Created:[/] {fmt_date(repo.get('created_at'))}   "
            f"[bold]Updated:[/] {fmt_date(repo.get('updated_at'))}   "
            f"[bold]Pushed:[/]  {fmt_date(repo.get('pushed_at'))}"
        ),
        f"[bold]Size:[/]    {repo.get('size', 0):,} KB",
    ]
    console.print(
        Panel(
            "\n".join(l for l in lines if l is not None),
            title=f"[bold cyan]📦  {repo['full_name']}[/]",
            subtitle=f"[dim]{repo['html_url']}[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def display_guidelines(guidelines: dict) -> None:
    console.rule("[bold]📄 Guidelines & Community Files[/]")
    checks = [
        ("README", guidelines["has_readme"]),
        ("CONTRIBUTING", guidelines["has_contributing"]),
        ("LICENSE", guidelines["has_license_file"]),
    ]
    for name, present in checks:
        icon = "[green]✓[/]" if present else "[red]✗[/]"
        console.print(f"  {icon}  {name}")
    console.print()


def display_issues(issues: list[dict]) -> None:
    open_cnt   = sum(1 for i in issues if i.get("state", "open") == "open")
    closed_cnt = sum(1 for i in issues if i.get("state") == "closed")
    wip_count  = sum(1 for i in issues if i.get("state", "open") == "open" and is_wip(i))
    has_closed = closed_cnt > 0

    title = "[bold yellow]🔧 Issues[/]" if has_closed else "[bold yellow]🔧 Open Issues[/]"
    console.rule(title)
    if not issues:
        console.print("  [dim]No issues.[/]\n")
        return
    parts = [f"Open: {open_cnt}"]
    if has_closed:
        parts.append(f"Closed: {closed_cnt}")
    parts.append(f"WIP/Assigned: {wip_count}")
    console.print(f"  [dim]{' | '.join(parts)}[/]\n")

    display_slice = issues[:50]
    table = Table(box=box.ROUNDED, show_lines=True, expand=True, border_style="yellow")
    table.add_column("#", style="bold", width=6)
    table.add_column("Title", ratio=4)
    table.add_column("State", width=9)
    table.add_column("Assignees", ratio=2)
    table.add_column("Labels", ratio=2)
    table.add_column("💬", justify="right", width=5)
    table.add_column("Updated", width=11)
    for issue in display_slice:
        title_text = Text(truncate(issue["title"], 60))
        title_text.stylize(f"link {issue['html_url']}")
        st = issue.get("state", "open")
        if st == "closed":
            status = "[dim]closed[/]"
        elif is_wip(issue):
            status = "[yellow]WIP[/]"
        else:
            status = "[green]open[/]"
        table.add_row(
            str(issue["number"]),
            title_text,
            status,
            assignee_names(issue),
            label_names(issue),
            str(issue.get("comments", 0)),
            fmt_date(issue.get("updated_at")),
        )
    console.print(table)
    if len(issues) > 50:
        console.print(
            f"  [dim]… and {len(issues) - 50} more — see JSON for full list[/]"
        )
    console.print()


def display_pull_requests(prs: list[dict]) -> None:
    open_cnt   = sum(1 for p in prs if p.get("state") == "open")
    merged_cnt = sum(1 for p in prs if p.get("merged_at") is not None)
    closed_cnt = sum(1 for p in prs if p.get("state") == "closed" and p.get("merged_at") is None)
    has_history = merged_cnt > 0 or closed_cnt > 0

    title = "[bold magenta]🔀 Pull Requests[/]" if has_history else "[bold magenta]🔀 Open Pull Requests[/]"
    console.rule(title)
    if not prs:
        console.print("  [dim]No pull requests.[/]\n")
        return
    parts = [f"Open: {open_cnt}"]
    if merged_cnt:
        parts.append(f"Merged: {merged_cnt}")
    if closed_cnt:
        parts.append(f"Closed (unmerged): {closed_cnt}")
    console.print(f"  [dim]{' | '.join(parts)}[/]\n")

    table = Table(box=box.ROUNDED, show_lines=True, expand=True, border_style="magenta")
    table.add_column("#", style="bold", width=6)
    table.add_column("Title", ratio=4)
    table.add_column("State", width=10)
    table.add_column("Author", ratio=2)
    table.add_column("Base ← Head", ratio=3)
    table.add_column("Labels", ratio=2)
    table.add_column("Updated", width=11)
    for pr in prs[:30]:
        title_text = Text(truncate(pr["title"], 58))
        title_text.stylize(f"link {pr['html_url']}")
        base = pr.get("base", {}).get("label", "?")
        head = pr.get("head", {}).get("label", "?")
        short = lambda ref: ref.split(":", 1)[-1]
        if pr.get("merged_at"):
            state_cell = "[blue]merged[/]"
        elif pr.get("state") == "closed":
            state_cell = "[dim]closed[/]"
        elif pr.get("draft"):
            state_cell = "[dim]draft[/]"
        else:
            state_cell = "[green]open[/]"
        table.add_row(
            str(pr["number"]),
            title_text,
            state_cell,
            pr.get("user", {}).get("login", "—"),
            f"{short(base)} ← {short(head)}",
            label_names(pr),
            fmt_date(pr.get("updated_at")),
        )
    console.print(table)
    if len(prs) > 30:
        console.print(f"  [dim]… and {len(prs) - 30} more[/]")
    console.print(f"  [dim]Total: {len(prs)} PR(s)[/]\n")


def display_releases(releases: list[dict]) -> None:
    console.rule("[bold green]🚀 Recent Releases[/]")
    if not releases:
        console.print("  [dim]No releases found.[/]\n")
        return
    table = Table(box=box.ROUNDED, show_lines=True, expand=True, border_style="green")
    table.add_column("Tag", style="bold cyan", ratio=2)
    table.add_column("Name", ratio=3)
    table.add_column("Author", ratio=2)
    table.add_column("Type", width=9)
    table.add_column("Published", width=12)
    for rel in releases[:10]:
        tag_text = Text(rel.get("tag_name", "—"))
        if rel.get("html_url"):
            tag_text.stylize(f"link {rel['html_url']}")
        rel_type = "[yellow]pre[/]" if rel.get("prerelease") else "[green]stable[/]"
        if rel.get("draft"):
            rel_type = "[dim]draft[/]"
        table.add_row(
            tag_text,
            truncate(rel.get("name") or rel.get("tag_name", "—"), 40),
            rel.get("author", {}).get("login", "—"),
            rel_type,
            fmt_date(rel.get("published_at")),
        )
    console.print(table)
    console.print(
        f"  [dim]Showing {min(10, len(releases))} of {len(releases)} release(s)[/]\n"
    )


def display_contributors(contributors: list[dict]) -> None:
    console.rule("[bold blue]👥 Top Contributors[/]")
    if not contributors:
        console.print("  [dim]No contributor data available.[/]\n")
        return
    table = Table(box=box.ROUNDED, show_lines=True, expand=True, border_style="blue")
    table.add_column("Login", ratio=2)
    table.add_column("Contributions", justify="right", width=14)
    table.add_column("Profile URL", ratio=3)
    for c in contributors[:20]:
        profile = Text(c.get("html_url", ""))
        profile.stylize(f"link {c.get('html_url', '')}")
        table.add_row(
            c.get("login", "?"),
            str(c.get("contributions", 0)),
            profile,
        )
    console.print(table)
    if len(contributors) > 20:
        console.print(f"  [dim]… and {len(contributors) - 20} more[/]")
    console.print()


def display_forks(forks: list[dict]) -> None:
    console.rule("[bold cyan]🍴 Top Forks (most recent)[/]")
    if not forks:
        console.print("  [dim]No forks.[/]\n")
        return
    table = Table(box=box.ROUNDED, show_lines=True, expand=True, border_style="cyan")
    table.add_column("Fork", ratio=3)
    table.add_column("⭐", justify="right", width=7)
    table.add_column("🍴", justify="right", width=7)
    table.add_column("Open Issues", justify="right", width=12)
    table.add_column("Last Pushed", width=12)
    for fork in forks:
        repo_text = Text(fork["full_name"])
        repo_text.stylize(f"link {fork['html_url']}")
        table.add_row(
            repo_text,
            str(fork.get("stargazers_count", 0)),
            str(fork.get("forks_count", 0)),
            str(fork.get("open_issues_count", 0)),
            fmt_date(fork.get("pushed_at")),
        )
    console.print(table)
    console.print(f"  [dim]Showing {len(forks)} fork(s)[/]\n")


def display_labels(labels: list[dict]) -> None:
    if not labels:
        return
    console.rule("[bold]🏷  Labels[/]")
    items = [lbl["name"] for lbl in labels]
    console.print("  " + "  •  ".join(items[:30]))
    if len(labels) > 30:
        console.print(f"  [dim]… and {len(labels) - 30} more[/]")
    console.print()


def display_milestones(milestones: list[dict]) -> None:
    if not milestones:
        return
    console.rule("[bold]🎯 Open Milestones[/]")
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("Title", ratio=3)
    table.add_column("Open", justify="right", width=8)
    table.add_column("Closed", justify="right", width=8)
    table.add_column("Due", width=12)
    for ms in milestones:
        table.add_row(
            ms.get("title", "?"),
            str(ms.get("open_issues", 0)),
            str(ms.get("closed_issues", 0)),
            fmt_date(ms.get("due_on")),
        )
    console.print(table)
    console.print()


# ═════════════════════════════════════════════════════════════════════════════
# JSON export helpers
# ═════════════════════════════════════════════════════════════════════════════


def _issue_dict(issue: dict) -> dict:
    return {
        "number": issue["number"],
        "title": issue["title"],
        "url": issue["html_url"],
        "state": issue["state"],
        "is_wip": is_wip(issue),
        "wip_reason": wip_reason(issue),
        "assignees": [a["login"] for a in (issue.get("assignees") or [])],
        "labels": [lbl["name"] for lbl in issue.get("labels", [])],
        "milestone": (issue.get("milestone") or {}).get("title"),
        "comments": issue.get("comments", 0),
        "reactions_total": (issue.get("reactions") or {}).get("total_count", 0),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "body_preview": truncate((issue.get("body") or "").strip(), 500),
    }


def _pr_dict(pr: dict) -> dict:
    merged_at = pr.get("merged_at")
    state = pr.get("state", "open")
    resolved = "merged" if merged_at else state
    return {
        "number": pr["number"],
        "title": pr["title"],
        "url": pr["html_url"],
        "state": resolved,               # "open" | "closed" | "merged"
        "author": pr.get("user", {}).get("login"),
        "draft": pr.get("draft", False),
        "base": pr.get("base", {}).get("label"),
        "head": pr.get("head", {}).get("label"),
        "labels": [lbl["name"] for lbl in pr.get("labels", [])],
        "milestone": (pr.get("milestone") or {}).get("title"),
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
        "merged_at": merged_at,
        "closed_at": pr.get("closed_at"),
        "body_preview": truncate((pr.get("body") or "").strip(), 500),
    }


def _release_dict(rel: dict) -> dict:
    return {
        "tag": rel.get("tag_name"),
        "name": rel.get("name"),
        "url": rel.get("html_url"),
        "author": rel.get("author", {}).get("login"),
        "prerelease": rel.get("prerelease", False),
        "draft": rel.get("draft", False),
        "published_at": rel.get("published_at"),
        "body_preview": truncate((rel.get("body") or "").strip(), 500),
    }


def _contributor_dict(c: dict) -> dict:
    return {
        "login": c.get("login"),
        "contributions": c.get("contributions", 0),
        "url": c.get("html_url"),
        "avatar": c.get("avatar_url"),
    }


def build_summary(
    repo: dict,
    issues: list[dict],
    prs: list[dict],
    releases: list[dict],
    forks: list[dict],
    contributors: list[dict],
    languages: dict[str, int],
    labels: list[dict],
    milestones: list[dict],
    readme: Optional[str],
    contributing: Optional[str],
    license_text: Optional[str],
    community: Optional[dict],
) -> dict:
    # ── Split issues by state ────────────────────────────────────────────────
    open_issues   = [i for i in issues if i.get("state", "open") == "open"]
    closed_issues = [i for i in issues if i.get("state") == "closed"]
    wip_issues    = [i for i in open_issues if is_wip(i)]

    # ── Split PRs by state (merged PRs are state=closed with merged_at set) ─
    open_prs   = [p for p in prs if p.get("state") == "open"]
    merged_prs = [p for p in prs if p.get("merged_at") is not None]
    closed_prs = [p for p in prs if p.get("state") == "closed" and p.get("merged_at") is None]

    license_spdx = (repo.get("license") or {}).get("spdx_id")
    lang_total = sum(languages.values()) or 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        # ── Core repo info ──────────────────────────────────────────────────
        "repo": {
            "full_name": repo["full_name"],
            "url": repo["html_url"],
            "description": repo.get("description"),
            "language": repo.get("language"),
            "license": license_spdx,
            "license_name": (repo.get("license") or {}).get("name"),
            "private": repo.get("private"),
            "archived": repo.get("archived", False),
            "disabled": repo.get("disabled", False),
            "fork": repo.get("fork", False),
            "default_branch": repo.get("default_branch"),
            "homepage": repo.get("homepage"),
            "stars": repo.get("stargazers_count"),
            "forks": repo.get("forks_count"),
            "watchers": repo.get("watchers_count"),
            "open_issues_count": repo.get("open_issues_count"),
            "size_kb": repo.get("size"),
            "topics": repo.get("topics", []),
            "has_wiki": repo.get("has_wiki"),
            "has_issues": repo.get("has_issues"),
            "has_projects": repo.get("has_projects"),
            "has_discussions": repo.get("has_discussions"),
            "allow_forking": repo.get("allow_forking"),
            "is_template": repo.get("is_template", False),
            "created_at": repo.get("created_at"),
            "updated_at": repo.get("updated_at"),
            "pushed_at": repo.get("pushed_at"),
        },
        # ── Quick-glance stats ──────────────────────────────────────────────
        "stats": {
            "open_issues": len(open_issues),
            "closed_issues": len(closed_issues),
            "total_issues": len(issues),
            "wip_assigned_issues": len(wip_issues),
            "open_prs": len(open_prs),
            "closed_prs": len(closed_prs),
            "merged_prs": len(merged_prs),
            "total_prs": len(prs),
            "draft_prs": sum(1 for p in prs if p.get("draft")),
            "total_releases": len(releases),
            "total_contributors": len(contributors),
            "total_labels": len(labels),
            "open_milestones": len(milestones),
        },
        # ── Languages ──────────────────────────────────────────────────────
        "languages": {
            lang: {
                "bytes": count,
                "pct": round(count / lang_total * 100, 2),
            }
            for lang, count in languages.items()
        },
        # ── Community & guidelines ──────────────────────────────────────────
        "community_health": community,
        "guidelines": {
            "has_readme": readme is not None,
            "has_contributing": contributing is not None,
            "has_license_file": license_text is not None,
            "readme_preview": truncate(readme.strip(), 3000) if readme else None,
            "contributing_preview": (
                truncate(contributing.strip(), 3000) if contributing else None
            ),
            "license_preview": (
                truncate(license_text.strip(), 1500) if license_text else None
            ),
        },
        # ── Issues ─────────────────────────────────────────────────────────
        "issues": {
            "open": [_issue_dict(i) for i in open_issues],
            "closed": [_issue_dict(i) for i in closed_issues],
            "wip_assigned": [_issue_dict(i) for i in wip_issues],
            "open_count": len(open_issues),
            "closed_count": len(closed_issues),
        },
        # ── Pull requests ───────────────────────────────────────────────────
        "pull_requests": {
            "open": [_pr_dict(p) for p in open_prs],
            "closed": [_pr_dict(p) for p in closed_prs],
            "merged": [_pr_dict(p) for p in merged_prs],
            "open_count": len(open_prs),
            "closed_count": len(closed_prs),
            "merged_count": len(merged_prs),
        },
        # ── Releases ───────────────────────────────────────────────────────
        "releases": [_release_dict(r) for r in releases],
        # ── Contributors ───────────────────────────────────────────────────
        "contributors": [_contributor_dict(c) for c in contributors],
        # ── Labels & milestones ────────────────────────────────────────────
        "labels": [
            {
                "name": lbl["name"],
                "color": lbl.get("color"),
                "description": lbl.get("description"),
            }
            for lbl in labels
        ],
        "milestones": [
            {
                "title": ms.get("title"),
                "description": ms.get("description"),
                "due_on": ms.get("due_on"),
                "open_issues": ms.get("open_issues"),
                "closed_issues": ms.get("closed_issues"),
                "url": ms.get("html_url"),
            }
            for ms in milestones
        ],
        # ── Forks ──────────────────────────────────────────────────────────
        "top_forks": [
            {
                "full_name": f["full_name"],
                "url": f["html_url"],
                "owner": f.get("owner", {}).get("login"),
                "stars": f.get("stargazers_count", 0),
                "forks": f.get("forks_count", 0),
                "open_issues": f.get("open_issues_count", 0),
                "default_branch": f.get("default_branch"),
                "pushed_at": f.get("pushed_at"),
                "created_at": f.get("created_at"),
            }
            for f in forks
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Core fetch orchestrator
# ═════════════════════════════════════════════════════════════════════════════


def fetch_all(
    session: requests.Session,
    owner: str,
    repo_name: str,
    fork_limit: int = 20,
    all_issues: bool = False,
    all_prs: bool = False,
    progress: Optional[Progress] = None,
    task_id: Optional[Any] = None,
) -> dict:
    """Fetch every data point for one repo. Returns a raw-data dict."""

    def upd(msg: str) -> None:
        if progress is not None and task_id is not None:
            progress.update(task_id, description=msg)

    upd(f"[cyan]{owner}/{repo_name}[/] — repo info")
    repo = get_repo(session, owner, repo_name)

    issue_state = "all" if all_issues else "open"
    upd(f"[cyan]{owner}/{repo_name}[/] — issues ({issue_state})")
    issues = get_all_issues(session, owner, repo_name, state=issue_state)

    pr_state = "all" if all_prs else "open"
    upd(f"[cyan]{owner}/{repo_name}[/] — pull requests ({pr_state})")
    prs = get_pull_requests(session, owner, repo_name, state=pr_state)

    upd(f"[cyan]{owner}/{repo_name}[/] — releases")
    releases = get_releases(session, owner, repo_name)

    upd(f"[cyan]{owner}/{repo_name}[/] — forks")
    forks = get_forks(session, owner, repo_name, limit=fork_limit)

    upd(f"[cyan]{owner}/{repo_name}[/] — contributors")
    contributors = get_contributors(session, owner, repo_name)

    upd(f"[cyan]{owner}/{repo_name}[/] — languages")
    languages = get_languages(session, owner, repo_name)

    upd(f"[cyan]{owner}/{repo_name}[/] — labels")
    labels = get_labels(session, owner, repo_name)

    upd(f"[cyan]{owner}/{repo_name}[/] — milestones")
    milestones = get_milestones(session, owner, repo_name)

    upd(f"[cyan]{owner}/{repo_name}[/] — README / CONTRIBUTING / LICENSE")
    readme = get_file_content(session, owner, repo_name, "README.md")
    contributing = get_file_content(session, owner, repo_name, "CONTRIBUTING.md")
    license_text = get_file_content(session, owner, repo_name, "LICENSE")
    if not license_text:
        license_text = get_file_content(session, owner, repo_name, "LICENSE.md")

    upd(f"[cyan]{owner}/{repo_name}[/] — community profile")
    community = get_community_profile(session, owner, repo_name)

    return dict(
        repo=repo,
        issues=issues,
        prs=prs,
        releases=releases,
        forks=forks,
        contributors=contributors,
        languages=languages,
        labels=labels,
        milestones=milestones,
        readme=readme,
        contributing=contributing,
        license_text=license_text,
        community=community,
    )


_FETCH_KEYS = (
    "repo",
    "issues",
    "prs",
    "releases",
    "forks",
    "contributors",
    "languages",
    "labels",
    "milestones",
    "readme",
    "contributing",
    "license_text",
    "community",
)


def summarize_single(
    session: requests.Session,
    slug: str,
    output_path: Optional[Path],
    show_ui: bool,
    fork_limit: int = 20,
    local_path: Optional[Path] = None,
    quiet: bool = False,
    all_issues: bool = False,
    all_prs: bool = False,
) -> dict:
    """Fetch, optionally display, save, and return one repo's summary."""
    owner, repo_name = slug.split("/", 1)

    # Each call gets its own session clone so token state is never shared
    sess = clone_session(session)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Fetching {owner}/{repo_name} …", total=None)
        try:
            data = fetch_all(
                sess,
                owner,
                repo_name,
                fork_limit=fork_limit,
                all_issues=all_issues,
                all_prs=all_prs,
                progress=progress,
                task_id=task,
            )
        except RepoError as exc:
            console.print(f"[bold red]✗[/]  {exc}")
            sys.exit(1)

    summary = build_summary(**{k: data[k] for k in _FETCH_KEYS})

    if show_ui:
        console.print()
        display_repo_info(data["repo"], data["languages"], data["community"])
        console.print()
        display_guidelines(summary["guidelines"])
        display_issues(data["issues"])
        display_pull_requests(data["prs"])
        display_releases(data["releases"])
        display_contributors(data["contributors"])
        display_forks(data["forks"])
        display_labels(data["labels"])
        display_milestones(data["milestones"])

    # ── Determine where to write ───────────────────────────────────────────
    if output_path:
        json_path = output_path
    elif local_path:
        json_path = local_path / "github_summary.json"
    else:
        json_path = Path(f"{owner}_{repo_name}_github_summary.json")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not quiet:
        console.print(
            Panel(
                f"[green]✓[/]  [bold]{owner}/{repo_name}[/] → [cyan]{json_path}[/]",
                border_style="green",
                padding=(0, 2),
            )
        )

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="github-summarizer",
        description=(
            "Summarize GitHub repos — local paths or owner/repo slugs.\n\n"
            "  SINGLE  python main.py path/to/repo\n"
            "  MULTI   python main.py path/to/parent/folder\n"
            "  REMOTE  python main.py owner/repo\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "target",
        help=(
            "Local path to a git repo (single mode), a parent folder containing "
            "multiple git repos (multi mode), or a GitHub 'owner/repo' slug."
        ),
    )
    parser.add_argument(
        "--token",
        "-t",
        default=None,
        metavar="TOKEN",
        help="GitHub Personal Access Token (default: $GITHUB_TOKEN env var or .env file)",
    )
    parser.add_argument(
        "--env-file",
        "-e",
        default=".env",
        metavar="FILE",
        help="Path to a .env file to load (default: .env in the current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help=(
            "Output path for the JSON report. "
            "Single mode: path to file. "
            "Multi mode: path to a folder (individual files written there). "
            "Default: github_summary.json inside each repo folder."
        ),
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help=(
            "Display full rich terminal UI with tables. "
            "Enabled by default in single mode; pass to enable in multi mode."
        ),
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Suppress the terminal UI entirely (JSON-only output).",
    )
    parser.add_argument(
        "--fork-limit",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of forks to fetch and include (default: 20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel workers for multi mode (default: 4)",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        metavar="NAME",
        help="Which git remote to read when resolving local paths (default: origin)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=(
            "Print per-folder diagnostics during multi-mode scanning "
            "(shows why folders are skipped: no .git, no GitHub URL found, etc.)"
        ),
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help=(
            "Also scan one level of subdirectories in multi mode. "
            "Useful when repos are nested inside topic folders."
        ),
    )
    parser.add_argument(
        "--check-token",
        action="store_true",
        help="Validate your GitHub token and show rate-limit status, then exit.",
    )
    parser.add_argument(
        "--all-issues",
        action="store_true",
        dest="all_issues",
        help=(
            "Fetch ALL issues (open + closed) instead of only open ones. "
            "Warning: can be slow on repos with thousands of closed issues."
        ),
    )
    parser.add_argument(
        "--all-prs",
        action="store_true",
        dest="all_prs",
        help=(
            "Fetch ALL pull requests (open + closed + merged) instead of only open. "
            "Warning: can be slow on high-activity repos."
        ),
    )
    parser.add_argument(
        "--list-only",
        "--dry-run",
        action="store_true",
        dest="list_only",
        help=(
            "Scan for repos and print detection results (slug, source, confidence) "
            "without making any GitHub API calls. Useful for verifying detection."
        ),
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help=(
            "Write a github_summary.md report alongside the JSON file. "
            "Includes a human-readable summary of the repo metadata."
        ),
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help=(
            "Multi mode only: write a master _github_index.json at the parent "
            "folder level summarising all processed repos in one file."
        ),
    )
    return parser


def resolve_token(args: argparse.Namespace) -> Optional[str]:
    if _DOTENV_AVAILABLE:
        loaded = _load_dotenv(dotenv_path=args.env_file, override=False)
        if loaded:
            console.print(f"[dim]✓ Loaded env from [cyan]{args.env_file}[/][/]\n")
        elif args.env_file != ".env":
            console.print(
                f"[bold yellow]⚠[/]  .env not found at [cyan]{args.env_file}[/]. "
                "Falling back to environment variables.\n"
            )

    token = args.token or (os.environ.get("GITHUB_TOKEN") or "").strip() or None
    return token.strip() if token else None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    token = resolve_token(args)

    session = make_session(token)

    # ── Token validation (costs 0 API quota) ─────────────────────────────────
    ok, rate_msg = validate_token(session)
    if not token:
        console.print(
            "[bold yellow]⚠[/]  No GitHub token — rate-limited to 60 req/hr.\n"
            "    Add [cyan]GITHUB_TOKEN=ghp_...[/] to your .env or use [cyan]--token[/]\n"
        )
    elif not ok:
        console.print(
            Panel(
                f"[bold red]✗  Token invalid:[/] {rate_msg}\n\n"
                "  Your token has likely expired or been revoked.\n\n"
                "  [bold]Fix:[/]\n"
                "  1. Go to [cyan]https://github.com/settings/tokens[/]\n"
                "  2. Generate a new classic token with [dim]public_repo[/] scope\n"
                "  3. Replace the value in your [cyan].env[/] file:\n"
                "     [dim]GITHUB_TOKEN=ghp_your_new_token_here[/]",
                title="[red]Authentication Error[/]",
                border_style="red",
                padding=(1, 2),
            )
        )
        sys.exit(1)
    else:
        console.print(f"[dim]✓ Token OK — {rate_msg}[/]\n")

    # ── --check-token: just print status and exit ─────────────────────────────
    if getattr(args, "check_token", False):
        if ok:
            console.print(
                Panel(
                    f"[green]✓ Token is valid[/]\n{rate_msg}",
                    border_style="green",
                    padding=(0, 2),
                )
            )
        else:
            console.print(
                Panel(
                    f"[red]✗ Token is invalid[/]\n{rate_msg}\n\n"
                    "  Get a new one at [cyan]https://github.com/settings/tokens[/]",
                    border_style="red",
                    padding=(0, 2),
                )
            )
        return

    target = args.target

    # ── Determine mode ────────────────────────────────────────────────────────
    target_path = Path(target).resolve()
    path_exists = target_path.exists()

    # Looks like owner/repo slug (no path separators, exactly one /)
    looks_like_slug = (
        not path_exists
        and "\\" not in target
        and target.count("/") == 1
        and all(p.strip() for p in target.split("/"))
    )

    # ── Shared fetch flags ────────────────────────────────────────────────────
    _all_issues = getattr(args, "all_issues", False)
    _all_prs    = getattr(args, "all_prs",    False)

    # ── Mode: remote-only owner/repo slug ─────────────────────────────────────
    if looks_like_slug:
        show_ui = not args.no_ui
        output_path = Path(args.output) if args.output else None
        console.print(f"\n[bold]Single (remote) mode:[/] [cyan]{target}[/]\n")
        summarize_single(
            session,
            target,
            output_path,
            show_ui=show_ui,
            fork_limit=args.fork_limit,
            all_issues=_all_issues,
            all_prs=_all_prs,
        )
        return

    # ── Path must exist ───────────────────────────────────────────────────────
    if not path_exists:
        console.print(f"[bold red]✗[/] Path not found: [cyan]{target_path}[/]")
        sys.exit(1)

    # ── Mode: single local repo (git clone or zip-extracted) ─────────────────
    if is_git_repo(target_path) or slug_from_project_files(target_path):
        slug = detect_slug(target_path, remote=args.remote)
        if not slug:
            console.print(
                f"[bold red]✗[/] Could not determine GitHub repo for [cyan]{target_path}[/].\n"
                "    If it's a git clone, check the remote: "
                "[cyan]git remote -v[/]\n"
                "    If it's a zip download, ensure a project file "
                "(README, package.json, etc.) contains a GitHub URL."
            )
            sys.exit(1)

        show_ui = not args.no_ui if not args.ui else True
        output_path = Path(args.output) if args.output else None
        src_tag = "git" if is_git_repo(target_path) else "files"
        console.print(
            f"\n[bold]Single (local) mode:[/] [cyan]{slug}[/]  "
            f"[dim]({target_path})  [{src_tag}][/]\n"
        )
        summarize_single(
            session,
            slug,
            output_path,
            show_ui=show_ui,
            fork_limit=args.fork_limit,
            local_path=target_path,
            all_issues=_all_issues,
            all_prs=_all_prs,
        )
        return

    # ── Mode: multi-repo folder ───────────────────────────────────────────────
    console.print(f"\n[bold]Multi mode:[/] scanning [cyan]{target_path}[/] …\n")

    verbose = getattr(args, "verbose", False)
    deep = getattr(args, "deep", False)
    if verbose:
        console.print("[dim]Verbose scan:[/]")
    repos = find_github_folders(
        target_path,
        remote=args.remote,
        verbose=verbose,
        deep=deep,
    )
    if verbose:
        console.print()

    if not repos:
        try:
            all_children = [p for p in target_path.iterdir() if p.is_dir()]
        except PermissionError:
            all_children = []
        git_dirs = [p for p in all_children if is_git_repo(p)]

        msg_lines = [
            "[bold yellow]⚠[/]  No repos to process.\n",
            f"    Subdirectories scanned  : [cyan]{len(all_children)}[/]",
            f"    With .git folder        : [cyan]{len(git_dirs)}[/]",
            f"    Detected via proj files : [cyan]0[/]",
            "",
            "    [bold]Tips:[/]",
            "      • Run with [cyan]--verbose[/] to see why each folder was skipped.",
            "      • If repos were cloned, check the remote: "
            "[cyan]git -C <repo> remote -v[/]",
            "      • Use [cyan]--remote <name>[/] if your remote isn't 'origin'.",
            "      • If repos are nested deeper, add [cyan]--deep[/].",
            "      • If repos were downloaded as zips, make sure they contain a",
            "        [dim]package.json / pyproject.toml / README.md[/] with a GitHub URL.",
        ]
        console.print("\n".join(msg_lines))
        sys.exit(1)

    # ── Confidence colour helper ─────────────────────────────────────────────
    _conf_style = {"high": "green", "medium": "yellow", "low": "dim"}

    console.print(f"Found [bold green]{len(repos)}[/] repo(s):\n")
    for rp, det in repos:
        conf_tag = f"[{_conf_style.get(det.confidence, 'dim')}]{det.confidence}[/]"
        console.print(
            f"  [cyan]•[/]  {rp.name:<30}  [dim]{det.slug}[/]"
            f"  [dim]via {det.source}[/]  {conf_tag}"
        )
    console.print()

    # ── --list-only: stop here, no API calls ─────────────────────────────────
    list_only = getattr(args, "list_only", False)
    if list_only:
        console.print(
            Panel(
                f"[bold]Dry-run complete.[/]  "
                f"{len(repos)} repo(s) detected — no API calls made.\n"
                "Remove [cyan]--list-only[/] to fetch and summarise.",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        return

    show_ui_multi = args.ui and not args.no_ui
    write_markdown = getattr(args, "markdown", False)
    write_index    = getattr(args, "index", False)
    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    errors:  dict[str, str]  = {}

    def _make_markdown(slug: str, summary: dict) -> str:
        """Convert a summary dict to a readable Markdown report."""
        r = summary.get("repo", {})
        lines = [
            f"# {r.get('full_name', slug)}",
            "",
            f"> {r.get('description') or '_No description._'}",
            "",
            (
                f"**Stars** {r.get('stargazers_count', 0):,}  |  "
                f"**Forks** {r.get('forks_count', 0):,}  |  "
                f"**Watchers** {r.get('watchers_count', 0):,}  |  "
                f"**Open issues** {r.get('open_issues_count', 0):,}"
            ),
            "",
            (
                f"**Language** {r.get('language') or 'n/a'}  |  "
                f"**License** {(r.get('license') or {}).get('spdx_id', 'n/a')}"
            ),
            "",
        ]
        if r.get("topics"):
            lines += [f"**Topics:** {', '.join(r['topics'])}", ""]
        if r.get("homepage"):
            lines += [f"**Homepage:** {r['homepage']}", ""]
        # Issues
        iss = summary.get("issues", {})
        open_i  = iss.get("open_count",   0) if isinstance(iss, dict) else 0
        close_i = iss.get("closed_count", 0) if isinstance(iss, dict) else 0
        lines += ["## Issues", "", f"Open: {open_i}  |  Closed: {close_i}", ""]
        # PRs
        prs = summary.get("pull_requests", {})
        open_p  = prs.get("open_count",   0) if isinstance(prs, dict) else 0
        close_p = prs.get("closed_count", 0) if isinstance(prs, dict) else 0
        merged  = prs.get("merged_count", 0) if isinstance(prs, dict) else 0
        lines += [
            "## Pull Requests", "",
            f"Open: {open_p}  |  Closed: {close_p}  |  Merged: {merged}", "",
        ]
        # Releases
        releases = summary.get("releases", [])
        if releases:
            lines += ["## Releases", ""]
            for rel in releases[:5]:
                lines.append(f"- **{rel.get('tag_name', '?')}** — {rel.get('name', '')}")
            lines.append("")
        # Contributors
        contribs = summary.get("contributors", [])
        if contribs:
            lines += ["## Top Contributors", ""]
            for c in contribs[:10]:
                login = c.get("login", "?")
                lines.append(
                    f"- [{login}](https://github.com/{login})"
                    f" ({c.get('contributions', 0)} commits)"
                )
            lines.append("")
        det_info = summary.get("_detection", {})
        lines += [
            "---",
            f"_Detected via `{det_info.get('source', '?')}`"
            f" (confidence: {det_info.get('confidence', '?')})_  ",
            f"_Generated by github-summarizer · {summary.get('fetched_at', '')}_",
        ]
        return "\n".join(lines)

    def _process(repo_path: Path, det: Detection) -> tuple[str, Optional[dict], str]:
        slug = det.slug
        owner, repo_name = slug.split("/", 1)

        # Skip slugs that are obviously template placeholders
        if owner.lower() in {p.lower() for p in _PLACEHOLDER_OWNERS}:
            return repo_path.name, None, f"Skipped — placeholder slug ({slug})"

        # Give each worker its own session clone so token state is never shared
        worker_session = clone_session(session)

        try:
            data = fetch_all(
                worker_session, owner, repo_name,
                fork_limit=args.fork_limit,
                all_issues=_all_issues,
                all_prs=_all_prs,
            )
            summary = build_summary(**{k: data[k] for k in _FETCH_KEYS})
            # Embed detection metadata into the summary
            summary["_detection"] = {
                "source": det.source,
                "confidence": det.confidence,
                "local_path": str(repo_path),
            }

            json_path = (
                out_dir / f"{repo_name}_github_summary.json"
                if out_dir
                else repo_path / "github_summary.json"
            )
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            if write_markdown:
                md_path = json_path.with_suffix(".md")
                md_path.write_text(_make_markdown(slug, summary), encoding="utf-8")

            if show_ui_multi:
                console.rule(f"[bold cyan]{slug}[/]")
                display_repo_info(data["repo"], data["languages"], data["community"])
                display_issues(data["issues"])
                display_pull_requests(data["prs"])

            return repo_path.name, summary, str(json_path)
        except (RepoError, Exception) as exc:
            return repo_path.name, None, str(exc)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("Processing repos ...", total=len(repos))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process, rp, det): rp
                for rp, det in repos
            }
            for future in as_completed(futures):
                name, summary, info = future.result()
                if summary is not None:
                    results[name] = summary
                    console.print(f"  [green]✓[/]  {name}  ->  [dim]{info}[/]")
                else:
                    errors[name] = info
                    console.print(f"  [red]✗[/]  {name}: {info}")
                progress.advance(overall)

    # ── Master index JSON ─────────────────────────────────────────────────────
    if write_index and results:
        import datetime as _dt
        index_path = target_path / "_github_index.json"
        index_data = {
            "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
            "source_folder": str(target_path),
            "repo_count": len(results),
            "repos": {
                name: {
                    "slug": s.get("repo", {}).get("full_name", "?"),
                    "stars": s.get("repo", {}).get("stargazers_count", 0),
                    "forks": s.get("repo", {}).get("forks_count", 0),
                    "open_issues": s.get("repo", {}).get("open_issues_count", 0),
                    "language": s.get("repo", {}).get("language"),
                    "detection_source": s.get("_detection", {}).get("source"),
                    "detection_confidence": s.get("_detection", {}).get("confidence"),
                }
                for name, s in results.items()
            },
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        console.print(f"[dim]Master index  ->  [cyan]{index_path}[/][/]")

    console.print()
    dest_note = (
        f"Output folder: [cyan]{out_dir}[/]"
        if out_dir
        else "Saved as [cyan]github_summary.json[/] inside each repo folder"
    )
    md_note  = "  +  [cyan].md[/] reports" if write_markdown else ""
    idx_note = "  +  [cyan]_github_index.json[/]" if write_index else ""
    console.print(
        Panel(
            f"[green]Done![/]  "
            f"{len(results)} succeeded  |  {len(errors)} failed\n"
            f"{dest_note}{md_note}{idx_note}",
            border_style="green",
            padding=(1, 2),
        )
    )

    if errors:
        console.print("\n[bold red]Failures:[/]")
        token_errors = 0
        for name, msg in errors.items():
            console.print(f"  [red]x[/]  {name}: {msg}")
            if "token" in msg.lower() or "401" in msg or "rate limit" in msg.lower():
                token_errors += 1
        if token_errors:
            console.print(
                "\n  [bold yellow]Warning:[/]  Token or rate-limit issues detected.\n"
                "    Generate a fresh token at [cyan]https://github.com/settings/tokens[/]\n"
                "    (classic token, [dim]public_repo[/] scope)\n"
                "    Then update your [cyan].env[/]: [dim]GITHUB_TOKEN=ghp_...[/]"
            )


if __name__ == "__main__":
    main()
