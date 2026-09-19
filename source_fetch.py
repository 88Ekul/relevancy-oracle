"""Read bounded GitHub source snapshots at an immutable commit, without Git.

Only source text is cached; generated judgements never enter the catalogue.
Remote code is data and is never imported or executed by this module.
"""
from __future__ import annotations

import base64
import ast
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests


class SourceError(RuntimeError):
    """A bounded, credential-free explanation of a source retrieval failure."""


@dataclass(frozen=True)
class SourceLimits:
    max_files: int = 12
    max_chars: int = 60000
    max_file_chars: int = 12000
    max_file_bytes: int = 128000
    max_tree_paths: int = 1500


CODE_SUFFIXES = {'.py', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs', '.go',
                 '.rs', '.java', '.kt', '.swift', '.cs', '.c', '.cpp', '.h',
                 '.hpp', '.rb', '.php', '.ex', '.exs', '.scala', '.vue',
                 '.svelte', '.sh', '.sql', '.r', '.dart', '.lua'}
MANIFESTS = {'pyproject.toml', 'requirements.txt', 'setup.py', 'setup.cfg',
             'package.json', 'cargo.toml', 'go.mod', 'gemfile', 'pom.xml',
             'composer.json', 'mix.exs', 'pubspec.yaml', 'dockerfile'}
_EXCLUDED_PARTS = {'.git', 'node_modules', 'vendor', 'dist', 'build',
                   '__pycache__', '.venv', 'venv'}


def readable_path(path: str) -> bool:
    """Select text relevant to inspection; never request secrets or binaries."""
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or '\\' in path:
        return False
    parts = {part.lower() for part in p.parts}
    name = p.name.lower()
    if parts & _EXCLUDED_PARTS or name.startswith('.env'):
        return False
    if any(word in name for word in ('credential', 'private_key', 'secret')):
        return False
    return (p.suffix.lower() in CODE_SUFFIXES | {'.md', '.rst', '.toml', '.yaml', '.yml', '.json', '.csproj', '.cfg', '.ini'}
            or name in MANIFESTS or name.startswith(('licence', 'license', 'copying', 'notice')))


def implementation_path(path: str) -> bool:
    p = PurePosixPath(path)
    return (p.suffix.lower() in CODE_SUFFIXES and p.name.lower() not in MANIFESTS
            and not any(part.lower() in {'test', 'tests', '__tests__', 'examples', 'docs'}
                        for part in p.parts)
            and not p.name.lower().startswith(('test_', 'conftest.'))
            and not re.search(r'\.(test|spec)\.', p.name.lower()))


def licence_path(path: str) -> bool:
    return PurePosixPath(path).name.lower().startswith(('licence', 'license', 'copying', 'notice'))


class GitHubSource:
    def __init__(self, token: str | None = None, session=None, cache_dir=None,
                 limits: SourceLimits | None = None):
        self.token = token if token is not None else os.getenv('GITHUB_TOKEN', '')
        self.session = session or requests.Session()
        self.limits = limits or SourceLimits()
        self.cache_dir = Path(cache_dir or os.getenv('ORACLE_SOURCE_CACHE_DIR', '.oracle-source-cache'))

    def _get(self, endpoint: str) -> dict:
        headers = {'Accept': 'application/vnd.github+json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        try:
            with self.session.get('https://api.github.com' + endpoint, headers=headers,
                                  timeout=(10, 30), stream=True, allow_redirects=False) as response:
                if response.status_code != 200:
                    raise SourceError(f'GitHub returned HTTP {response.status_code}; source unavailable.')
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > 8_000_000:
                        raise SourceError('GitHub response exceeded the inspection size limit.')
                    chunks.append(chunk)
                value = json.loads(b''.join(chunks))
                if not isinstance(value, dict):
                    raise SourceError('Unexpected GitHub response shape.')
                return value
        except (requests.RequestException, ValueError) as exc:
            raise SourceError(f'GitHub source request failed ({type(exc).__name__}).') from None

    def snapshot(self, name: str) -> dict:
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', name)
                or any(part in ('.', '..') for part in name.split('/'))):
            raise SourceError('Invalid GitHub repository name.')
        repo = self._get(f'/repos/{name}')
        branch = repo.get('default_branch')
        if not isinstance(branch, str) or not branch:
            raise SourceError('Repository has no accessible default branch.')
        commit = self._get(f'/repos/{name}/commits/{quote(branch, safe="")}')
        revision = commit.get('sha', '')
        tree_sha = commit.get('commit', {}).get('tree', {}).get('sha', '')
        if not re.fullmatch(r'[0-9a-f]{40}', revision) or not re.fullmatch(r'[0-9a-f]{40}', tree_sha):
            raise SourceError('GitHub did not provide a pinned commit and tree.')
        tree = self._get(f'/repos/{name}/git/trees/{tree_sha}?recursive=1')
        entries = {x['path']: x for x in tree.get('tree', [])
                   if isinstance(x, dict) and x.get('type') == 'blob'
                   and x.get('mode') in ('100644', '100755')
                   and isinstance(x.get('path'), str) and readable_path(x['path'])}
        limitations = []
        if tree.get('truncated'):
            limitations.append('GitHub returned a truncated tree; some files could not be considered.')
        return {'repository': name, 'revision': revision, 'entries': entries,
                'files': {}, 'limitations': limitations,
                'tree_truncated': bool(tree.get('truncated')),
                'metadata': {'licence': (repo.get('license') or {}).get('spdx_id'),
                             'last_push': repo.get('pushed_at'), 'archived': repo.get('archived')}}

    def path_catalogue(self, snapshot: dict) -> list[str]:
        paths = sorted(snapshot['entries'], key=lambda p: (p.count('/'), p))
        if len(paths) > self.limits.max_tree_paths:
            snapshot['limitations'].append('The path-selection list was limited; source coverage is incomplete.')
        return paths[:self.limits.max_tree_paths]

    def _blob(self, name: str, entry: dict) -> bytes:
        sha = entry.get('sha', '')
        if not re.fullmatch(r'[0-9a-f]{40}', sha):
            raise SourceError('Invalid source blob identifier.')
        if not isinstance(entry.get('size'), int) or entry['size'] > self.limits.max_file_bytes:
            raise SourceError('File exceeds the inspection size limit or has no declared size.')
        cache = self.cache_dir / (sha + '.blob')
        raw = None
        try:
            if cache.is_file() and cache.stat().st_size <= self.limits.max_file_bytes:
                raw = cache.read_bytes()
        except OSError:
            pass
        def valid(data):
            return (data is not None and len(data) <= self.limits.max_file_bytes
                    and hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == sha)
        if not valid(raw):
            value = self._get(f'/repos/{name}/git/blobs/{sha}')
            if value.get('encoding') != 'base64':
                raise SourceError('Source blob was not available as bounded text.')
            try:
                raw = base64.b64decode(''.join(value.get('content', '').split()), validate=True)
            except (ValueError, TypeError):
                raise SourceError('Invalid source blob encoding.') from None
            if not valid(raw):
                raise SourceError('Source content did not match its pinned blob identifier.')
            temporary = None
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(raw)
                os.replace(temporary, cache)
            except OSError:
                pass  # The cache is optional; retrieval already succeeded.
            finally:
                try:
                    if temporary and temporary.exists():
                        temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return raw

    def fetch(self, snapshot: dict, paths: list[str]) -> None:
        """Fetch exact tree members, keeping full-line text within the run limits."""
        for path in dict.fromkeys(paths):
            if path in snapshot['files']:
                continue
            if path not in snapshot['entries']:
                snapshot['limitations'].append(f'Requested path was unavailable: {path}')
                continue
            if len(snapshot['files']) >= self.limits.max_files:
                snapshot['limitations'].append('Source file-count limit reached.')
                break
            remaining = self.limits.max_chars - sum(len(f['text']) for f in snapshot['files'].values())
            if remaining <= 0:
                snapshot['limitations'].append('Source character limit reached.')
                break
            try:
                entry = snapshot['entries'][path]
                raw = self._blob(snapshot['repository'], entry)
                text = raw.decode('utf-8-sig').replace('\r\n', '\n')
                if '\0' in text:
                    raise SourceError('File appears to be binary.')
                limit = min(remaining, self.limits.max_file_chars)
                truncated = len(text) > limit
                excerpt = text[:limit]
                if truncated:
                    excerpt = excerpt.rsplit('\n', 1)[0] if '\n' in excerpt else ''
                snapshot['files'][path] = {'text': excerpt, 'blob_sha': entry['sha'],
                                           'truncated': truncated, 'line_count': len(excerpt.splitlines())}
                if truncated:
                    snapshot['limitations'].append(f'Only a prefix of {path} was inspected.')
            except (SourceError, UnicodeError) as exc:
                detail = str(exc) if isinstance(exc, SourceError) else 'File is not UTF-8 text.'
                snapshot['limitations'].append(f'{path}: {detail}')

    def follow_dependencies(self, snapshot: dict) -> None:
        """Follow a bounded set of static Python and relative JS/TS imports."""
        links = []
        for _ in range(2):
            links = local_dependencies(snapshot)
            missing = list(dict.fromkeys(x['to'] for x in links if x['to'] not in snapshot['files']))
            if not missing:
                break
            before = len(snapshot['files'])
            self.fetch(snapshot, missing[:3])
            if len(snapshot['files']) == before:
                break
        snapshot['dependency_links'] = local_dependencies(snapshot)
        missing = sorted({x['to'] for x in snapshot['dependency_links'] if x['to'] not in snapshot['files']})
        if missing:
            snapshot['limitations'].append('Internal dependencies remain uninspected: ' + ', '.join(missing))


def local_dependencies(snapshot: dict) -> list[dict]:
    """Find only import targets proven to be tree members; no code execution."""
    entries, links = snapshot['entries'], []
    for path, record in snapshot['files'].items():
        p = PurePosixPath(path)
        modules = []
        if p.suffix == '.py':
            try:
                tree = ast.parse(record['text'])
            except SyntaxError:
                continue  # Truncation is already disclosed in the snapshot.
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend((alias.name.replace('.', '/'), False) for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    root = '/'.join(p.parent.parts[:len(p.parent.parts) - node.level + 1]) if node.level else ''
                    module = (node.module or '').replace('.', '/')
                    stem = '/'.join(x for x in (root, module) if x)
                    modules.append((stem, bool(node.level)))
                    modules.extend(('/'.join(x for x in (stem, alias.name) if x), bool(node.level))
                                   for alias in node.names if alias.name != '*')
            targets = []
            for stem, relative in modules:
                for prefix in ('',) if relative else ('', 'src/'):
                    targets.extend((prefix + stem + '.py', prefix + stem + '/__init__.py'))
        elif p.suffix in {'.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs'}:
            specs = re.findall(r'''(?:from\s*|require\s*\(\s*|import\s*\(?\s*)["'](\.[^"']+)["']''', record['text'])
            targets = []
            for spec in specs:
                parts = list(p.parent.parts)
                for part in PurePosixPath(spec).parts:
                    if part == '..':
                        if parts:
                            parts.pop()
                    elif part != '.':
                        parts.append(part)
                stem = '/'.join(parts)
                targets.append(stem)
                base = stem[:-3] if stem.endswith('.js') else stem
                targets.extend(base + suffix for suffix in ('.ts', '.tsx', '.js', '.jsx', '/index.ts', '/index.js'))
        else:
            continue
        for target in dict.fromkeys(targets):
            if target in entries and target != path:
                links.append({'from': path, 'to': target})
    return links
