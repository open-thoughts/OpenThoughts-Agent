#!/usr/bin/env python3
"""
Shared utility functions for downloading datasets from various sources.

This module provides common functions for:
- Downloading from Zenodo
- Downloading from GitHub
- Caching downloaded datasets
- Loading from local files
"""

import hashlib
import os
import shutil
import sqlite3
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded datasets."""
    cache_dir = Path(os.environ.get("DCLM_CACHE_DIR", DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# =============================================================================
# Download Functions
# =============================================================================

def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> Path:
    """
    Download a file from a URL with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Download chunk size in bytes

    Returns:
        Path to downloaded file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if output_path.exists():
        print(f"File already exists: {output_path}")
        return output_path

    print(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                pbar.update(size)

    return output_path


def download_from_zenodo(
    record_id: str,
    filename: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Download a file from Zenodo.

    Args:
        record_id: Zenodo record ID (e.g., "14264519")
        filename: Name of the file to download
        output_dir: Directory to save the file (defaults to cache dir)

    Returns:
        Path to downloaded file
    """
    if output_dir is None:
        output_dir = get_cache_dir() / f"zenodo_{record_id}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://zenodo.org/records/{record_id}/files/{filename}"
    output_path = output_dir / filename

    return download_file(url, output_path)


def download_from_github(
    repo: str,
    output_dir: Optional[Path] = None,
    branch: str = "main",
    sparse_checkout: Optional[List[str]] = None
) -> Path:
    """
    Clone or download a GitHub repository.

    Args:
        repo: Repository in "owner/repo" format
        output_dir: Directory to clone into (defaults to cache dir)
        branch: Branch to checkout
        sparse_checkout: List of paths for sparse checkout (for large repos)

    Returns:
        Path to cloned repository
    """
    if output_dir is None:
        output_dir = get_cache_dir() / "github" / repo.replace("/", "_")

    output_dir = Path(output_dir)

    if output_dir.exists() and (output_dir / ".git").exists():
        print(f"Repository already exists: {output_dir}")
        # Pull latest changes
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=output_dir,
                capture_output=True,
                timeout=60
            )
        except Exception:
            pass
        return output_dir

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    repo_url = f"https://github.com/{repo}.git"
    print(f"Cloning: {repo_url}")

    if sparse_checkout:
        # Use sparse checkout for large repos
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(output_dir)],
            check=True,
            timeout=120
        )
        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"],
            cwd=output_dir,
            check=True
        )
        subprocess.run(
            ["git", "sparse-checkout", "set"] + sparse_checkout,
            cwd=output_dir,
            check=True
        )
        subprocess.run(
            ["git", "checkout", branch],
            cwd=output_dir,
            check=True
        )
    else:
        # Regular shallow clone
        subprocess.run(
            ["git", "clone", "--depth", "1", "-b", branch, repo_url, str(output_dir)],
            check=True,
            timeout=300
        )

    return output_dir


def download_github_archive(
    repo: str,
    output_dir: Optional[Path] = None,
    branch: str = "main"
) -> Path:
    """
    Download a GitHub repository as a zip archive (faster than git clone).

    Args:
        repo: Repository in "owner/repo" format
        output_dir: Directory to extract into (defaults to cache dir)
        branch: Branch to download

    Returns:
        Path to extracted repository
    """
    if output_dir is None:
        output_dir = get_cache_dir() / "github" / repo.replace("/", "_")

    output_dir = Path(output_dir)

    if output_dir.exists():
        print(f"Repository already exists: {output_dir}")
        return output_dir

    # Try main first, then master
    branches_to_try = [branch, "master"] if branch == "main" else [branch]

    zip_path = output_dir.parent / f"{repo.split('/')[-1]}.zip"
    downloaded = False

    for try_branch in branches_to_try:
        zip_url = f"https://github.com/{repo}/archive/refs/heads/{try_branch}.zip"
        try:
            download_file(zip_url, zip_path)
            branch = try_branch
            downloaded = True
            break
        except Exception as e:
            print(f"Failed to download from branch '{try_branch}': {e}")
            continue

    if not downloaded:
        raise ValueError(f"Could not download {repo} from any branch")

    # Extract
    print(f"Extracting to: {output_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir.parent)

    # Rename extracted folder
    extracted_name = f"{repo.split('/')[-1]}-{branch}"
    extracted_path = output_dir.parent / extracted_name
    if extracted_path.exists():
        shutil.move(str(extracted_path), str(output_dir))

    # Clean up zip
    zip_path.unlink()

    return output_dir


# =============================================================================
# Archive Extraction Functions
# =============================================================================

def extract_archive(archive_path: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Extract an archive (zip, tar.gz, tar.xz).

    Args:
        archive_path: Path to archive file
        output_dir: Directory to extract to (defaults to same directory as archive)

    Returns:
        Path to extracted directory
    """
    archive_path = Path(archive_path)

    if output_dir is None:
        output_dir = archive_path.parent / archive_path.stem.rstrip('.tar')

    output_dir = Path(output_dir)

    if output_dir.exists():
        print(f"Already extracted: {output_dir}")
        return output_dir

    print(f"Extracting: {archive_path}")

    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif archive_path.suffix in ('.gz', '.xz', '.bz2') or '.tar' in archive_path.name:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    return output_dir


# =============================================================================
# SQLite Loading Functions
# =============================================================================

def load_sqlite_table(
    db_path: Path,
    table_name: str,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
    where_clause: Optional[str] = None
) -> List[Dict]:
    """
    Load data from a SQLite database table.

    Args:
        db_path: Path to SQLite database file
        table_name: Name of table to query
        columns: List of columns to select (None for all)
        limit: Maximum number of rows to return
        where_clause: Optional WHERE clause (without 'WHERE' keyword)

    Returns:
        List of dictionaries (one per row)
    """
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build query
    col_str = ", ".join(columns) if columns else "*"
    query = f"SELECT {col_str} FROM {table_name}"

    if where_clause:
        query += f" WHERE {where_clause}"

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    rows = cursor.fetchall()

    # Convert to list of dicts
    result = [dict(row) for row in rows]

    conn.close()
    return result


def get_sqlite_tables(db_path: Path) -> List[str]:
    """Get list of tables in a SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


# =============================================================================
# Caching Functions
# =============================================================================

def cache_dataset(
    name: str,
    generator_func: Callable,
    *args,
    force_regenerate: bool = False,
    **kwargs
) -> Any:
    """
    Cache the result of a dataset generator function.

    Args:
        name: Unique name for the cached dataset
        generator_func: Function that generates the dataset
        force_regenerate: If True, regenerate even if cached
        *args, **kwargs: Arguments to pass to generator_func

    Returns:
        The generated/cached dataset
    """
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{name}.json"

    if cache_file.exists() and not force_regenerate:
        print(f"Loading cached dataset: {cache_file}")
        import json
        with open(cache_file, 'r') as f:
            return json.load(f)

    print(f"Generating dataset: {name}")
    result = generator_func(*args, **kwargs)

    # Cache if it's a list of dicts (common case)
    if isinstance(result, list):
        import json
        with open(cache_file, 'w') as f:
            json.dump(result, f)
        print(f"Cached to: {cache_file}")

    return result


# =============================================================================
# Local File Loading Functions
# =============================================================================

def load_json_file(file_path: Path) -> Union[List, Dict]:
    """Load data from a JSON file."""
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl_file(file_path: Path, limit: Optional[int] = None) -> List[Dict]:
    """Load data from a JSONL file."""
    import json
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            results.append(json.loads(line))
    return results


def load_csv_file(file_path: Path, limit: Optional[int] = None) -> List[Dict]:
    """Load data from a CSV file."""
    import csv
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            results.append(dict(row))
    return results


# =============================================================================
# Dataset-specific Loading Functions
# =============================================================================

def load_bugswarm_local(json_path: Path, limit: Optional[int] = None) -> List[Dict]:
    """
    Load BugSwarm data from a locally downloaded JSON file.

    To get this file:
    1. Go to https://www.bugswarm.org/dataset/
    2. Apply filters if desired
    3. Click "Export" to download artifacts.json

    Args:
        json_path: Path to the exported JSON file
        limit: Maximum number of samples to load

    Returns:
        List of BugSwarm artifact dictionaries
    """
    data = load_json_file(json_path)

    if isinstance(data, dict) and 'artifacts' in data:
        data = data['artifacts']

    samples = []
    for item in data[:limit] if limit else data:
        samples.append({
            "artifact_id": item.get("image_tag", item.get("_id", "")),
            "repo": item.get("repo", ""),
            "language": item.get("lang", item.get("language", "java")),
            "failed_job_id": str(item.get("failed_job", {}).get("job_id", "")),
            "passed_job_id": str(item.get("passed_job", {}).get("job_id", "")),
            "failing_code": "",  # Would need to extract from Docker image
            "passing_code": "",  # Would need to extract from Docker image
            "diff": "",
            "build_system": item.get("build_system", ""),
            "test_framework": item.get("test_framework", ""),
            "pr_num": item.get("pr_num", ""),
            "base_branch": item.get("base_branch", ""),
        })

    return samples


def load_crosscodeeval_local(data_dir: Path, language: str = "python", limit: Optional[int] = None) -> List[Dict]:
    """
    Load CrossCodeEval data from locally downloaded GitHub repo.

    To get this data:
    1. Clone https://github.com/amazon-science/cceval
    2. Extract crosscodeeval_data.tar.xz

    Args:
        data_dir: Path to extracted crosscodeeval_data directory
        language: Language to load (python, java, typescript, csharp)
        limit: Maximum number of samples to load

    Returns:
        List of CrossCodeEval sample dictionaries
    """
    data_dir = Path(data_dir)

    # Try various possible file locations
    possible_paths = [
        data_dir / language / "line_completion.jsonl",  # Main data file
        data_dir / language / "test.jsonl",
        data_dir / "data" / language / "line_completion.jsonl",
        data_dir / "data" / language / "test.jsonl",
        data_dir / f"{language}_test.jsonl",
        data_dir / f"{language}.jsonl",
    ]

    data_file = None
    for path in possible_paths:
        if path.exists():
            data_file = path
            break

    if data_file is None:
        # Search recursively for any jsonl file in language directory
        lang_dir = data_dir / language
        if not lang_dir.exists():
            lang_dir = data_dir / "data" / language

        if lang_dir.exists():
            for jsonl_file in lang_dir.glob("*.jsonl"):
                data_file = jsonl_file
                break

    if data_file is None:
        raise FileNotFoundError(f"Could not find CrossCodeEval data for {language} in {data_dir}")

    print(f"Loading from: {data_file}")
    return load_jsonl_file(data_file, limit=limit)


def load_manybugs_local(repo_dir: Path, limit: Optional[int] = None) -> List[Dict]:
    """
    Load ManyBugs data from locally cloned GitHub repo.

    To get this data:
    1. Clone https://github.com/squaresLab/ManyBugs

    Args:
        repo_dir: Path to cloned ManyBugs repository
        limit: Maximum number of samples to load

    Returns:
        List of ManyBugs bug dictionaries
    """
    import yaml

    repo_dir = Path(repo_dir)

    # Load bug-data.csv for metadata
    bug_csv = repo_dir / "bug-data.csv"

    if bug_csv.exists():
        bugs = load_csv_file(bug_csv, limit=limit)
        if bugs:
            return bugs

    # Try to parse bugzoo YAML files which contain bug definitions
    bugs = []
    for yml_file in repo_dir.rglob("*.bugzoo.yml"):
        if limit and len(bugs) >= limit:
            break

        try:
            with open(yml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not data or 'blueprints' not in data:
                continue

            # Extract project name from filename (e.g., php.bugzoo.yml -> php)
            project = yml_file.stem.replace('.bugzoo', '')

            for blueprint in data.get('blueprints', []):
                if limit and len(bugs) >= limit:
                    break

                # Skip base images (no scenario/arguments)
                args = blueprint.get('arguments', {})
                if not args or 'scenario' not in args:
                    continue

                scenario = args.get('scenario', '')
                bug_revision = args.get('bug_revision', '')
                fix_revision = args.get('fix_revision', '')

                bugs.append({
                    "bug_id": scenario,
                    "project": project,
                    "scenario": scenario,
                    "bug_revision": bug_revision,
                    "fix_revision": fix_revision,
                    "tag": blueprint.get('tag', ''),
                })

        except Exception as e:
            print(f"Error parsing {yml_file}: {e}")
            continue

    if bugs:
        return bugs

    # Fallback: scan scenarios directories
    scenarios_dir = repo_dir / "scenarios"

    if not scenarios_dir.exists():
        scenarios_dir = repo_dir

    for project_dir in scenarios_dir.iterdir():
        if not project_dir.is_dir():
            continue

        for bug_dir in project_dir.iterdir():
            if not bug_dir.is_dir():
                continue

            bugs.append({
                "bug_id": f"{project_dir.name}-{bug_dir.name}",
                "project": project_dir.name,
                "scenario": bug_dir.name,
            })

            if limit and len(bugs) >= limit:
                return bugs

    return bugs


def load_e2egit_from_sqlite(db_path: Path, limit: Optional[int] = None) -> List[Dict]:
    """
    Load E2EGit data from SQLite database.

    To get this data:
    1. Download from https://zenodo.org/records/14234731
    2. Extract E2EGit.db

    Args:
        db_path: Path to E2EGit.db SQLite file
        limit: Maximum number of samples to load

    Returns:
        List of E2E test sample dictionaries
    """
    # Get tables to find the right one
    tables = get_sqlite_tables(db_path)
    print(f"Available tables: {tables}")

    # Common table names in E2EGit
    possible_tables = ["gui_testing_test_details", "tests", "test_cases", "e2e_tests"]

    table_name = None
    for t in possible_tables:
        if t in tables:
            table_name = t
            break

    if table_name is None and tables:
        table_name = tables[0]

    if table_name is None:
        raise ValueError(f"No tables found in {db_path}")

    rows = load_sqlite_table(db_path, table_name, limit=limit)

    # Map to expected format
    samples = []
    for row in rows:
        samples.append({
            "test_code": row.get("test_code", row.get("code", row.get("content", ""))),
            "repo": row.get("repo", row.get("repository", row.get("repo_name", ""))),
            "file_path": row.get("file_path", row.get("path", "")),
            "framework": row.get("framework", row.get("test_framework", "selenium")),
            "language": row.get("language", "python"),
        })

    return samples


def load_ghactions_from_zenodo(data_dir: Path, limit: Optional[int] = None) -> List[Dict]:
    """
    Load GitHub Actions workflow data from Zenodo download.

    To get this data:
    1. Download from https://zenodo.org/records/10566003
    2. Extract workflows.tar.gz

    Args:
        data_dir: Path to extracted workflows directory
        limit: Maximum number of samples to load

    Returns:
        List of workflow dictionaries
    """
    data_dir = Path(data_dir)

    # Load metadata CSV if available
    csv_path = data_dir / "workflows.csv"
    if not csv_path.exists():
        csv_path = data_dir.parent / "workflows.csv"

    samples = []

    if csv_path.exists():
        import gzip
        csv_gz_path = data_dir.parent / "workflows.csv.gz"

        if csv_gz_path.exists():
            import gzip
            import csv
            with gzip.open(csv_gz_path, 'rt', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if limit and i >= limit:
                        break
                    samples.append(dict(row))
        else:
            samples = load_csv_file(csv_path, limit=limit)
    else:
        # Scan workflow files directly
        # First try .yml and .yaml files
        for yaml_file in data_dir.rglob("*.yml"):
            if limit and len(samples) >= limit:
                break

            try:
                content = yaml_file.read_text(encoding='utf-8')
                samples.append({
                    "workflow_content": content,
                    "workflow_path": str(yaml_file.relative_to(data_dir)),
                    "repo_name": yaml_file.parent.name,
                })
            except Exception:
                continue

        for yaml_file in data_dir.rglob("*.yaml"):
            if limit and len(samples) >= limit:
                break

            try:
                content = yaml_file.read_text(encoding='utf-8')
                samples.append({
                    "workflow_content": content,
                    "workflow_path": str(yaml_file.relative_to(data_dir)),
                    "repo_name": yaml_file.parent.name,
                })
            except Exception:
                continue

        # If no .yml/.yaml files found, try files without extension (Zenodo format)
        if not samples:
            for workflow_file in data_dir.iterdir():
                if limit and len(samples) >= limit:
                    break

                if not workflow_file.is_file():
                    continue

                try:
                    content = workflow_file.read_text(encoding='utf-8')
                    # Verify it looks like a YAML workflow file
                    if 'name:' in content or 'on:' in content or 'jobs:' in content:
                        samples.append({
                            "workflow_content": content,
                            "workflow_path": workflow_file.name,
                            "repo_name": workflow_file.name,
                        })
                except Exception:
                    continue

    return samples


def load_pymethods2test_from_zenodo(data_dir: Path, limit: Optional[int] = None) -> List[Dict]:
    """
    Load PyMethods2Test data from Zenodo download.

    To get this data:
    1. Download from https://zenodo.org/records/14264519
    2. Extract focal-data.zip

    Args:
        data_dir: Path to extracted focal-data directory
        limit: Maximum number of samples to load

    Returns:
        List of focal-test pair dictionaries
    """
    data_dir = Path(data_dir)

    samples = []

    # Look for JSONL files
    for jsonl_file in data_dir.rglob("*.jsonl"):
        if limit and len(samples) >= limit:
            break

        file_samples = load_jsonl_file(jsonl_file, limit=limit - len(samples) if limit else None)
        samples.extend(file_samples)

    # Look for JSON files if no JSONL found
    if not samples:
        for json_file in data_dir.rglob("*.json"):
            if limit and len(samples) >= limit:
                break

            data = load_json_file(json_file)
            if isinstance(data, list):
                samples.extend(data[:limit - len(samples) if limit else None])

    return samples


# =============================================================================
# Convenience Functions for Downloading Full Datasets
# =============================================================================

def download_crosscodeeval() -> Path:
    """Download CrossCodeEval from GitHub and extract data."""
    repo_dir = download_github_archive("amazon-science/cceval")

    # Look for tar.xz files in various locations
    possible_tars = [
        repo_dir / "crosscodeeval_data.tar.xz",
        repo_dir / "data" / "crosscodeeval_data.tar.xz",
    ]

    for data_tar in possible_tars:
        if data_tar.exists():
            print(f"Extracting {data_tar}...")
            extracted = extract_archive(data_tar, data_tar.parent / "crosscodeeval_data")
            return extracted

    # If no tar.xz, check if data already exists
    for check_dir in [repo_dir / "data", repo_dir]:
        if (check_dir / "python").exists() or (check_dir / "crosscodeeval_data").exists():
            return check_dir

    return repo_dir


def download_manybugs() -> Path:
    """Download ManyBugs from GitHub."""
    return download_github_archive("squaresLab/ManyBugs")


def download_pymethods2test() -> Path:
    """Download PyMethods2Test from Zenodo."""
    cache_dir = get_cache_dir() / "zenodo_14264519"
    cache_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = cache_dir / "focal-data"
    if extracted_dir.exists():
        print(f"Already downloaded: {extracted_dir}")
        return extracted_dir

    zip_path = download_from_zenodo("14264519", "focal-data.zip")
    return extract_archive(zip_path, extracted_dir)


def download_e2egit() -> Path:
    """Download E2EGit from Zenodo."""
    db_path = download_from_zenodo("14234731", "E2EGit.db")
    return db_path


def download_ghactions() -> Path:
    """Download GitHub Actions Workflows from Zenodo."""
    cache_dir = get_cache_dir() / "zenodo_10566003"
    cache_dir.mkdir(parents=True, exist_ok=True)

    workflows_dir = cache_dir / "workflows"
    if workflows_dir.exists() and any(workflows_dir.rglob("*.yml")):
        print(f"Already downloaded: {workflows_dir}")
        return workflows_dir

    tar_path = download_from_zenodo("10566003", "workflows.tar.gz")
    extracted = extract_archive(tar_path, cache_dir / "workflows_extracted")

    # The tar may have a nested structure, find the actual workflows directory
    for subdir in extracted.rglob("*"):
        if subdir.is_dir() and any(subdir.glob("*.yml")):
            # Move or symlink to workflows_dir
            if not workflows_dir.exists():
                shutil.move(str(subdir), str(workflows_dir))
            return workflows_dir

    return extracted


def download_bugswarm(output_dir: Optional[Path] = None) -> Path:
    """
    Download BugSwarm artifacts data.

    BugSwarm requires either:
    1. Using their REST API with authentication
    2. Manual download from https://www.bugswarm.org/dataset/

    This function attempts to use the bugswarm-common package if installed,
    otherwise provides instructions for manual download.

    Args:
        output_dir: Directory to save the data

    Returns:
        Path to the downloaded data
    """
    if output_dir is None:
        output_dir = get_cache_dir() / "bugswarm"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_file = output_dir / "artifacts.json"

    if artifacts_file.exists():
        print(f"Already downloaded: {artifacts_file}")
        return artifacts_file

    # Try to use bugswarm-common package
    try:
        from bugswarm.common.rest_api.database_api import DatabaseAPI

        print("Using bugswarm-common package to fetch artifacts...")
        api = DatabaseAPI(token=os.environ.get("BUGSWARM_TOKEN", ""))

        # Fetch all artifacts
        artifacts = api.list_artifacts()

        import json
        with open(artifacts_file, 'w') as f:
            json.dump(artifacts, f, indent=2)

        print(f"Downloaded {len(artifacts)} artifacts to {artifacts_file}")
        return artifacts_file

    except ImportError:
        pass
    except Exception as e:
        print(f"BugSwarm API error: {e}")

    # Try to download pre-exported data from alternative sources
    alternative_urls = [
        "https://raw.githubusercontent.com/BugSwarm/bugswarm/master/database/artifacts.json",
    ]

    for url in alternative_urls:
        try:
            print(f"Trying alternative source: {url}")
            download_file(url, artifacts_file)
            return artifacts_file
        except Exception as e:
            print(f"Failed: {e}")
            continue

    # Provide manual instructions
    print("")
    print("=" * 60)
    print("BugSwarm requires manual download:")
    print("=" * 60)
    print("")
    print("Option 1: Download from web interface")
    print("  1. Go to https://www.bugswarm.org/dataset/")
    print("  2. Apply any filters you want")
    print("  3. Click 'Export' to download artifacts.json")
    print(f"  4. Save to: {artifacts_file}")
    print("")
    print("Option 2: Install bugswarm-common package")
    print("  pip install bugswarm-common")
    print("  export BUGSWARM_TOKEN=<your_token>")
    print("  # Get token from https://www.bugswarm.org/")
    print("")
    print("Option 3: Use BugSwarm Docker images directly")
    print("  docker pull bugswarm/images:<artifact_id>")
    print("")

    raise ValueError("BugSwarm requires manual download. See instructions above.")


def download_all_datasets(datasets: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Download all datasets (or specified subset).

    Args:
        datasets: List of dataset names to download. If None, downloads all.
                  Options: bugswarm, manybugs, e2egit, ghactions, pymethods2test, crosscodeeval

    Returns:
        Dictionary mapping dataset names to their paths
    """
    all_datasets = {
        "crosscodeeval": download_crosscodeeval,
        "manybugs": download_manybugs,
        "pymethods2test": download_pymethods2test,
        "e2egit": download_e2egit,
        "ghactions": download_ghactions,
        "bugswarm": download_bugswarm,
    }

    if datasets is None:
        datasets = list(all_datasets.keys())

    results = {}

    for name in datasets:
        if name not in all_datasets:
            print(f"Unknown dataset: {name}")
            continue

        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"{'='*60}")

        try:
            path = all_datasets[name]()
            results[name] = path
            print(f"✓ {name}: {path}")
        except Exception as e:
            print(f"✗ {name}: {e}")
            results[name] = None

    return results


if __name__ == "__main__":
    # Test the utilities
    print("Cache directory:", get_cache_dir())

    # Example: Download CrossCodeEval
    # path = download_crosscodeeval()
    # print(f"Downloaded to: {path}")
