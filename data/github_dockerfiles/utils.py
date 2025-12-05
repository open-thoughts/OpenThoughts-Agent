#!/usr/bin/env python3
"""
GitHub toolkit: prompt -> Dockerfile from GitHub repositories
Searches GitHub for Dockerfiles using keywords and returns the top result
"""

import requests
import aiohttp
import asyncio
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import base64
import re
import json
import hashlib
import os
import pickle
from pathlib import Path

# Suppress requests library logging
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from bespokelabs import curator
KEYBERT_AVAILABLE = False
_keybert_model = None

from data.gcs_cache import gcs_cache

class KeywordExtractor(curator.LLM):
    def prompt(self, prompt: str) -> str:
        return f"""
        Extract the single most important keyword or short phrase (1-3 words) from the following prompt that would be best for searching GitHub repositories for Dockerfiles:
        
        Prompt: {prompt}
        
        Focus on:
        - Programming languages (python, node, java, etc.)
        - Frameworks (django, react, nginx, etc.) 
        - Technologies (database, redis, elasticsearch, etc.)
        - Services (web server, api, microservice, etc.)
        
        Avoid generic words like: application, system, environment, container, docker, setup, install, create
        
        Return only the keyword/phrase, nothing else.
        """

    def parse(self, prompt: str, response: str):
        # Clean the response to get just the keyword
        keyword = response.strip().strip('"').strip("'").lower()
        return {"prompt": prompt, "keyword": keyword}

@dataclass
class GitHubDockerfile:
    repository: str
    path: str
    content: str
    relevance_score: float = 0.0
    stars: int = 0
    description: str = ""

def prompt_to_dockerfile(prompt: str) -> str:
    """Main function: prompt -> Dockerfile content from GitHub search"""
    keywords = extract_keywords(prompt)
    # Search GitHub for Dockerfiles based on keywords
    dockerfile = search_github_dockerfiles_for_keywords(keywords)
    
    if dockerfile:
        return dockerfile.content
    else:
        # Fallback to a basic Ubuntu Dockerfile if no results found
        return generate_basic_dockerfile(keywords)

def prompt_to_dockerfile_with_keyword(prompt: str, keyword: str) -> str:
    """Main function with pre-extracted keyword: prompt -> Dockerfile content from GitHub search"""
    # Search GitHub for Dockerfiles based on the provided keyword
    dockerfile = search_github_dockerfiles_for_keywords([keyword])
    
    if dockerfile:
        return dockerfile.content
    else:
        # Fallback to a basic Ubuntu Dockerfile if no results found
        return generate_basic_dockerfile([keyword])

def search_github_dockerfiles_for_keywords(keywords: List[str]) -> Optional[GitHubDockerfile]:
    """Search GitHub for Dockerfiles based on keywords and return the top result"""
    
    all_results = []
    
    # Search GitHub for just the first keyword to reduce API usage
    if keywords:
        keyword = keywords[0]  # Only use the first/best keyword
        results = search_github_dockerfiles(keyword)
        print(f"Found {len(results)} Dockerfiles for keyword '{keyword}'")
        all_results.extend(results)
    
    if not all_results:
        print("No Dockerfiles found on GitHub for the given keywords")
        return None
    
    # Sort by relevance score and star count
    all_results.sort(key=lambda x: (x.relevance_score, x.stars), reverse=True)
    
    top_result = all_results[0]
    print(f"Selected Dockerfile from {top_result.repository} (stars: {top_result.stars}, score: {top_result.relevance_score:.1f})")
    
    return top_result

# Persistent cache with disk storage
CACHE_DIR = Path.home() / '.cache' / 'dockerfile_scraper'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / 'dockerfile_cache.pkl'

# Load persistent cache
_dockerfile_cache = {}
if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, 'rb') as f:
            _dockerfile_cache = pickle.load(f)
        print(f"Loaded {len(_dockerfile_cache)} cached dockerfiles from disk")
    except:
        _dockerfile_cache = {}

def _save_cache():
    """Save cache to disk"""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(_dockerfile_cache, f)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

def _get_cache_key(keyword: str) -> str:
    """Generate cache key for keyword"""
    return hashlib.md5(keyword.lower().encode()).hexdigest()

# Token rotation setup
def get_all_github_tokens() -> List[str]:
    """Get all available GitHub tokens for rotation"""
    tokens = []
    
    # Primary tokens from env vars
    for var in ['GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_ACCESS_TOKEN']:
        token = os.getenv(var)
        if token and token not in tokens:
            tokens.append(token)
    
    # Additional numbered tokens for rotation
    for i in range(1, 10):  # Check GITHUB_TOKEN_1 through GITHUB_TOKEN_9
        token = os.getenv(f'GITHUB_TOKEN_{i}')
        if token and token not in tokens:
            tokens.append(token)
    
    # GitHub CLI config
    try:
        gh_config_path = Path.home() / '.config' / 'gh' / 'hosts.yml'
        if gh_config_path.exists():
            with open(gh_config_path, 'r') as f:
                content = f.read()
                import re
                match = re.search(r'oauth_token:\s*(\S+)', content)
                if match and match.group(1) not in tokens:
                    tokens.append(match.group(1))
    except:
        pass
    
    return tokens

# Global token rotation state
_github_tokens = get_all_github_tokens()
_current_token_index = 0
_token_usage_count = {}

def get_next_github_token() -> Optional[str]:
    """Get next token in rotation"""
    global _current_token_index, _token_usage_count
    
    if not _github_tokens:
        return None
    
    # Rotate to next token every 100 requests
    current_token = _github_tokens[_current_token_index]
    _token_usage_count[current_token] = _token_usage_count.get(current_token, 0) + 1
    
    if _token_usage_count[current_token] >= 100:
        _current_token_index = (_current_token_index + 1) % len(_github_tokens)
        print(f"Rotating to token {_current_token_index + 1}/{len(_github_tokens)}")
    
    return current_token

# Keyword deduplication and smart batching
def deduplicate_and_batch_keywords(keywords: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """Deduplicate keywords and create mapping for batch processing"""
    unique_keywords = []
    keyword_to_indices = {}
    
    for i, keyword in enumerate(keywords):
        normalized = keyword.lower().strip()
        if normalized not in keyword_to_indices:
            unique_keywords.append(keyword)
            keyword_to_indices[normalized] = [i]
        else:
            keyword_to_indices[normalized].append(i)
    
    print(f"Deduplicated {len(keywords)} keywords to {len(unique_keywords)} unique ({len(keywords) - len(unique_keywords)} duplicates)")
    return unique_keywords, keyword_to_indices

async def search_github_dockerfiles_async(session: aiohttp.ClientSession, keyword: str, token_override: str = None) -> List[GitHubDockerfile]:
    """Async search GitHub for Dockerfiles using the GitHub API with caching"""
    
    # Check cache first
    cache_key = _get_cache_key(keyword)
    if cache_key in _dockerfile_cache:
        print(f"Using cached results for '{keyword}'")
        return _dockerfile_cache[cache_key]
    
    # GitHub API search for Dockerfiles
    url = "https://api.github.com/search/code"
    
    # Clean and encode the keyword to avoid query parsing errors
    clean_keyword = keyword.strip().replace('"', '').replace("'", "")
    # Escape special characters that might cause parsing issues
    clean_keyword = re.sub(r'[^\w\s\-\+]', '', clean_keyword)
    
    params = {
        'q': f'{clean_keyword} filename:Dockerfile',
        'sort': 'stars',  # Sort by stars for better results
        'order': 'desc',
        'per_page': 5  # Reduce to 5 for faster response
    }
    
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'DockerfileSearchTool/1.0'
    } 
    
    # Add GitHub token with rotation support
    github_token = token_override or get_next_github_token()
    if github_token:
        headers['Authorization'] = f'token {github_token}'
    
    max_retries = 2  # Reduce retries for speed
    base_delay = 10  # Reduce delay for speed
    
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params, headers=headers) as response:
                # Handle rate limiting with shorter backoff
                if response.status == 403:
                    response_text = await response.text()
                    if 'rate limit' in response_text.lower():
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"GitHub API rate limit exceeded for '{keyword}'. Retrying in {delay} seconds...")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            print(f"Warning: GitHub API rate limit exceeded for keyword '{keyword}'")
                            _dockerfile_cache[cache_key] = []  # Cache empty result
                            return []
                
                if response.status != 200:
                    print(f"Warning: GitHub API request failed for '{keyword}' (status: {response.status})")
                    _dockerfile_cache[cache_key] = []
                    return []
                
                data = await response.json()
                
                # Process only first 3 results for speed, and do it concurrently
                items = data.get('items', [])[:3]
                if not items:
                    _dockerfile_cache[cache_key] = []
                    return []
                
                # Process results concurrently
                tasks = [process_github_result_async(session, item, keyword) for item in items]
                dockerfiles = []
                
                for task in asyncio.as_completed(tasks):
                    dockerfile = await task
                    if dockerfile:
                        dockerfiles.append(dockerfile)
                        # Return first good result immediately for speed
                        _dockerfile_cache[cache_key] = dockerfiles
                        _save_cache()  # Save to disk
                        return dockerfiles
                
                _dockerfile_cache[cache_key] = dockerfiles
                _save_cache()  # Save to disk
                return dockerfiles
                
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay
                print(f"Error searching GitHub for '{keyword}': {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                continue
            else:
                print(f"Error searching GitHub for '{keyword}' after {max_retries} attempts: {e}")
                _dockerfile_cache[cache_key] = []
                _save_cache()  # Save to disk
                return []

def search_github_dockerfiles(keyword: str) -> List[GitHubDockerfile]:
    """Synchronous wrapper for async search (for backward compatibility)"""
    async def _search():
        async with aiohttp.ClientSession() as session:
            return await search_github_dockerfiles_async(session, keyword)
    
    return asyncio.run(_search())

async def process_github_result_async(session: aiohttp.ClientSession, item: Dict, keyword: str) -> Optional[GitHubDockerfile]:
    """Async process a single GitHub search result and fetch the Dockerfile content"""
    
    try:
        repo_name = item['repository']['full_name']
        file_path = item['path']
        download_url = item.get('download_url')
        
        # Use repository data from search results if available to avoid extra API call
        repo_data = item.get('repository', {})
        stars = repo_data.get('stargazers_count', 0)
        description = repo_data.get('description', '')
        
        # Skip if the repository has very few stars (likely not a good example)
        if stars < 1:
            return None
        
        # Fetch the Dockerfile content
        content = await fetch_dockerfile_content_async(session, download_url, repo_name, file_path)
        if not content:
            return None
        
        # Skip if Dockerfile is too simple or just a base image
        if is_dockerfile_too_simple(content):
            return None
        
        # Calculate relevance score
        score = calculate_dockerfile_relevance(keyword, repo_name, description, content, stars)
        
        return GitHubDockerfile(
            repository=repo_name,
            path=file_path,
            content=content,
            relevance_score=score,
            stars=stars,
            description=description
        )
        
    except Exception as e:
        print(f"Error processing GitHub result: {e}")
        return None

def process_github_result(item: Dict, keyword: str) -> Optional[GitHubDockerfile]:
    """Synchronous wrapper (for backward compatibility)"""
    async def _process():
        async with aiohttp.ClientSession() as session:
            return await process_github_result_async(session, item, keyword)
    
    return asyncio.run(_process())

def get_repository_info(repo_name: str) -> Optional[Dict]:
    """Get repository information including stars and description"""
    
    url = f"https://api.github.com/repos/{repo_name}"
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'DockerfileSearchTool/1.0'
    }
    
    github_token = get_github_token()
    if github_token:
        headers['Authorization'] = f'token {github_token}'
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        print(f"Error fetching repo info for {repo_name}: {e}")
        return None

async def fetch_dockerfile_content_async(session: aiohttp.ClientSession, download_url: str, repo_name: str = None, file_path: str = None) -> Optional[str]:
    """Async fetch the actual Dockerfile content"""
    
    # Try download_url first if available
    if download_url:
        try:
            async with session.get(download_url) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            print(f"Error fetching Dockerfile content from download_url: {e}")
    
    # Fallback: construct raw GitHub URL if repo_name and file_path are provided
    if repo_name and file_path:
        try:
            headers = {'User-Agent': 'DockerfileSearchTool/1.0'}
            github_token = get_github_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            # Try main branch first
            raw_url = f"https://raw.githubusercontent.com/{repo_name}/main/{file_path}"
            async with session.get(raw_url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
            
            # Try 'master' branch if 'main' doesn't work
            raw_url = f"https://raw.githubusercontent.com/{repo_name}/master/{file_path}"
            async with session.get(raw_url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
                
        except Exception as e:
            print(f"Error fetching Dockerfile content from raw URL: {e}")
    
    return None

def fetch_dockerfile_content(download_url: str, repo_name: str = None, file_path: str = None) -> Optional[str]:
    """Synchronous wrapper (for backward compatibility)"""
    async def _fetch():
        async with aiohttp.ClientSession() as session:
            return await fetch_dockerfile_content_async(session, download_url, repo_name, file_path)
    
    return asyncio.run(_fetch())

def is_dockerfile_too_simple(content: str) -> bool:
    """Check if Dockerfile is too simple to be useful"""
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Filter out comments
    command_lines = [line for line in lines if not line.startswith('#')]
    
    # Too simple if it has very few commands
    if len(command_lines) < 3:
        return True
    
    # Too simple if it's just FROM + CMD/ENTRYPOINT
    commands = set()
    for line in command_lines:
        cmd = line.split()[0].upper() if line.split() else ''
        commands.add(cmd)
    
    essential_commands = {'FROM', 'CMD', 'ENTRYPOINT'}
    if commands.issubset(essential_commands):
        return True
    
    return False

def calculate_dockerfile_relevance(keyword: str, repo_name: str, description: str, content: str, stars: int) -> float:
    """Calculate how relevant a Dockerfile is to the keyword"""
    
    score = 0.0
    kw_lower = keyword.lower()
    repo_lower = repo_name.lower() if repo_name else ''
    desc_lower = description.lower() if description else ''
    content_lower = content.lower() if content else ''
    
    # Keyword in repository name (high priority)
    if kw_lower in repo_lower:
        score += 50
    
    # Keyword in description
    if kw_lower in desc_lower:
        score += 30
    
    # Keyword in Dockerfile content
    if kw_lower in content_lower:
        score += 20
    
    # Star count boost (logarithmic to avoid overwhelming other factors)
    star_score = min(30, stars / 100)  # Max 30 points from stars
    score += star_score
    
    # Prefer repositories that seem actively maintained
    if stars > 50:
        score += 10
    
    # Prefer official or well-known repositories
    if any(org in repo_lower for org in ['docker-library', 'docker', 'official']):
        score += 25
    
    return score

def get_github_token() -> Optional[str]:
    """Get GitHub token from environment or config (optional)"""
    import os
    
    # Try to get from environment variable
    token = os.getenv('GITHUB_TOKEN') or os.getenv('GH_TOKEN')
    
    if not token:
        # Try to read from GitHub CLI config
        try:
            gh_config_path = os.path.expanduser('~/.config/gh/hosts.yml')
            if os.path.exists(gh_config_path):
                with open(gh_config_path, 'r') as f:
                    content = f.read()
                    # Simple parsing for oauth_token
                    import re
                    match = re.search(r'oauth_token:\s*(\S+)', content)
                    if match:
                        token = match.group(1)
        except Exception:
            pass
    
    return token

def generate_basic_dockerfile(keywords: List[str]) -> str:
    """Generate a basic Dockerfile when no GitHub results are found"""
    
    # Determine base image based on keywords
    base_image = "ubuntu:latest"
    packages = ["curl", "wget", "git"]
    
    for keyword in keywords:
        kw_lower = keyword.lower()
        if any(lang in kw_lower for lang in ['python', 'django', 'flask', 'fastapi']):
            base_image = "python:3.11-slim"
            packages = ["git", "curl"]
            break
        elif any(lang in kw_lower for lang in ['node', 'javascript', 'npm', 'yarn']):
            base_image = "node:18-slim"
            packages = ["git", "curl"]
            break
        elif any(lang in kw_lower for lang in ['nginx', 'apache', 'web server']):
            base_image = "nginx:alpine"
            packages = []
            break
    
    lines = [
        f"FROM {base_image}",
        "",
        "WORKDIR /app",
        ""
    ]
    
    if packages and base_image.startswith("ubuntu"):
        lines.extend([
            "RUN apt-get update && apt-get install -y \\",
            *[f"    {pkg} \\" for pkg in packages],
            "    && rm -rf /var/lib/apt/lists/*",
            ""
        ])
    
    lines.append("CMD [\"/bin/bash\"]")
    lines.append("ENTRYPOINT [\"sleep\", \"infinity\"]")
    return '\n'.join(lines)

def generate_enhanced_dockerfile(keywords: List[str]) -> str:
    """Generate a comprehensive Dockerfile with better fallback templates"""
    
    # Enhanced template matching
    for keyword in keywords:
        kw_lower = keyword.lower()
        
        # Python ecosystem
        if any(lang in kw_lower for lang in ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy']):
            return _get_python_dockerfile(kw_lower)
        
        # Node.js ecosystem  
        elif any(lang in kw_lower for lang in ['node', 'javascript', 'npm', 'yarn', 'react', 'vue', 'express']):
            return _get_nodejs_dockerfile(kw_lower)
        
        # Web servers
        elif any(lang in kw_lower for lang in ['nginx', 'apache', 'web server', 'httpd']):
            return _get_webserver_dockerfile(kw_lower)
        
        # Java ecosystem
        elif any(lang in kw_lower for lang in ['java', 'maven', 'gradle', 'spring', 'tomcat']):
            return _get_java_dockerfile(kw_lower)
        
        # Go
        elif 'go' in kw_lower or 'golang' in kw_lower:
            return _get_go_dockerfile()
        
        # Databases
        elif any(db in kw_lower for db in ['postgres', 'mysql', 'mongodb', 'redis']):
            return _get_database_dockerfile(kw_lower)
        
        # DevOps tools
        elif any(tool in kw_lower for tool in ['docker', 'kubernetes', 'jenkins', 'ansible']):
            return _get_devops_dockerfile(kw_lower)
    
    # Default Ubuntu-based dockerfile
    return _get_ubuntu_dockerfile()

def _get_python_dockerfile(keyword: str) -> str:
    """Enhanced Python Dockerfile template"""
    if 'django' in keyword:
        return """FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir django gunicorn

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD [\"python\", \"manage.py\", \"runserver\", \"0.0.0.0:8000\"]"""
    
    elif 'flask' in keyword:
        return """FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flask gunicorn

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD [\"gunicorn\", \"--bind\", \"0.0.0.0:5000\", \"app:app\"]"""
    
    elif any(ds in keyword for ds in ['pandas', 'numpy', 'jupyter', 'data']):
        return """FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy matplotlib seaborn jupyter

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

CMD [\"jupyter\", \"notebook\", \"--ip=0.0.0.0\", \"--allow-root\"]"""
    
    else:
        return """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [\"python\", \"app.py\"]"""

def _get_nodejs_dockerfile(keyword: str) -> str:
    """Enhanced Node.js Dockerfile template"""
    if 'react' in keyword or 'vue' in keyword:
        return """FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

RUN npm run build

EXPOSE 3000

CMD [\"npm\", \"start\"]"""
    
    else:
        return """FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

CMD [\"node\", \"server.js\"]"""

def _get_webserver_dockerfile(keyword: str) -> str:
    """Web server Dockerfile template"""
    if 'apache' in keyword:
        return """FROM httpd:2.4-alpine

COPY ./public-html/ /usr/local/apache2/htdocs/

EXPOSE 80

CMD [\"httpd-foreground\"]"""
    
    else:  # nginx
        return """FROM nginx:alpine

COPY ./dist/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD [\"nginx\", \"-g\", \"daemon off;\"]"""

def _get_java_dockerfile(keyword: str) -> str:
    """Java Dockerfile template"""
    return """FROM openjdk:17-jre-slim

WORKDIR /app

COPY target/*.jar app.jar

EXPOSE 8080

CMD [\"java\", \"-jar\", \"app.jar\"]"""

def _get_go_dockerfile() -> str:
    """Go Dockerfile template"""
    return """FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o main .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/main .

EXPOSE 8080

CMD [\"./main\"]"""

def _get_database_dockerfile(keyword: str) -> str:
    """Database Dockerfile template"""
    if 'postgres' in keyword:
        return """FROM postgres:15-alpine

ENV POSTGRES_DB=mydb
ENV POSTGRES_USER=user
ENV POSTGRES_PASSWORD=password

COPY init.sql /docker-entrypoint-initdb.d/

EXPOSE 5432"""
    
    elif 'mysql' in keyword:
        return """FROM mysql:8.0

ENV MYSQL_ROOT_PASSWORD=rootpassword
ENV MYSQL_DATABASE=mydb
ENV MYSQL_USER=user
ENV MYSQL_PASSWORD=password

COPY init.sql /docker-entrypoint-initdb.d/

EXPOSE 3306"""
    
    elif 'redis' in keyword:
        return """FROM redis:7-alpine

COPY redis.conf /usr/local/etc/redis/redis.conf

EXPOSE 6379

CMD [\"redis-server\", \"/usr/local/etc/redis/redis.conf\"]"""
    
    else:  # mongodb
        return """FROM mongo:7

ENV MONGO_INITDB_ROOT_USERNAME=admin
ENV MONGO_INITDB_ROOT_PASSWORD=password
ENV MONGO_INITDB_DATABASE=mydb

COPY init.js /docker-entrypoint-initdb.d/

EXPOSE 27017"""

def _get_devops_dockerfile(keyword: str) -> str:
    """DevOps tools Dockerfile template"""
    return """FROM alpine:latest

RUN apk add --no-cache \\
    curl \\
    wget \\
    git \\
    bash \\
    openssh-client \\
    docker \\
    kubectl

WORKDIR /workspace

CMD [\"/bin/bash\"]"""

def _get_ubuntu_dockerfile() -> str:
    """Default Ubuntu Dockerfile template"""
    return """FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

CMD [\"/bin/bash\"]
ENTRYPOINT [\"sleep\", \"infinity\"]"""

async def convert_prompt_to_dockerfile_batched(prompts: List[str]) -> List[str]:
    """Convert prompts to dockerfiles in batch - async version with caching and optimizations"""
    
    print(f"Extracting keywords for {len(prompts)} prompts in batch...")
    
    # Step 1: Extract all keywords in batch using GPT
    extractor = KeywordExtractor(
        model_name="gpt-4o-mini-2024-07-18"
    )
    
    try:
        keyword_results = extractor(prompts).dataset
        keywords = keyword_results["keyword"]
        print(f"Successfully extracted {len(keywords)} keywords")
    except Exception as e:
        print(f"Error with batch keyword extraction: {e}")
        # Fallback to individual extraction
        keywords = []
        for prompt in prompts:
            try:
                result = extract_keywords_fallback(prompt)
                keywords.append(result[0] if result else "python")
            except:
                keywords.append("python")
    
    # Step 2: Deduplicate keywords and create smart batching
    unique_keywords, keyword_mapping = deduplicate_and_batch_keywords(keywords)
    
    print(f"Processing {len(unique_keywords)} unique keywords with smart rate limiting...")
    
    # Process unique keywords with controlled concurrency
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        connector=aiohttp.TCPConnector(limit=5, limit_per_host=3)  # Very conservative limits
    ) as session:
        # Process in smaller batches with delays
        batch_size = 5  # Small batches to avoid rate limits
        unique_results = {}
        
        # Create overall progress bar
        with tqdm(total=len(unique_keywords), desc="Processing keywords", unit="keyword") as pbar:
            for i in range(0, len(unique_keywords), batch_size):
                batch_keywords = unique_keywords[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(unique_keywords) + batch_size - 1)//batch_size
                
                pbar.set_description(f"Batch {batch_num}/{total_batches}")
                
                # Process batch with staggered starts
                batch_tasks = []
                for j, keyword in enumerate(batch_keywords):
                    # Stagger requests by 200ms each to avoid burst limits
                    await asyncio.sleep(0.2 * j)
                    task = search_github_dockerfiles_async(session, keyword)
                    batch_tasks.append((keyword, task))
                
                # Collect batch results with individual keyword progress
                for keyword, task in batch_tasks:
                    try:
                        result = await task
                        unique_results[keyword.lower().strip()] = result[0].content if result else generate_enhanced_dockerfile([keyword])
                        pbar.set_postfix(last_keyword=keyword[:20])
                    except Exception as e:
                        print(f"Error processing '{keyword}': {e}")
                        unique_results[keyword.lower().strip()] = generate_enhanced_dockerfile([keyword])
                        pbar.set_postfix(last_keyword=f"{keyword[:15]}(err)")
                    
                    pbar.update(1)
                
                # Add delay between batches if not the last batch
                if i + batch_size < len(unique_keywords):
                    pbar.set_description(f"Waiting between batches...")
                    await asyncio.sleep(3)  # 3 second delay between batches
    
    # Step 3: Map unique results back to all prompts
    all_dockerfiles = []
    print("Mapping results back to all prompts...")
    for keyword in tqdm(keywords, desc="Mapping dockerfiles", unit="prompt"):
        normalized = keyword.lower().strip()
        dockerfile = unique_results.get(normalized, generate_enhanced_dockerfile([keyword]))
        all_dockerfiles.append(dockerfile)
    
    return all_dockerfiles

async def process_prompt_with_keyword_async(session: aiohttp.ClientSession, i: int, prompt: str, keyword: str) -> Tuple[int, str]:
    """Async process a single prompt with pre-extracted keyword"""
    try:
        dockerfiles = await search_github_dockerfiles_async(session, keyword)
        if dockerfiles:
            # Return the first good dockerfile
            return i, dockerfiles[0].content
        else:
            # Fallback to basic dockerfile
            dockerfile = generate_basic_dockerfile([keyword])
            return i, dockerfile
    except Exception as e:
        print(f"Error processing prompt {i}: {e}")
        # Fallback to basic dockerfile
        dockerfile = generate_basic_dockerfile([keyword])
        return i, dockerfile

@gcs_cache()
def convert_prompt_to_dockerfile_batched_sync(prompts: List[str]) -> List[str]:
    """Synchronous wrapper for the async batch function"""
    return asyncio.run(convert_prompt_to_dockerfile_batched(prompts))

def extract_keywords(prompt: str) -> List[str]:
    """Extract keywords using GPT"""
    return extract_keywords_gpt(prompt)

def extract_keywords_gpt(prompt: str) -> List[str]:
    """GPT-based keyword extraction for GitHub search"""
    extractor = KeywordExtractor(
        model_name="gpt-4o-mini-2024-07-18"  # Use a fast, cheap model for keyword extraction
    )
    
    try:
        result = extractor([prompt]).dataset
        keyword = result["keyword"][0]
        return [keyword] if keyword else []
    except Exception as e:
        print(f"Error extracting keywords with GPT: {e}")
        # Fallback to simple extraction if GPT fails
        return extract_keywords_fallback(prompt)

def extract_keywords_fallback(prompt: str) -> List[str]:
    """Simple fallback keyword extraction"""
    # Simple keyword matching for common technologies
    prompt_lower = prompt.lower()
    
    # Technology keywords in order of priority
    tech_keywords = [
        'python', 'node', 'javascript', 'java', 'go', 'rust', 'php', 'ruby',
        'django', 'flask', 'fastapi', 'react', 'vue', 'angular', 'express',
        'nginx', 'apache', 'redis', 'postgres', 'mysql', 'mongodb',
        'elasticsearch', 'kafka', 'rabbitmq', 'docker', 'kubernetes'
    ]
    
    # Find first matching keyword
    for keyword in tech_keywords:
        if keyword in prompt_lower:
            return [keyword]
    
    # Extract first meaningful word as fallback
    words = prompt_lower.split()
    for word in words:
        if len(word) > 3 and word not in ['with', 'using', 'need', 'want', 'create', 'setup']:
            return [word]
    
    return ['python']  # Ultimate fallback

def extract_keywords_keybert(prompt: str) -> List[str]:
    """KeyBERT semantic keyword extraction"""
    global _keybert_model
    if _keybert_model is None:
        _keybert_model = KeyBERT()

    # Generic words to filter out (not useful for GitHub search)
    generic_words = {
        'startup', 'running', 'using', 'create', 'setup', 'install',
        'need', 'want', 'use', 'set', 'application', 'system',
        'environment', 'container', 'docker', 'image', 'dockerfile'
    }

    # KeyBERT extraction
    keybert_results = _keybert_model.extract_keywords(
        prompt,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=10  # Get more to filter
    )

    # Filter out generic words and low-confidence keywords
    keybert_keywords = []
    for kw, score in keybert_results:
        # Skip if score too low
        if score < 0.3:
            continue

        # Skip if it's a generic word
        kw_lower = kw.lower()
        if kw_lower in generic_words:
            continue

        # Skip if it contains only generic words
        words = kw_lower.split()
        if all(w in generic_words for w in words):
            continue

        keybert_keywords.append(kw)

    return list(dict.fromkeys(keybert_keywords))[:5]  # Limit to top 5 for GitHub search

if __name__ == "__main__":
    test_prompts = [
        "I need to set up a web server with nginx",
        "Create a Python data science environment with numpy and pandas",
        "Set up a database server with PostgreSQL", 
        "I need a Node.js application server",
        "Machine learning environment with TensorFlow"
    ]
    
    print("Testing single prompt:")
    print("=" * 60)
    dockerfile = prompt_to_dockerfile(test_prompts[0])
    print(dockerfile)
    print("\n" + "=" * 60)
    
    print("\nTesting batch processing:")
    print("=" * 60)
    dockerfiles = convert_prompt_to_dockerfile_batched(test_prompts[:2])  # Test with fewer to avoid rate limiting
    for i, (prompt, dockerfile) in enumerate(zip(test_prompts[:2], dockerfiles), 1):
        print(f"\nPrompt {i}: {prompt}")
        print("-" * 60)
        print(dockerfile)
        print()