#!/usr/bin/env python3
"""
Docker toolkit: prompt -> Dockerfile text
Ubuntu-based with online apt package search (no apt-cache required)
"""

import requests
from typing import List, Dict, Set
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from bs4 import BeautifulSoup
import logging
from data.gcs_cache import gcs_cache
# Suppress requests library logging
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Import keyword extraction from github_dockerfiles
from data.github_dockerfiles.utils import extract_keywords as extract_keywords_github

@dataclass
class AptPackage:
    name: str
    description: str
    relevance_score: float = 0.0

def prompt_to_dockerfile(prompt: str) -> str:
    """Main function: prompt -> Dockerfile text with Ubuntu base"""
    keywords = extract_keywords(prompt)
    
    # Search for apt packages based on keywords
    apt_packages = search_apt_packages_for_keywords(keywords)
    
    # Generate Dockerfile with Ubuntu base
    return generate_ubuntu_dockerfile(apt_packages, keywords)

def search_apt_packages_for_keywords(keywords: List[str]) -> Set[str]:
    """Search apt packages for given keywords"""

    # Always include base utilities (known to exist)
    packages = {
        'curl',
        'wget',
        'git',
        'vim',
        'nano'
    }

    # Search apt for each keyword
    for keyword in keywords:
        results = search_ubuntu_packages_online(keyword)
        print(f"Found {len(results)} packages for keyword '{keyword}':")

        # Add top relevant packages, but validate they exist first
        for pkg in results[:3]:  # Reduced from 5 to 3 for better quality
            if is_valid_ubuntu_package(pkg.name):
                normalized_name = normalize_package_name(pkg.name)
                print(f"  Adding package: {normalized_name} (score: {pkg.relevance_score:.1f})")
                packages.add(normalized_name)
            else:
                print(f"  Skipping invalid package: {pkg.name}")

    return packages

def normalize_package_name(package_name: str) -> str:
    """Normalize package name using known mappings"""
    mappings = get_package_name_mappings()
    name_lower = package_name.lower()
    return mappings.get(name_lower, package_name)

def validate_package_exists_online(package_name: str) -> bool:
    """Check if a package actually exists in Ubuntu repositories via HTTP"""
    try:
        # Try multiple Ubuntu versions to be thorough
        versions = ['noble', 'jammy', 'focal']  # 24.04, 22.04, 20.04

        for version in versions:
            url = f"https://packages.ubuntu.com/{version}/{package_name}"
            response = requests.get(url, timeout=5, allow_redirects=True)

            if response.status_code == 200:
                # Check if the page contains error message or valid package details
                # Invalid packages get a 200 but with "Ubuntu – Error" in title
                if 'Ubuntu – Error' in response.text or 'Error 404' in response.text:
                    continue
                # Valid package page should have "Details of package" in title
                if 'Details of package' in response.text or f'Package: {package_name}' in response.text:
                    return True

        # If none of the versions had valid package content, package doesn't exist
        return False
    except Exception as e:
        print(f"  Warning: Could not validate package '{package_name}': {e}")
        # On network error, be conservative and reject the package
        return False

def is_valid_ubuntu_package(package_name: str) -> bool:
    """Validate that a package exists in Ubuntu repositories"""
    # Normalize the package name first
    normalized_name = normalize_package_name(package_name)
    
    # Check against our known invalid packages (using normalized name)
    if should_exclude_package(normalized_name):
        print(f"  Package '{package_name}' -> '{normalized_name}' excluded (obsolete/problematic)")
        return False

    # List of known valid common packages to speed up validation
    known_valid = {
        'curl', 'wget', 'git', 'vim', 'nano', 'python3', 'python3-pip',
        'build-essential', 'nodejs', 'npm', 'nginx', 'apache2', 'mysql-server',
        'postgresql', 'redis-server', 'docker.io', 'openssh-server',
        'rsync', 'zip', 'unzip', 'tar', 'gzip', 'sqlite3', 'tree',
        'htop', 'tmux', 'screen', 'jq', 'vim-nox', 'lsb-release'
    }

    if normalized_name.lower() in known_valid:
        if normalized_name != package_name:
            print(f"  Package '{package_name}' mapped to '{normalized_name}'")
        return True

    # For suspicious packages (short names, contain common problematic patterns), validate online
    suspicious_patterns = ['lsb', 'lsm', 'lsdb', 'lib-', 'dev-', 'oslo', 'postgresql-', 'python-', 'python3-']
    is_suspicious = (len(normalized_name) <= 4 or
                    any(pattern in normalized_name.lower() for pattern in suspicious_patterns) or
                    '-' in normalized_name)  # Any package with hyphen should be validated

    if is_suspicious:
        print(f"  Validating suspicious package '{normalized_name}' online...")
        exists = validate_package_exists_online(normalized_name)
        if not exists:
            print(f"  Package '{normalized_name}' does not exist in repositories")
            return False
        else:
            print(f"  Package '{normalized_name}' validated successfully")

    # For other packages, we'll assume they're valid if they pass our exclusion rules
    # This is a pragmatic approach to avoid making too many HTTP requests
    return True

def search_ubuntu_packages_online(keyword: str) -> List[AptPackage]:
    """Search Ubuntu packages online (no apt-cache needed)"""
    # Use Ubuntu packages search
    url = "https://packages.ubuntu.com/search"
    params = {
        'keywords': keyword,
        'searchon': 'names',
        'suite': 'jammy',  # Ubuntu 22.04 LTS
        'section': 'all'
    }

    response = requests.get(url, params=params, timeout=15)

    if response.status_code != 200:
        print(f"Warning: Failed to search packages for '{keyword}' (status: {response.status_code})")
        return []

    # Parse HTML response
    soup = BeautifulSoup(response.text, 'html.parser')

    packages = []

    # Look for package links - they typically have href like /jammy/packagename
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        # Package links look like: /jammy/packagename or /jammy/arch/packagename
        if href.startswith('/jammy/') and href.count('/') >= 2:
            parts = href.split('/')
            if len(parts) >= 3:
                package_name = parts[2]  # Get the package name part

                # Skip if it's just the architecture
                if package_name in ['all', 'amd64', 'arm64', 'armhf', 'i386', 'ppc64el', 's390x']:
                    continue

                # Skip excluded packages (GUI, desktop, etc.)
                if should_exclude_package(package_name):
                    continue

                # Get description - usually in the same row or nearby
                description = ""
                parent = link.find_parent('li')
                if parent:
                    desc_elem = parent.find('br')
                    if desc_elem and desc_elem.next_sibling:
                        description = str(desc_elem.next_sibling).strip()

                # Calculate relevance
                score = calculate_relevance(keyword, package_name, description)

                packages.append(AptPackage(
                    name=package_name,
                    description=description,
                    relevance_score=score
                ))

    # Sort by relevance
    packages.sort(key=lambda x: x.relevance_score, reverse=True)

    return packages

def search_repology_packages(keyword: str) -> List[AptPackage]:
    """Alternative: Search using Repology API"""
    try:
        # Repology aggregates package info from many distros
        url = f"https://repology.org/api/v1/projects/{keyword}/"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            packages = []
            
            for item in data:
                if item.get('repo', '').startswith('ubuntu'):
                    name = item.get('srcname') or item.get('binname', '')
                    if name:
                        packages.append(AptPackage(
                            name=name,
                            description=item.get('summary', ''),
                            relevance_score=100.0
                        ))
            
            return packages[:5]
        
        return []
        
    except Exception as e:
        print(f"Error searching Repology for '{keyword}': {e}")
        return []

def search_debian_packages(keyword: str) -> List[AptPackage]:
    """Search Debian packages (compatible with Ubuntu)"""
    try:
        # Debian package search API
        url = "https://sources.debian.org/api/search/"
        params = {'q': keyword}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            packages = []
            
            # Get exact matches
            for result in data.get('results', {}).get('exact', {}).get('results', [])[:3]:
                name = result.get('name', '')
                if name:
                    packages.append(AptPackage(
                        name=name,
                        description=f"Debian package: {name}",
                        relevance_score=100.0
                    ))
            
            # Get other matches
            for result in data.get('results', {}).get('other', [])[:3]:
                name = result.get('name', '')
                if name:
                    packages.append(AptPackage(
                        name=name,
                        description=f"Debian package: {name}",
                        relevance_score=50.0
                    ))
            
            return packages
        
        return []
        
    except Exception as e:
        print(f"Error searching Debian packages for '{keyword}': {e}")
        return []

def get_package_name_mappings() -> Dict[str, str]:
    """Get mappings from obsolete/incorrect package names to correct ones"""
    return {
        # Obsolete packages with modern replacements
        'lsb': 'lsb-release',
        'lsb-core': 'lsb-release', 
        'python': 'python3',
        'python-dev': 'python3-dev',
        'python-pip': 'python3-pip',
        'nodejs-legacy': 'nodejs',
        'mysql-server-5.7': 'mysql-server',
        'postgresql-9.6': 'postgresql',
        'openjdk-8-jdk': 'openjdk-11-jdk',
        'php7.4': 'php',
        'apache2.4': 'apache2',
        'nginx-full': 'nginx',
        # Common typos/variations
        'gcc-dev': 'build-essential',
        'make-dev': 'build-essential',
        'g++': 'build-essential',
        'vim-enhanced': 'vim',
        'emacs-nox': 'emacs-nox',
    }

def get_obsolete_packages() -> Set[str]:
    """Get set of packages that are known to be obsolete or problematic"""
    return {
        # Packages that no longer exist in modern Ubuntu
        'lsb-base',  # Replaced by lsb-release
        'python2.7', 'python2.7-dev',  # Python 2 is deprecated
        'mysql-server-5.5', 'mysql-server-5.6',  # Old MySQL versions
        'postgresql-9.1', 'postgresql-9.2', 'postgresql-9.3', 'postgresql-9.4', 'postgresql-9.5',
        'openjdk-6-jdk', 'openjdk-7-jdk',  # Old Java versions
        'php5', 'php5-cli', 'php5-fpm',  # PHP 5 is obsolete
        'nodejs-legacy',  # No longer needed
        'libssl1.0-dev',  # Old SSL version
        'gcc-4.8', 'gcc-4.9',  # Very old GCC versions
        # Packages with naming issues
        'lsdb',  # Often confused with lsb
        'lsm',   # Often confused with lsb
    }

def should_exclude_package(package_name: str) -> bool:
    """Check if a package should be excluded (GUI, problematic packages)"""
    import re
    name_lower = package_name.lower()

    # Whitelist of known-good short package names
    short_valid_packages = {'curl', 'wget', 'git', 'vim', 'nano', 'ssh', 'npm', 'tar', 'zip', 'gcc', 'gdb', 'make', 'sed', 'awk', 'jq'}

    # Exclude very short package names (likely false matches like "os8", "osc", "osmo")
    # UNLESS they are in our whitelist
    if len(package_name) <= 4 and name_lower not in short_valid_packages:
        return True

    # Exclude packages with version numbers in the name (e.g., postgresql-14-debversion)
    # These are often version-specific and may not exist
    if re.search(r'-\d+\.\d+', package_name) or re.search(r'-\d+-', package_name):
        return True

    # Check if package is obsolete
    if name_lower in get_obsolete_packages():
        return True

    # GUI/Desktop packages that don't belong in containers
    gui_patterns = [
        'gnome-', 'kde-', 'xfce-', 'lxde-', 'mate-',  # Desktop environments
        'x11-', 'libx11-', 'xorg-',  # X11 packages
        'gtk-', 'qt-', 'qtwebengine-',  # GUI toolkits
        'libstartup-notification',  # X11 startup notification
        'unclutter-',  # X11 utilities
        'desktop-', 'plasma-',  # Desktop related
        'fonts-', 'ttf-',  # Font packages (usually not needed)
        'aspell-', 'hunspell-', 'myspell-',  # Spell checkers
        'libreoffice-', 'thunderbird-', 'firefox-',  # Desktop applications
        'r-cran-',  # R packages (use R's own package manager)
        'texlive-',  # LaTeX (very large, should be explicit)
        'vim-gtk', 'vim-tjp',  # VIM GUI variants that don't exist
        'ejabberd-mod-', 'libeclipse-', 'restricted-ssh-', 'ruby-spring-',  # Questionable packages
        'python-oslo.', 'python3-oslo.',  # OpenStack packages (usually too specific)
    ]

    # Check if package matches any exclusion pattern
    for pattern in gui_patterns:
        if name_lower.startswith(pattern) or pattern in name_lower:
            return True

    # Additional specific excludes
    excluded_packages = {
        'task-desktop', 'task-gnome-desktop', 'task-kde-desktop',
        'desktop-base', 'desktop-file-utils',
        'vim-gtk', 'vim-tjp',  # These specific VIM variants don't exist
        'ejabberd-mod-shcommands', 'libeclipse-core-commands-java',
        'libeclipse-e4-core-commands-java', 'restricted-ssh-commands',
        'ruby-spring-commands-rspec',  # These packages likely don't exist
    }

    return name_lower in excluded_packages

def calculate_relevance(keyword: str, package_name: str, description: str) -> float:
    """Calculate how relevant a package is to the keyword"""
    score = 0.0
    kw_lower = keyword.lower()
    name_lower = package_name.lower()
    desc_lower = description.lower()

    # Exact match in name (highest priority)
    if kw_lower == name_lower:
        score += 100
    # Keyword is in package name
    elif kw_lower in name_lower:
        score += 50
    # Package name starts with keyword
    if name_lower.startswith(kw_lower):
        score += 30

    # Keyword in description
    if kw_lower in desc_lower:
        score += 10

    # Prefer shorter package names (usually more canonical)
    score -= len(package_name) * 0.1

    # Penalize -dev, -doc, -dbg packages unless specifically requested
    if any(suffix in name_lower for suffix in ['-dev', '-doc', '-dbg', '-data']):
        if 'dev' not in kw_lower and 'doc' not in kw_lower:
            score -= 20

    return score

def get_python_pip_packages(keywords: List[str]) -> Set[str]:
    """Get Python packages that should be installed via pip"""
    pip_packages = set()
    
    # Check if any keyword suggests Python packages
    python_keywords = {
        'numpy', 'pandas', 'scipy', 'matplotlib', 
        'scikit-learn', 'sklearn', 'tensorflow', 'pytorch',
        'requests', 'flask', 'django', 'fastapi'
    }
    
    for keyword in keywords:
        kw_lower = keyword.lower()
        if kw_lower in python_keywords:
            pip_packages.add(kw_lower)
        elif 'data science' in kw_lower:
            pip_packages.update(['numpy', 'pandas', 'matplotlib'])
        elif 'machine learning' in kw_lower:
            pip_packages.update(['numpy', 'scikit-learn', 'pandas'])
    
    return pip_packages

def generate_ubuntu_dockerfile(apt_packages: Set[str], keywords: List[str]) -> str:
    """Generate Dockerfile with Ubuntu base and apt packages"""
    
    # Check if we need Python pip packages
    pip_packages = get_python_pip_packages(keywords)

    print(f"Final apt packages to install: {sorted(apt_packages)}")
    if pip_packages:
        print(f"Pip packages to install: {sorted(pip_packages)}")

    # If we have pip packages, ensure python3 and pip are installed
    if pip_packages:
        apt_packages.update(['python3', 'python3-pip', 'python3-dev', 'build-essential'])
    
    lines = [
        "FROM ubuntu:latest",
        "",
        "ENV DEBIAN_FRONTEND=noninteractive",
        "",
        "WORKDIR /app",
        "",
        "# Install system packages",
        "RUN apt-get update && apt-get install -y --no-install-recommends \\",
    ]
    
    # Add all apt packages
    sorted_packages = sorted(apt_packages)
    for i, pkg in enumerate(sorted_packages):
        lines.append(f"    {pkg} \\")
    
    lines.append("    && rm -rf /var/lib/apt/lists/*")
    
    # Add Python pip packages if any
    if pip_packages:
        lines.append("")
        lines.append("# Install Python packages")
        lines.append("RUN pip3 install --no-cache-dir \\")
        sorted_pip = sorted(pip_packages)
        for i, pkg in enumerate(sorted_pip):
            if i < len(sorted_pip) - 1:
                lines.append(f"    {pkg} \\")
            else:
                lines.append(f"    {pkg}")
    
    lines.append("")
    lines.append("ENTRYPOINT [\"sleep\", \"infinity\"]")
    
    return '\n'.join(lines)

@gcs_cache()
def convert_prompt_to_dockerfile_batched(prompts: List[str]) -> List[str]:
    """Convert prompts to dockerfiles in batch"""
    
    all_dockerfiles = []
    # Warm up the model
    prompt_to_dockerfile(prompts[0])
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_prompt = {executor.submit(prompt_to_dockerfile, prompt): prompt for prompt in prompts}
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_prompt), total=len(prompts)):
            dockerfile = future.result()
            all_dockerfiles.append(dockerfile)
    
    return all_dockerfiles

def extract_keywords(prompt: str) -> List[str]:
    """Extract keywords using the GitHub Dockerfiles extraction method"""
    return extract_keywords_github(prompt)

if __name__ == "__main__":
    test_prompts = [
        "I need to set up a web server with nginx",
        "Create a Python data science environment with numpy and pandas",
        "Set up a database server with PostgreSQL",
        "I need a DNS server with bind9",
        "Machine learning environment with TensorFlow"
    ]
    
    print("Testing single prompt:")
    print("=" * 60)
    dockerfile = prompt_to_dockerfile(test_prompts[0])
    print(dockerfile)
    print("\n" + "=" * 60)
    
    print("\nTesting batch processing:")
    print("=" * 60)
    dockerfiles = convert_prompt_to_dockerfile_batched(test_prompts)
    for i, (prompt, dockerfile) in enumerate(zip(test_prompts, dockerfiles), 1):
        print(f"\nPrompt {i}: {prompt}")
        print("-" * 60)
        print(dockerfile)
        print()