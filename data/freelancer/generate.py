#!/usr/bin/env python3
"""
Generate Freelancer projects dataset in sandboxes style
"""

import requests
import time
import asyncio
import aiohttp
from typing import List
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
from tqdm import tqdm

# Import from parent package
from data.commons import upload_tasks_to_hf, generate_tasks_from_questions, upsample_tasks_directory, upload_traces_to_hf
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.gcs_cache import gcs_cache
def scrape_single_project(url: str) -> str:
    """Scrape a single project URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        
        # Handle 429 (Too Many Requests) with retry
        if response.status_code == 429:
            print(f"Rate limited (429) for {url}, retrying in 1 second...")
            time.sleep(1)
            response = requests.get(url, headers=headers, timeout=30)
        
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the project description section
        description = None

        # Try common section selectors for project descriptions
        selectors = [
            'div.ContentContainer',  # Main project content container
            'div[class*="project-description"]',
            'div[class*="description"]',
            'div[class*="brief"]',
            'div[class*="content"]',
            'section[class*="project"]',
            'div.ql-editor',  # Rich text editor content
            '.project-view-details',
            '.project-brief'
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    description = text
                    print(f"Found project section ({selector}): {len(description)} chars")
                    break

        # Fallback to meta description if no section found
        if not description:
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                description = meta_desc.get('content').strip()
                print(f"Used meta description fallback: {len(description)} chars")

        if description:
            return description
        else:
            # Final fallback
            title = soup.title.string if soup.title else url.split('/')[-1]
            desc = f"Project: {title}. Please analyze this freelancer project and provide a detailed proposal including technical approach, timeline, and deliverables."
            print(f"Used title fallback: {len(desc)} chars")
            return desc

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        # Add a basic description based on URL
        url_desc = url.split('/')[-1].replace('-', ' ').title()
        desc = f"Freelancer project: {url_desc}. Analyze this project and provide a comprehensive proposal."
        return desc

async def async_scrape_single_project(session: aiohttp.ClientSession, url: str) -> str:
    """Async scrape a single project URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')

                # Find the project description section
                description = None

                # Try common section selectors for project descriptions
                selectors = [
                    'div.ContentContainer',  # Main project content container
                    'div[class*="project-description"]',
                    'div[class*="description"]',
                    'div[class*="brief"]',
                    'div[class*="content"]',
                    'section[class*="project"]',
                    'div.ql-editor',  # Rich text editor content
                    '.project-view-details',
                    '.project-brief'
                ]

                for selector in selectors:
                    element = soup.select_one(selector)
                    if element:
                        text = element.get_text(separator=' ', strip=True)
                        if len(text) > 100:
                            description = text
                            print(f"Found project section ({selector}): {len(description)} chars")
                            break

                # Fallback to meta description if no section found
                if not description:
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    if meta_desc and meta_desc.get('content'):
                        description = meta_desc.get('content').strip()
                        print(f"Used meta description fallback: {len(description)} chars")

                if description:
                    return description
                else:
                    # Final fallback
                    title = soup.title.string if soup.title else url.split('/')[-1]
                    desc = f"Project: {title}. Please analyze this freelancer project and provide a detailed proposal including technical approach, timeline, and deliverables."
                    print(f"Used title fallback: {len(desc)} chars")
                    return desc

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        # Add a basic description based on URL
        url_desc = url.split('/')[-1].replace('-', ' ').title()
        desc = f"Freelancer project: {url_desc}. Analyze this project and provide a comprehensive proposal."
        return desc

async def async_discover_all_job_categories(session: aiohttp.ClientSession) -> List[str]:
    """Async discover all available job categories from freelancer.com/jobs page"""
    print("Discovering all job categories...")

    categories = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        async with session.get('https://www.freelancer.com/projects', headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            print(f"Category discovery response status: {response.status}")
            if response.status == 200:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')

                # Debug: print some of the content to see what we're getting
                print(f"Page content length: {len(content)} characters")

                # Find all links that point to job categories
                all_links = soup.find_all('a', href=True)
                print(f"Found {len(all_links)} total links on page")

                job_links = 0
                for link in all_links:
                    href = link.get('href')
                    if href and '/jobs/' in href and href != '/jobs/':
                        job_links += 1
                        print(f"Job link found: {href}")
                        # Extract category from URL like /jobs/web-development or /jobs/python-programming
                        if href.startswith('/jobs/') and href.count('/') == 2:
                            category = href.split('/jobs/')[1].rstrip('/')
                            if category and category not in categories:
                                categories.append(category)
                                print(f"Added category: {category}")

                print(f"Total job-related links found: {job_links}")

        print(f"Found {len(categories)} job categories: {categories[:10]}..." if len(categories) > 10 else f"Found {len(categories)} job categories: {categories}")

        # If no categories found, use fallback
        if not categories:
            print("No categories discovered, using fallback list")
            categories = ['web-development', 'mobile-app-development', 'python-programming', 'javascript', 'c-sharp-programming', 'php', 'java']

        return categories

    except Exception as e:
        print(f"Error discovering categories: {e}")
        # Fallback to known categories
        return ['web-development', 'mobile-app-development', 'python-programming', 'javascript', 'c-sharp-programming', 'php', 'java']

async def async_discover_job_urls_from_pages(session: aiohttp.ClientSession, category: str, max_pages: int = 1000) -> List[str]:
    """Async discover job URLs from freelancer.com/jobs/ pages by iterating through page numbers SEQUENTIALLY"""
    print(f"[{category}] Starting discovery (max {max_pages} pages)...")

    job_urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for page in range(1, max_pages + 1):
        try:
            # Construct URL for current page
            url = f"https://www.freelancer.com/jobs/{category}/{page}"

            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                print(f"[{category}] Page {page} → Status: {response.status}")

                # Debug 403 errors
                if response.status == 403:
                    print(f"[{category}] 403 FORBIDDEN on page {page}")
                    print(f"[{category}] Response headers: {dict(response.headers)}")

                    # Try to read a bit of the response content for debugging
                    try:
                        content_preview = await response.text()
                        print(f"[{category}] Response content preview (first 200 chars): {content_preview[:200]}")
                    except:
                        print(f"[{category}] Could not read response content")

                    break

                if response.status == 404:
                    print(f"[{category}] Reached end of pages at page {page} (404 Not Found)")
                    break

                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')

                    # Find job links on this page
                    page_job_urls = []
                    for link in soup.find_all('a', href=True):
                        href = link.get('href')
                        if href and '/projects/' in href:
                            # Convert relative to absolute URL
                            if href.startswith('/'):
                                full_url = urljoin('https://www.freelancer.com', href)
                            else:
                                full_url = href

                            # Basic validation - should be a specific project
                            parts = full_url.split('/')
                            if (len(parts) >= 5 and
                                parts[4] != '' and  # Has project slug
                                full_url not in job_urls and
                                '/projects/' in full_url and
                                not full_url.endswith('/projects')):
                                page_job_urls.append(full_url)
                                job_urls.append(full_url)

                    print(f"[{category}] Found {len(page_job_urls)} job URLs on page {page}")

                    # If no URLs found on this page, might have reached the end
                    if len(page_job_urls) == 0:
                        print(f"[{category}] No job URLs found on page {page}, stopping early")
                        break

                else:
                    print(f"[{category}] Unexpected status {response.status} on page {page}")
                    break

        except Exception as e:
            print(f"[{category}] Error fetching page {page}: {e}")
            break

        # Small delay between pages to be respectful
        await asyncio.sleep(0.5)

    print(f"[{category}] Completed: Found {len(job_urls)} URLs from {page-1} pages processed")
    return job_urls

async def async_discover_job_urls_from_all_categories(session: aiohttp.ClientSession) -> List[str]:
    """Async discover job URLs from all available categories with limited concurrency"""
    print("Discovering job URLs from categories with limited concurrency...")

    # Use the SYNC category discovery that works
    categories = discover_all_job_categories()

    if not categories:
        print("No categories found, returning empty list")
        return []

    # LIMIT to a few known working categories for debugging
    # categories = [cat for cat in test_categories if cat in categories]
    # print(f"DEBUGGING: Testing with known working categories: {categories}")

    # Process in small batches to avoid overwhelming the server
    batch_size = 3
    all_job_urls = []

    for i in range(0, len(categories), batch_size):
        batch = categories[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(categories)-1)//batch_size + 1}: {batch}")

        # Create tasks for this batch
        category_tasks = []
        for category in batch:
            task = asyncio.create_task(async_discover_job_urls_from_pages(session, category, max_pages=2))
            category_tasks.append(task)

        # Wait for batch to complete
        category_results = await asyncio.gather(*category_tasks, return_exceptions=True)

        # Collect URLs from this batch
        for j, result in enumerate(category_results):
            if isinstance(result, Exception):
                print(f"Error processing category {batch[j]}: {result}")
                continue

            category_urls = result
            print(f"Category {batch[j]} contributed {len(category_urls)} URLs")

            # Remove duplicates while preserving order
            for url in category_urls:
                if url not in all_job_urls:
                    all_job_urls.append(url)

        # Pause between batches to be respectful
        if i + batch_size < len(categories):
            print(f"Waiting 3 seconds before next batch...")
            await asyncio.sleep(3)

    print(f"\nCompleted batched discovery: {len(all_job_urls)} total job URLs from {len(categories)} categories")
    return all_job_urls

async def async_scrape_all_projects(job_urls: List[str]) -> List[str]:
    """Async scrape all project descriptions in parallel"""
    print(f"Starting parallel scraping of {len(job_urls)} projects...")

    connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create scraping tasks for all URLs
        scraping_tasks = []
        for url in job_urls:
            task = asyncio.create_task(async_scrape_single_project(session, url))
            scraping_tasks.append(task)

        # Execute all scraping tasks in parallel
        print(f"Executing {len(scraping_tasks)} scraping tasks in parallel...")
        scraping_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)

        # Process results
        questions = []
        for i, result in enumerate(scraping_results):
            if isinstance(result, Exception):
                print(f"Error scraping {job_urls[i]}: {result}")
                # Add fallback description
                url_desc = job_urls[i].split('/')[-1].replace('-', ' ').title()
                desc = f"Freelancer project: {url_desc}. Analyze this project and provide a comprehensive proposal."
                questions.append(desc)
            else:
                questions.append(result)
                print(f"Completed scraping: {job_urls[i]}")

    print(f"Successfully scraped {len(questions)} project descriptions in parallel")
    return questions

async def async_scrape_freelancer_projects() -> List[str]:
    """Main async function to discover and scrape all freelancer projects"""
    # More conservative connection limits to avoid rate limiting
    connector = aiohttp.TCPConnector(limit=20, limit_per_host=5)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Step 1: Discover all job URLs from all categories with rate limiting
        job_urls = await async_discover_job_urls_from_all_categories(session)

        if not job_urls:
            print("No job URLs discovered")
            return []

        # Step 2: Scrape all projects in parallel (with batching)
        questions = await async_scrape_all_projects(job_urls)
        return questions

def discover_all_job_categories() -> List[str]:
    """Discover all available job categories from freelancer.com/jobs page"""
    print("Discovering all job categories...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    categories = []

    try:
        response = requests.get('https://www.freelancer.com/projects', headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all links that point to job categories
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and '/jobs/' in href and href != '/jobs/':
                category = href.split('/jobs/')[1].rstrip('/')
                if category and category not in categories:
                    categories.append(category)

        print(f"Found {len(categories)} job categories: {categories[:10]}..." if len(categories) > 10 else f"Found {len(categories)} job categories: {categories}")
        return categories

    except Exception as e:
        print(f"Error discovering categories: {e}")
        # Fallback to known categories
        return ['web-development', 'mobile-app-development', 'python-programming', 'javascript', 'c-sharp-programming', 'php', 'java']

def discover_job_urls_from_pages(max_urls: int = 20, category: str = "c-sharp-programming") -> List[str]:
    """Discover job URLs from freelancer.com/jobs/ pages by iterating through page numbers"""
    print(f"Discovering job URLs from /jobs/{category}/ pages...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    job_urls = []
    page = 1

    while len(job_urls) < max_urls:
        try:
            # Construct URL for current page
            url = f"https://www.freelancer.com/jobs/{category}/{page}"
            print(f"Checking page {page}: {url}")

            response = requests.get(url, headers=headers, timeout=30)

            # Check if page exists (404 means we've reached the end)
            if response.status_code == 404:
                print(f"Reached end of pages at page {page} (404 Not Found)")
                break

            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find job links on this page
            page_job_urls = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href and '/projects/' in href:
                    # Convert relative to absolute URL
                    if href.startswith('/'):
                        full_url = urljoin('https://www.freelancer.com', href)
                    else:
                        full_url = href

                    # Basic validation - should be a specific project
                    parts = full_url.split('/')
                    if (len(parts) >= 5 and
                        parts[4] != '' and  # Has project slug
                        full_url not in job_urls and
                        '/projects/' in full_url and
                        not full_url.endswith('/projects')):
                        page_job_urls.append(full_url)
                        job_urls.append(full_url)

                        if len(job_urls) >= max_urls:
                            break

            print(f"Found {len(page_job_urls)} job URLs on page {page}")

            # If no URLs found on this page, might have reached the end
            if len(page_job_urls) == 0:
                print(f"No job URLs found on page {page}, stopping")
                break

            page += 1
            time.sleep(2)  # Be respectful with rate limiting

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
        except Exception as e:
            print(f"Unexpected error on page {page}: {e}")
            break

    print(f"Found total of {len(job_urls)} job URLs from {page-1} pages")
    return job_urls

def discover_job_urls_from_pages_unlimited(category: str) -> List[str]:
    """Discover job URLs from freelancer.com/jobs/ pages by iterating through ALL pages until 404"""
    print(f"Discovering ALL job URLs from /jobs/{category}/ pages...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    job_urls = []
    page = 1

    while True:
        try:
            # Construct URL for current page
            url = f"https://www.freelancer.com/jobs/{category}/{page}"
            print(f"Checking page {page}: {url}")

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find job links on this page
            page_job_urls = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href and '/projects/' in href:
                    # Convert relative to absolute URL
                    if href.startswith('/'):
                        full_url = urljoin('https://www.freelancer.com', href)
                    else:
                        full_url = href

                    # Basic validation - should be a specific project (not category)
                    parts = full_url.split('/')
                    if (len(parts) >= 5 and
                        parts[4] != '' and  # Has project slug
                        full_url not in job_urls and
                        '/projects/' in full_url and
                        not full_url.endswith('/projects')):
                        page_job_urls.append(full_url)
                        job_urls.append(full_url)

            print(f"Found {len(page_job_urls)} job URLs on page {page}")

            # If no URLs found on this page, might have reached the end
            if len(page_job_urls) == 0:
                print(f"No job URLs found on page {page}, stopping")
                break

            page += 1
            time.sleep(2)  # Be respectful

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            if "404" in str(e):
                print(f"Reached end of pages at page {page} (404)")
            break
        except Exception as e:
            print(f"Unexpected error on page {page}: {e}")
            break

    print(f"Found total of {len(job_urls)} job URLs from {page-1} pages for category {category}")
    return job_urls

def discover_job_urls_from_all_categories(max_categories: int = 50, max_per_category: int = None) -> List[str]:
    """Discover job URLs from all available categories with limits on categories but not pages"""
    print(f"Discovering job URLs from up to {max_categories} categories...")
    if max_per_category:
        print(f"Max {max_per_category} URLs per category")
    else:
        print("No limit on URLs per category (all pages)")

    # Get all available categories
    categories = discover_all_job_categories()

    # Limit the number of categories to process
    categories = categories[:max_categories]

    all_job_urls = []

    # Create progress bar for category processing
    with tqdm(total=len(categories),
              desc="Processing categories",
              unit="category") as pbar:

        for category in categories:
            pbar.set_postfix_str(f"{category[:30]}...")

            try:
                if max_per_category:
                    # Use the limited version
                    category_urls = discover_job_urls_from_pages(max_per_category, category)
                else:
                    # Use unlimited version - discover all pages for this category
                    category_urls = discover_job_urls_from_pages_unlimited(category)

                # Remove duplicates while preserving order
                for url in category_urls:
                    if url not in all_job_urls:
                        all_job_urls.append(url)

                pbar.set_postfix_str(f"{category[:20]}: {len(category_urls)} URLs | Total: {len(all_job_urls)}")
                pbar.update(1)

                # Brief pause between categories
                time.sleep(1)

            except Exception as e:
                pbar.set_postfix_str(f"ERROR: {category}")
                pbar.update(1)
                continue

    print(f"\nCompleted discovery: {len(all_job_urls)} total job URLs from {len(categories)} categories")
    return all_job_urls

@gcs_cache()
def scrape_freelancer_projects(use_discovery: bool = False, max_projects: int = 20, category: str = "c-sharp-programming", use_all_categories: bool = False) -> List[str]:
    """Scrape real Freelancer project descriptions in parallel"""

    project_urls = discover_job_urls_from_all_categories(max_categories=10000, max_per_category=None)
    questions = []

    # Use ThreadPoolExecutor for parallel scraping with progress bar
    with ThreadPoolExecutor(max_workers=min(8, len(project_urls))) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(scrape_single_project, url): url for url in project_urls}

        # Process completed tasks with progress bar
        with tqdm(total=len(project_urls), desc="Scraping projects", unit="project") as pbar:
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    description = future.result()
                    questions.append(description)
                    pbar.set_postfix_str(f"✓ {url.split('/')[-1][:30]}...")
                    pbar.update(1)
                except Exception as e:
                    # Add fallback description
                    url_desc = url.split('/')[-1].replace('-', ' ').title()
                    desc = f"Freelancer project: {url_desc}. Analyze this project and provide a comprehensive proposal."
                    questions.append(desc)
                    pbar.set_postfix_str(f"✗ {url.split('/')[-1][:30]}...")
                    pbar.update(1)

    print(f"Successfully scraped {len(questions)} project descriptions in parallel")
    return questions

def main() -> None:
    """Main function - coordinates the pipeline"""
    
    questions = scrape_freelancer_projects(
        use_discovery=True,
        max_projects=None,
        use_all_categories=True
    )

    final_dataset_dir = generate_tasks_from_questions(questions, "freelancer")
    upsampled_dataset_dir = upsample_tasks_directory(final_dataset_dir, 10_000)
    hf_dataset = run_dataset_to_traces(upsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    upload_traces_to_hf(hf_dataset, "mlfoundations-dev/freelancer-projects-sandboxes-traces-terminus-2", "SFT")
    upload_tasks_to_hf(upsampled_dataset_dir, "mlfoundations-dev/freelancer-projects-sandboxes")

if __name__ == "__main__":
    main()