import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://www.bis.gov.in"
DOMAIN = "bis.gov.in"
MAX_PAGES = 50 # Limit for demonstration purposes, can be increased
OUTPUT_FILE = "scraped_data.json"

visited_urls = set()
scraped_data = []

def is_valid_url(url):
    """Check if the URL belongs to the BIS domain and is an HTML page."""
    try:
        parsed = urlparse(url)
        # Ensure it's the correct domain and HTTP/HTTPS
        if DOMAIN not in parsed.netloc or parsed.scheme not in ['http', 'https']:
            return False
        
        # Skip media files and documents
        skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.jpg', '.png', '.zip', '.rar', '.mp4']
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        return True
    except Exception:
        return False

def clean_text(text):
    """Clean extracted text from HTML."""
    # Remove extra whitespace and newlines
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned = '\n'.join(chunk for chunk in chunks if chunk)
    return cleaned

def crawl(url, depth=0, max_depth=3):
    """Recursively crawl pages."""
    if url in visited_urls or len(visited_urls) >= MAX_PAGES or depth > max_depth:
        return

    visited_urls.add(url)
    logger.info(f"Crawling: {url} (Depth: {depth})")

    try:
        # Include headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
            return

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, and navigation tags
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.extract()

        # Extract title
        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
        
        # Extract main text content (focus on body or main div if possible to avoid boilerplate)
        main_content = soup.find('main') or soup.find('div', id='content') or soup.body
        
        if main_content:
            text = clean_text(main_content.get_text(separator=' '))
            
            # Only add if there is substantial text
            if len(text) > 100:
                scraped_data.append({
                    "url": url,
                    "title": title,
                    "content": text
                })
                logger.info(f"Saved content from {url} ({len(text)} characters)")

        # Find all local links and crawl them
        if len(visited_urls) < MAX_PAGES:
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                absolute_url = urljoin(url, href)
                # Clean URL (remove fragments)
                absolute_url = urlparse(absolute_url)._replace(fragment="").geturl()
                
                if absolute_url not in visited_urls and is_valid_url(absolute_url):
                    links.append(absolute_url)
            
            # Delay to be polite
            time.sleep(1)
            
            for link in set(links):
                crawl(link, depth + 1, max_depth)

    except Exception as e:
        logger.error(f"Error crawling {url}: {e}")

def main():
    logger.info(f"Starting crawl of {BASE_URL}")
    crawl(BASE_URL)
    
    # Save the scraped data
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Finished crawling. Scraped {len(scraped_data)} pages. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
