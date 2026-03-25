import requests
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urljoin, urlparse
import time

def is_valid_internal_link(url, base_domain):
    """Checks if the URL is valid and stays within the target domain."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and base_domain in parsed.netloc

def get_iitj_corpus(start_url, max_pages=100):
    base_domain = urlparse(start_url).netloc
    queue = deque([start_url])
    visited = set()
    corpus = []

    print(f"Starting crawl on: {start_url}")

    while queue and len(visited) < max_pages:
        current_url = queue.popleft()

        if current_url in visited:
            continue

        try:
            # Respect the server - add a small delay
            time.sleep(1) 
            
            response = requests.get(current_url, timeout=10)
            if response.status_code != 200:
                continue

            visited.add(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # 1. Extract and Clean Text from this page
            # We remove script and style elements so we don't get code/CSS
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            
            clean_page_text = soup.get_text(separator=' ')
            corpus.append(clean_page_text)
            
            print(f"[{len(visited)}] Scraped: {current_url}")

            # 2. Find all links to follow
            for link in soup.find_all('a', href=True):
                full_url = urljoin(current_url, link['href'])
                
                # Only add to queue if it's an IITJ link and not visited
                if is_valid_internal_link(full_url, base_domain) and full_url not in visited:
                    queue.append(full_url)

        except Exception as e:
            print(f"Error scraping {current_url}: {e}")

    return "\n".join(corpus)

# --- EXECUTION ---
seed_url = "https://indianexpress.com/about/iit-jodhpur/"
# Start with 50-100 pages to test; increase max_pages for the full million words
seeds = [
    "https://iitj.ac.in/techscape",
    "https://iitjsenateportal.vercel.app/",
    "https://www.iitj.ac.in/computer-science-engineering/en/Research-Archive"
]

for start_link in seeds:
    # Run your scraper function here
    final_text_data = get_iitj_corpus(start_link, max_pages=50)

# Save to a file
with open("iitj_corpus.txt", "a", encoding="utf-8") as f:
    f.write(final_text_data)

print(f"Done! Saved text from {len(final_text_data.split())} words.")