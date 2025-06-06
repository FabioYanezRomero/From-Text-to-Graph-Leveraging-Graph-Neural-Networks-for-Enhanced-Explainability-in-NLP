import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaAIScraper:
    BASE_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    IMAGE_DIR = "wikipedia_ai_images"
    TABLE_DIR = "wikipedia_ai_tables"
    TEXT_FILE = "wikipedia_ai_text.txt"

    def __init__(self):
        os.makedirs(self.IMAGE_DIR, exist_ok=True)
        os.makedirs(self.TABLE_DIR, exist_ok=True)

    def fetch_page(self):
        logger.info(f"Fetching Wikipedia page: {self.BASE_URL}")
        try:
            response = requests.get(self.BASE_URL)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch page: {e}")
            raise

    def parse_soup(self, html):
        return BeautifulSoup(html, 'html.parser')

    def extract_text(self, soup):
        logger.info("Extracting text content...")
        content = soup.find('div', {'class': 'mw-parser-output'})
        paragraphs = content.find_all('p', recursive=False)
        text = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        with open(self.TEXT_FILE, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved text content to {self.TEXT_FILE}")

    def extract_images(self, soup):
        logger.info("Extracting images...")
        content = soup.find('div', {'class': 'mw-parser-output'})
        images = content.find_all('img')
        for img in images:
            src = img.get('src')
            if not src:
                continue
            if src.startswith('//'):
                src = 'https:' + src
            elif src.startswith('/'):
                src = urljoin(self.BASE_URL, src)
            img_name = os.path.basename(urlparse(src).path)
            img_path = os.path.join(self.IMAGE_DIR, img_name)
            try:
                img_resp = requests.get(src, timeout=10)
                img_resp.raise_for_status()
                with open(img_path, 'wb') as f:
                    f.write(img_resp.content)
                logger.info(f"Saved image: {img_path}")
            except Exception as e:
                logger.warning(f"Failed to download image {src}: {e}")

    def extract_tables(self, soup):
        logger.info("Extracting tables...")
        tables = soup.find_all('table', {'class': 'wikitable'})
        for idx, table in enumerate(tables, 1):
            try:
                df = pd.read_html(str(table))[0]
                table_path = os.path.join(self.TABLE_DIR, f"table_{idx}.csv")
                df.to_csv(table_path, index=False)
                logger.info(f"Saved table to {table_path}")
            except Exception as e:
                logger.warning(f"Failed to parse/save table {idx}: {e}")

    def run(self):
        html = self.fetch_page()
        soup = self.parse_soup(html)
        self.extract_text(soup)
        self.extract_images(soup)
        self.extract_tables(soup)
        logger.info("Scraping completed.")

if __name__ == "__main__":
    scraper = WikipediaAIScraper()
    scraper.run()
