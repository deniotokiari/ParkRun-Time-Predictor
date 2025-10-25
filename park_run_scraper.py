from __future__ import annotations
import requests
from bs4 import BeautifulSoup
import time
import re
import argparse
from typing import List, Optional
from dataclasses import dataclass
import random

OUTPUT_FILE = "output/park_run_results.csv"
PARK_RUN_HISTORY_URI = "https://www.parkrun.pl/krakow/results/{id}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
REQUEST_DELAY_FLOOR = 30.0
REQUEST_DELAY_CELL = 60.0

@dataclass
class ParkRunResult:
    position: str
    time_seconds: str

class ParkRunScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
    
    def get_results_page(self, event_id: int) -> Optional[str]:
        url = PARK_RUN_HISTORY_URI.format(id=event_id)
        
        try:
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def parse_results(self, html_content: str) -> List[ParkRunResult]:
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        results_table = soup.find('table', class_='Results-table')
        
        if not results_table:
            print("No results table found")
            return results
        
        print(f"Found results table with class 'Results-table'")
        
        rows = results_table.find('tbody').find_all('tr') if results_table.find('tbody') else results_table.find_all('tr')[1:]
        print(f"Found {len(rows)} rows to process")
        
        for row in rows:
            try:
                cells = row.find_all('td')
                if len(cells) < 6:
                    continue
                
                position = self._extract_position(cells[0])
                time_seconds = self._extract_time(cells[5])
                
                result = ParkRunResult(
                    position=position,
                    time_seconds=time_seconds
                )
                results.append(result)
                print(f"Parsed: position={position}, time={time_seconds}s")
                    
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue
        
        return results
    
    def _extract_position(self, cell) -> str:
        try:
            text = cell.get_text(strip=True)
            position = re.sub(r'[^\d]', '', text)
            return position if position else ""
        except Exception as e:
            print(f"Error extracting position: {e}")
            return ""
    
    def _extract_time(self, cell) -> str:
        try:
            time_text = cell.find_all('div')[0].get_text(strip=True)
            parts = time_text.split(':')
            if len(parts) == 2:
                return str(int(parts[0]) * 60 + int(parts[1]))
            elif len(parts) == 3:
                return str(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
            return ""
        except Exception as e:
            print(f"Error extracting position: {e}")
            return ""
    
    def _extract_event_date(self, html_content: str) -> Optional[int]:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            date_patterns = [
                r'(\d{1,2})/(\d{1,2})/(\d{4})',
                r'(\d{1,2})-(\d{1,2})-(\d{4})',
            ]
            
            search_elements = [
                soup.find_all('span', class_=lambda x: x and 'date' in x.lower()),
                soup.find_all('div', class_=lambda x: x and 'date' in x.lower()),
                soup.find_all('h1'),
                soup.find_all('h2'),
                soup.find_all('p'),
            ]
            
            for elements in search_elements:
                for element in elements:
                    text = element.get_text()
                    for pattern in date_patterns:
                        match = re.search(pattern, text)
                        if match:
                            day, month, year = match.groups()
                            return int(month)
            
            return None
        except Exception as e:
            print(f"Error in get_last_event_id: {e}")
            return None
    
    def scrape_event(self, event_id: int) -> tuple[Optional[int], int, List[ParkRunResult]]:
        html_content = self.get_results_page(event_id)

        if not html_content:
            return None, 0, []
        
        month = self._extract_event_date(html_content)
        results = self.parse_results(html_content)
        participants = len(results)
        
        print(f"Scraped {participants} results for event {event_id} (month: {month})")
        return month, participants, results

def _initialize_csv_file(filename: str):
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['event_id', 'position', 'time', 'month', 'participants']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"📄 Initialized CSV file: {filename}")

def _append_batch_to_csv(event_id, batch_data: List[ParkRunResult], month: Optional[int], participants: int, filename: str):
    import csv
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['event_id', 'position', 'time', 'month', 'participants']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for result in batch_data:
            writer.writerow({
                'event_id': event_id, 
                'position': result.position,
                'time': result.time_seconds,
                'month': month if month is not None else "",
                'participants': participants
            })

def _get_last_event_id(csv_file: str) -> int:
    import os
    import csv

    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        return 1
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            last_event_id = 1
            
            for row in reader:
                if 'event_id' in row and row['event_id'].strip():
                    last_event_id = int(row['event_id'])
            
            return last_event_id
    except (ValueError, KeyError, FileNotFoundError):
        return 1

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Scrape ParkRun results from Krakow events',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python park_run_scraper.py                    # Run without delays (fast)
  python park_run_scraper.py --enable-delays   # Run with delays between requests
        """
    )
    
    parser.add_argument(
        '--enable-delays', 
        action='store_true',
        help='Enable delays between requests (default: disabled for faster execution)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    print("="*50)
    print("PARKRUN SCRAPER STARTED")
    print("="*50)
    print(f"Delays between requests: {'ENABLED' if args.enable_delays else 'DISABLED'}")
    print("="*50)
    
    scraper = ParkRunScraper()
    
    event_id = _get_last_event_id(OUTPUT_FILE)
    print(f"Starting from event ID: {event_id}")

    if event_id == 1:
        print("Initializing CSV file...")
        _initialize_csv_file(OUTPUT_FILE)
    else:
        event_id += 1
        print(f"Resuming from event ID: {event_id}")
    
    events_scraped = 0
    
    while True:
        print(f"\n--- Scraping Event {event_id} ---")
        
        try:
            month, participants, results = scraper.scrape_event(event_id)
            
            if results:
                print(f"Found {len(results)} results for event {event_id}")
                _append_batch_to_csv(event_id, results, month, participants, OUTPUT_FILE)
                events_scraped += 1
                event_id += 1

                if args.enable_delays:
                    delay = random.uniform(REQUEST_DELAY_FLOOR, REQUEST_DELAY_CELL)
                    print(f"Waiting {delay:.1f} seconds before next request...")
                    time.sleep(delay)
                else:
                    print("Skipping delay (disabled for faster execution)")
            else:
                print(f"No results found for event {event_id}. Stopping scraper.")
                break
                
        except Exception as e:
            print(f"Error scraping event {event_id}: {e}")
            break
    
    print(f"\nScraping completed. Processed {events_scraped} events.")
    print("="*50)