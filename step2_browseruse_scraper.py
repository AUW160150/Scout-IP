"""
STEP 2 HYBRID: Intelligent hybrid scraper
Traditional scraping first, Browser Use for tough cases
"""

import json
import logging
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser
import os

from config.settings import (
    SCRAPED_DIR, OPENAI_API_KEY, REQUEST_TIMEOUT,
    RATE_LIMIT_DELAY, GALILEO_API_KEY, GALILEO_PROJECT, GALILEO_LOG_STREAM, BROWSER_USE_API_KEY
)


# Galileo integration
from galileo import galileo_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridIPScraper:
    """
    Intelligent hybrid scraper:
    1. Tries traditional scraping (fast, cheap)
    2. Falls back to Browser Use for tough cases (403 errors, JS-heavy sites)
    """
    
    def __init__(self):
        # Traditional scraping session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Browser Use LLM (for fallback)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=OPENAI_API_KEY,  #openai_api_key (not api_key)
            temperature=0.3
        )
        
        # Stats tracking
        self.traditional_success = 0
        self.browseruse_success = 0
        self.total_failed = 0
        
        # Initialize Galileo
        galileo_context.init(project=GALILEO_PROJECT, log_stream=GALILEO_LOG_STREAM)
        
        logger.info("✓ Hybrid scraper initialized (Traditional + Browser Use fallback)")
    
    def scrape_traditional(self, url: str, ip_id: str) -> Optional[Dict[str, Any]]:
        """
        Traditional scraping with BeautifulSoup
        Returns None if blocked or fails
        """
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            # Check for blocks
            if response.status_code == 403:
                logger.warning(f"   ⚠️  403 Forbidden - site is blocking us")
                return None
            
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            raw_text = soup.get_text(separator='\n', strip=True)
            
            # Extract data
            data = {
                'ip_id': ip_id,
                'url': url,
                'scraped_date': datetime.now().isoformat(),
                'raw_text': raw_text,
                'scraping_method': 'traditional'
            }
            
            # Extract title
            title = soup.find('h1')
            data['title'] = title.get_text(strip=True) if title else None
            
            # Extract docket
            docket = None
            for text in soup.stripped_strings:
                if 'docket' in text.lower() and '#' in text:
                    docket = text.split('#')[-1].strip() if '#' in text else text
                    break
            data['docket'] = docket
            
            # Extract paragraphs for summary
            paragraphs = soup.find_all('p')
            summary_parts = []
            for p in paragraphs[:3]:
                text = p.get_text(strip=True)
                if len(text) > 50:
                    summary_parts.append(text)
            
            data['summary'] = summary_parts[0] if summary_parts else None
            data['abstract'] = summary_parts[1] if len(summary_parts) > 1 else None
            
            # Extract sections
            data['stage_of_development'] = self._extract_section(soup, ['stage of development', 'development stage'])
            data['applications'] = self._extract_list_section(soup, ['application', 'uses'])
            data['advantages'] = self._extract_list_section(soup, ['advantage', 'benefit'])
            
            # Extract researchers
            researchers = []
            for header in soup.find_all(['h2', 'h3', 'h4', 'strong']):
                if any(keyword in header.get_text().lower() for keyword in ['innovator', 'inventor', 'researcher']):
                    next_elem = header.find_next_sibling()
                    if next_elem:
                        if next_elem.name == 'ul':
                            researchers = [li.get_text(strip=True) for li in next_elem.find_all('li')]
                        else:
                            researchers = [next_elem.get_text(strip=True)]
                    break
            data['researchers'] = researchers
            
            # Check completeness
            completeness = self._check_completeness(data)
            data['completeness_score'] = completeness
            
            # If completeness is too low, return None to trigger Browser Use
            if completeness < 0.3:
                logger.warning(f"   ⚠️  Low completeness ({completeness:.0%}) - may need Browser Use")
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"   ⚠️  Traditional scraping failed: {e}")
            return None
    
    async def scrape_browseruse(self, url: str, ip_id: str) -> Optional[Dict[str, Any]]:
        """
        Browser Use Cloud - bypasses local browser issues
        """
        try:
            with galileo_context(project=GALILEO_PROJECT, log_stream=GALILEO_LOG_STREAM):
                logger.info(f"   🤖 Using Browser Use Cloud...")
                
                # Use Browser Cloud (no local setup needed!)
                browser = Browser(use_cloud=True)
                
                # Simpler, focused task
                task = f"""
                Go to: {url}
                
                Extract these key details about this technology:
                1. The main title/name
                2. A brief summary (1-2 sentences)
                3. Key applications or uses
                4. Any contact information
                
                Return what you find clearly.
                """
                
                agent = Agent(
                    task=task,
                    llm=self.llm,
                    browser=browser  # ← Pass the cloud browser!
                )
                
                result = await agent.run()
                
                # Extract result as text
                result_text = str(result)
                
                logger.info(f"   ✅ Browser Use Cloud succeeded!")
                
                # Structure the data
                data = {
                    'ip_id': ip_id,
                    'url': url,
                    'scraped_date': datetime.now().isoformat(),
                    'scraping_method': 'browser_use_cloud',
                    'title': f"Technology: {ip_id}",
                    'raw_text': result_text,
                    'summary': result_text[:500] if result_text else None,
                    'completeness_score': 0.6  # Cloud extraction
                }
                
                return data
                
        except Exception as e:
            logger.error(f"   ✗ Browser Use Cloud failed: {e}")
            import traceback
            traceback.print_exc()  # Show full error for debugging
            return None
    
    async def scrape_single_ip(self, url: str, ip_id: str) -> Optional[Dict[str, Any]]:
        """
        Hybrid scraping: Traditional first, Browser Use as fallback
        """
        logger.info(f"🔍 Scraping {ip_id}")
        logger.info(f"   URL: {url}")
        
        # STEP 1: Try traditional scraping first
        logger.info(f"   1️⃣  Trying traditional scraping...")
        data = self.scrape_traditional(url, ip_id)
        
        if data:
            logger.info(f"   ✅ Traditional scraping succeeded (completeness: {data['completeness_score']:.0%})")
            self.traditional_success += 1
            return data
        
        # STEP 2: Traditional failed, try Browser Use
        logger.info(f"   2️⃣  Traditional failed/blocked, using Browser Use...")
        data = await self.scrape_browseruse(url, ip_id)
        
        if data:
            logger.info(f"   ✅ Browser Use succeeded!")
            self.browseruse_success += 1
            return data
        
        # STEP 3: Both failed
        logger.error(f"   ❌ Both methods failed for {ip_id}")
        self.total_failed += 1
        return None
    
    def _extract_section(self, soup: BeautifulSoup, keywords: list) -> Optional[str]:
        """Extract text section by keywords"""
        for header in soup.find_all(['h2', 'h3', 'h4', 'strong']):
            header_text = header.get_text().lower()
            if any(kw in header_text for kw in keywords):
                next_elem = header.find_next_sibling()
                if next_elem:
                    return next_elem.get_text(strip=True)
        return None
    
    def _extract_list_section(self, soup: BeautifulSoup, keywords: list) -> list:
        """Extract bullet list section by keywords"""
        for header in soup.find_all(['h2', 'h3', 'h4', 'strong']):
            header_text = header.get_text().lower()
            if any(kw in header_text for kw in keywords):
                next_elem = header.find_next_sibling()
                if next_elem and next_elem.name == 'ul':
                    return [li.get_text(strip=True) for li in next_elem.find_all('li')]
        return []
    
    def _check_completeness(self, data: Dict[str, Any]) -> float:
        """Check what % of expected fields are populated"""
        important_fields = [
            'title', 'summary', 'abstract', 'applications', 
            'advantages', 'researchers', 'stage_of_development'
        ]
        
        filled = sum(1 for field in important_fields if data.get(field))
        return filled / len(important_fields)
    
    async def scrape_all(self, urls_file: Path, max_to_scrape: int = 10) -> list:
        """Scrape multiple IPs with hybrid approach"""
        logger.info(f"Loading URLs from {urls_file}")
        
        with open(urls_file, 'r', encoding='utf-8') as f:
            url_data = json.load(f)
        
        urls = url_data['urls'][:max_to_scrape]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"HYBRID SCRAPING: {len(urls)} IPs")
        logger.info(f"Strategy: Traditional first → Browser Use fallback")
        logger.info(f"{'='*60}\n")
        
        results = []
        
        for i, url_entry in enumerate(urls, 1):
            logger.info(f"\n[{i}/{len(urls)}] " + "="*50)
            
            data = await self.scrape_single_ip(url_entry['url'], url_entry['id'])
            
            if data:
                results.append(data)
            
            # Rate limiting
            if i < len(urls):
                await asyncio.sleep(RATE_LIMIT_DELAY)
        
        # Print stats
        logger.info(f"\n{'='*60}")
        logger.info("HYBRID SCRAPING RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total attempted: {len(urls)}")
        logger.info(f"✅ Traditional scraping: {self.traditional_success}")
        logger.info(f"🤖 Browser Use fallback: {self.browseruse_success}")
        logger.info(f"❌ Failed completely: {self.total_failed}")
        logger.info(f"📊 Success rate: {len(results)/len(urls)*100:.0f}%")
        logger.info(f"{'='*60}")
        
        # Flush Galileo
        galileo_context.flush()
        
        return results
    
    def save_results(self, results: list, source_name: str = None) -> Path:
        """Save scraped results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output = {
            'scraped_date': datetime.now().isoformat(),
            'total_count': len(results),
            'scraping_method': 'hybrid',
            'traditional_success': self.traditional_success,
            'browseruse_success': self.browseruse_success,
            'ips': results
        }
        
        if source_name:
            main_file = SCRAPED_DIR / f"hybrid_ips_{source_name}.json"
        else:
            main_file = SCRAPED_DIR / f"hybrid_ips_{timestamp}.json"
        
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Results saved to {main_file}")
        return main_file


async def run_hybrid_scraper(urls_file: Path, max_to_scrape: int = 10) -> Path:
    """Main execution for hybrid scraper"""
    logger.info("="*60)
    logger.info("HYBRID SCRAPER - Best of Both Worlds")
    logger.info("Traditional (fast) + Browser Use (intelligent)")
    logger.info("="*60)
    
    if not urls_file.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_file}")
    
    # Extract source name
    source_name = None
    if "stanford" in str(urls_file).lower():
        source_name = "stanford"
    elif "mit" in str(urls_file).lower():
        source_name = "mit"
    elif "hku" in str(urls_file).lower() or "hsk" in str(urls_file).lower():
        source_name = "hku"
    elif "nus" in str(urls_file).lower():
        source_name = "nus"
    
    scraper = HybridIPScraper()
    
    # Scrape
    results = await scraper.scrape_all(urls_file, max_to_scrape=max_to_scrape)
    
    if not results:
        raise ValueError("No IPs scraped successfully")
    
    # Save
    output_file = scraper.save_results(results, source_name)
    
    logger.info("\n✅ Hybrid scraping completed successfully!")
    
    return output_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        logger.error("❌ No URLs file specified!")
        logger.error("\nUsage: python step2_hybrid_scraper.py <urls_file> [max_to_scrape]")
        logger.error("\nExample:")
        logger.error("  python step2_hybrid_scraper.py data/raw/filtered_urls_hku.json 5")
        sys.exit(1)
    
    urls_file = Path(sys.argv[1])
    max_to_scrape = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Run hybrid scraper
    asyncio.run(run_hybrid_scraper(urls_file, max_to_scrape))