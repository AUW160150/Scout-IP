

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedUniversalExtractor:
    """
    Properly extracts technologies from all pages
    Excludes pagination links from results
    """
    
    def __init__(self, parent_url: str, max_ips: int = None, max_pages: int = 50):
        raw = urlparse(parent_url)
        self.base_domain = raw.netloc
        self.parent_url = self._normalize_url(parent_url)
        self.base_domain = urlparse(self.parent_url).netloc
        self.max_ips = max_ips
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Track what we've found
        self.pagination_urls = []
        self.technology_urls = []
        self.seen_urls = set()

    def _normalize_url(self, u: str) -> str:
        p = urlparse(u)
        scheme = p.scheme or 'https'
        netloc = p.netloc or self.base_domain     
        path = re.sub(r'/+', '/', p.path or '/')
        query = '&'.join(
            kv for kv in (p.query or '').split('&')
            if kv and not kv.lower().startswith(('utm_', 'fbclid='))
        )
        return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")



    def _same_org(self, netloc: str) -> bool:
        # treat foo.hku.hk and tto.hku.hk as same org
        parts = netloc.split('.')
        base = '.'.join(parts[-2:]) if len(parts) >= 2 else netloc
        base_self = '.'.join(self.base_domain.split('.')[-2:])
        return base == base_self
    
    def _harvest_nus_show_more(self, category_url: str) -> List[Dict[str, str]]:
        """
        For category pages that use 'Show More' ( /search-load-more ),
        iterate hidden_offset until no more items are returned.
        Returns a list of technology dicts {id,url,title}.
        """
        out = []
        base = f"{urlparse(self.parent_url).scheme}://{urlparse(self.parent_url).netloc}"

        # parse existing params to keep filters consistent
        P = urlparse(category_url)
        from urllib.parse import parse_qsl, urlencode
        base_params = dict(parse_qsl(P.query))

        offset = 0
        seen_fragment = set()
        session = self.session

        while True:
            params = base_params.copy()
            params['hidden_offset'] = str(offset)
            load_more_url = self._normalize_url(f"{base}/search-load-more?{urlencode(params, doseq=True)}")
            try:
                r = session.get(load_more_url, timeout=30)
                if r.status_code != 200:
                    break
                html = r.text.strip()
                # stop if empty or repeated
                sig = (offset, len(html))
                if not html or sig in seen_fragment:
                    break
                seen_fragment.add(sig)

                frag = BeautifulSoup(html, 'html.parser')
                for a in frag.find_all('a', href=True):
                    href = a['href']
                    full = self._normalize_url(urljoin(category_url, href))
                    if self._is_technology_url(href, full) and full not in self.seen_urls:
                        title = self._extract_title(a, frag, href)
                        out.append({'id': self._extract_id(full), 'url': full, 'title': title})
                        self.seen_urls.add(full)

                # heuristic: NUS loads 9 per click; increase by 9
                # if the fragment contained < 1 anchor, we likely reached the end
                if len([1 for _ in frag.find_all('a', href=True)]) == 0:
                    break

                offset += 9
                time.sleep(0.5)
            except Exception:
                break

        return out


    def _is_listing_url(self, href: str, full_url: str) -> bool:
        if not self._same_org(urlparse(full_url).netloc):
            return False

        p = urlparse(full_url)
        path = p.path.lower()
        q = p.query.lower()

        # Known hubs + paginated hubs
        if (
            '/categories/' in path or
            path.endswith('/technologies') or
            '/available-technologies' in path or
            re.search(r'/categories/[^?#]*/page/\d+/?$', path) is not None
        ):
            return True

        # NUS category and “show more” endpoints
        if path == '/search/category' and 'category_id=' in q:
            return True
        if path == '/search-load-more':
            return True

        # Classic ?page= pagination on tech listings
        if (('/technology' in path) or ('/technologies' in path)) and 'page=' in q:
            return True

        # Generic search/list filters
        if path.startswith('/search') and any(k in q for k in ['category=', 'category_id=', 'tag=', 'topic=', 'area=']):
            return True

        return False


    def extract_urls(self) -> List[Dict[str, str]]:
        """Main extraction process"""
        logger.info("="*60)
        logger.info("Fixed Universal Extractor - Starting")
        logger.info("="*60)
        logger.info(f"Target: {self.parent_url}")
        
        # Step 1: Analyze first page to find pagination
        self._discover_pagination()
        
        # Step 2: Extract from all pages
        all_technologies = self._extract_from_all_pages()
        
        logger.info(f"\n✅ Extraction complete: {len(all_technologies)} technologies found")
        return all_technologies
    
    def _discover_pagination(self):
        """Discover all pagination URLs from the first page"""
        logger.info("\n🔍 Discovering pagination...")
        
        try:
            response = self.session.get(self.parent_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links
            all_links = soup.find_all('a', href=True)
            
            pagination_patterns = []
            page_urls = {}  # page_number -> url
            
            for link in all_links:
                href = link['href']
                full_url = self._normalize_url(urljoin(self.parent_url, href))
                text = link.get_text(strip=True)
                
                # Check for pagination patterns
                
                # Pattern 1: /pageN or /page/N or /page-N
                page_match = re.search(r'/page[-/]?(\d+)/?$', href)
                if page_match:
                    page_num = int(page_match.group(1))
                    page_urls[page_num] = full_url
                    logger.debug(f"Found page link: {page_num} -> {full_url}")
                
                # Pattern 2: ?page=N or &page=N
                elif '?page=' in href or '&page=' in href:
                    page_match = re.search(r'[?&]page=(\d+)', href)
                    if page_match:
                        page_num = int(page_match.group(1))
                        page_urls[page_num] = full_url
                        logger.debug(f"Found page param: {page_num} -> {full_url}")
                
                # Pattern 3: Link text is just a number
                elif text.isdigit() and 1 <= int(text) <= 100:
                    page_num = int(text)
                    page_urls[page_num] = full_url
                    logger.debug(f"Found numbered link: {page_num} -> {full_url}")
                
                # Pattern 4: Next/Previous indicators
                elif text.lower() in ['next', 'previous', 'prev', '»', '›', '‹', '«']:
                    pagination_patterns.append(full_url)
            
            # Build ordered list of pages to visit
            if page_urls:
                # Sort by page number
                max_page = max(page_urls.keys())
                logger.info(f"✓ Found {len(page_urls)} page links (up to page {max_page})")
                
                # Create list of URLs to visit
                self.pagination_urls = [self.parent_url]  # Start with page 1
                for page_num in sorted(page_urls.keys()):
                    if page_num > 1 and page_num <= self.max_pages:
                        self.pagination_urls.append(page_urls[page_num])
                
                logger.info(f"📑 Will visit {len(self.pagination_urls)} pages")
            else:
                # No pagination found - single page site
                logger.info("No pagination detected - single page site")
                self.pagination_urls = [self.parent_url]
                
        except Exception as e:
            logger.error(f"Error discovering pagination: {e}")
            self.pagination_urls = [self.parent_url]
    
    def _extract_from_all_pages(self) -> List[Dict[str, str]]:
        all_technologies = []
        idx = 0

        # pagination_urls grows as we enqueue more listing/category pages
        while idx < len(self.pagination_urls):
            page_url = self.pagination_urls[idx]

            if self.max_ips and len(all_technologies) >= self.max_ips:
                logger.info(f"Reached max IPs limit ({self.max_ips})")
                break

            logger.info(f"\n📄 Processing page {idx+1}/{len(self.pagination_urls)}")
            logger.info(f"   URL: {page_url}")

            page_technologies = self._extract_from_single_page(page_url)

            if page_technologies:
                all_technologies.extend(page_technologies)
                logger.info(f"   ✓ Found {len(page_technologies)} technologies (total: {len(all_technologies)})")
            else:
                logger.info(f"   - No new technologies on this page")

            self.seen_urls.add(page_url)
            idx += 1
            if idx < len(self.pagination_urls):
                time.sleep(1)

        return all_technologies

    
    def _extract_from_single_page(self, page_url: str) -> List[Dict[str, str]]:
        """Extract technology URLs from a single page"""
        technologies = []
        
        try:
            response = self.session.get(page_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links
            all_links = soup.find_all('a', href=True)

            # If current page is a category, harvest its “show more” pages now
            try:
                pth = urlparse(page_url).path.lower()
                qry = urlparse(page_url).query.lower()
                if pth == '/search/category' and 'category_id=' in qry:
                    extra = self._harvest_nus_show_more(page_url)
                    if extra:
                        technologies.extend(extra)
                        logger.info(f"   + Harvested {len(extra)} via 'Show More' API")
            except Exception:
                pass

            
            for link in all_links:
                href = link['href']
                full_url = self._normalize_url(urljoin(page_url, href))
                # Skip if already seen
                if full_url in self.seen_urls:
                    continue
                
                # NEW: enqueue more listing pages (BFS-like)
                if self._is_listing_url(href, full_url) and full_url not in self.pagination_urls:
                    self.pagination_urls.append(full_url)

                # Check if this is a technology URL (not pagination)
                if self._is_technology_url(href, full_url):
                    title = self._extract_title(link, soup, href)
                    tech_id = self._extract_id(full_url)
                    
                    technologies.append({'id': tech_id,'url': full_url,'title': title})
                    self.seen_urls.add(full_url)
                    logger.debug(f"Found: {title[:50]}...")
            
            return technologies
            
        except Exception as e:
            logger.error(f"Error extracting from page {page_url}: {e}")
            return []
    
    def _is_technology_url(self, href: str, full_url: str) -> bool:
        # Same organization (allow subdomains like versitech.hku.hk for hku.hk)
        if not self._same_org(urlparse(full_url).netloc):
            return False

        href_lower = href.lower()

        # exclude NUS ajax/listing endpoints outright
        if '/search-load-more' in href_lower or (href_lower.startswith('/search/category') or 'search/category' in href_lower):
            return False

        # EXCLUDE: obvious non-detail links
        exclude_patterns = [
            r'/page\d+/?$', r'/page/\d+/?$', r'[?&]page=\d+',
            r'^/$', r'^#', r'^javascript:', r'^mailto:',
            r'/about/?$', r'/contact/?$', r'/search/?$', r'/login/?$', r'/register/?$',
            r'/news/?$', r'/events/?$', r'/blog/?$',
            r'\.(pdf|jpg|jpeg|png|gif|doc|docx|xls|xlsx|zip)$'
        ]
        for pattern in exclude_patterns:
            if re.search(pattern, href_lower):
                return False

        # POSITIVE: common tech paths (Stanford/MIT/others)
        tech_indicators = [
            '/technology/', '/technologies/', '/tech/',
            '/patent/', '/patents/', '/invention/', '/inventions/',
            '/innovation/', '/innovations/', '/ip/', '/portfolio/', '/disclosure/',
            '/product/',                           # NUS product route
            '/available-technologies/'            # MIT listing & details
        ]
        if any(ind in href_lower for ind in tech_indicators):
            last = href_lower.rstrip('/').split('/')[-1]
            if re.search(r'[a-z0-9]', last) and not re.match(r'^page\d+$', last):
                return True

        # POSITIVE (fallback): NUS-style root-level slugs
        # e.g. /advanced-food-image-recognition-technology-deep-learning
        parsed = urlparse(href_lower)
        path = parsed.path.strip('/')

        nav_blacklist = {
            'about-us', 'contact-us', 'faq', 'login', 'sign-up',
            'success-stories', 'privacy-policy', 'cookies-policy',
            'categories', 'legal-information-notices'
        }

        # depth==1, not a nav page, looks like a descriptive slug (>= 2 hyphens)
        if path and '/' not in path and path not in nav_blacklist:
            if path.count('-') >= 2 and re.search(r'[a-z]', path) and not path.isdigit():
                return True

        return False

    
    def _extract_title(self, link_elem, soup, href: str) -> str:
        """Extract title with multiple fallback strategies"""

        bad_labels = {'read more','view','details','link','show more','load more','more'}

        # Strategy 1: Link text
        title = link_elem.get_text(strip=True)
        if title and len(title) > 3 and title.lower() not in bad_labels:
            return title
        
        # Strategy 2: Title attribute
        title = link_elem.get('title', '').strip()
        if title and len(title) > 3:
            return title
        
        # Strategy 3: Aria label
        title = link_elem.get('aria-label', '').strip()
        if title and len(title) > 3:
            return title
        
        # Strategy 4: Look for heading in parent
        parent = link_elem.parent
        for _ in range(3):  # Look up to 3 levels
            if parent:
                heading = parent.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if heading and heading != link_elem:
                    title = heading.get_text(strip=True)
                    if title and len(title) > 3:
                        return title
                parent = parent.parent
        
        # Strategy 5: Convert URL slug to title
        parts = href.rstrip('/').split('/')
        if parts:
            last_part = parts[-1]
            # Remove file extensions
            last_part = re.sub(r'\.[a-z]+$', '', last_part)
            # Convert hyphens/underscores to spaces
            title = last_part.replace('-', ' ').replace('_', ' ')
            # Capitalize
            return title.title()

        return "No title found"
    
    def _extract_id(self, url: str) -> str:
        """Extract ID from URL"""
        parts = url.rstrip('/').split('/')
        
        # Work backwards to find the ID
        for part in reversed(parts):
            # Skip common folder names
            if part.lower() in ['technology', 'technologies', 'tech', 'patent', 'patents',
                    'invention', 'inventions', 'innovation', 'innovations',
                    'portfolio', 'disclosure', 'product', 'available-technologies']:
                continue
            
            # Skip page indicators
            if re.match(r'^page\d+$', part):
                continue
            
            # This looks like an ID
            if part and re.search(r'[a-z0-9]', part, re.I):
                return part
        
        return "unknown"
    
    def save_results(self, urls: List[Dict[str, str]], output_dir: Path = None) -> Path:
        """Save results to JSON file"""
        if output_dir is None:
            output_dir = Path('data/raw')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = urlparse(self.parent_url).netloc
        clean_name = domain.replace('www.', '').split('.')[0]
        
        output = {
            'parent_url': self.parent_url,
            'extracted_date': datetime.now().isoformat(),
            'total_count': len(urls),
            'pages_processed': len(self.pagination_urls),
            'urls': urls
        }
        
        output_file = output_dir / f"raw_urls_{clean_name}.json"
        backup_file = output_dir / f"raw_urls_{clean_name}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Saved to: {output_file}")
        return output_file


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("FIXED UNIVERSAL EXTRACTOR")
        print("="*60)
        print("\nUsage: python fixed_extractor.py <URL> [OPTIONS]")
        print("\nOptions:")
        print("  --max-ips N     Maximum technologies to extract")
        print("  --max-pages N   Maximum pages to process (default: 50)")
        print("\nExamples:")
        print("  python fixed_extractor.py https://www.tto.hku.hk/technology")
        print("  python fixed_extractor.py https://techfinder.stanford.edu/")
        print("  python fixed_extractor.py https://tlo.mit.edu/technologies")
        sys.exit(1)
    
    url = sys.argv[1]
    max_ips = None
    max_pages = 50
    
    if '--max-ips' in sys.argv:
        idx = sys.argv.index('--max-ips')
        if idx + 1 < len(sys.argv):
            max_ips = int(sys.argv[idx + 1])
    
    if '--max-pages' in sys.argv:
        idx = sys.argv.index('--max-pages')
        if idx + 1 < len(sys.argv):
            max_pages = int(sys.argv[idx + 1])
    
    # Run extraction
    extractor = FixedUniversalExtractor(url, max_ips=max_ips, max_pages=max_pages)
    technologies = extractor.extract_urls()
    
    if technologies:
        output_file = extractor.save_results(technologies)
        
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Technologies found: {len(technologies)}")
        print(f"Pages processed: {len(extractor.pagination_urls)}")
        print(f"\nSample results:")
        for i, tech in enumerate(technologies[:5], 1):
            print(f"  {i}. {tech['title'][:60]}...")
        if len(technologies) > 5:
            print(f"  ... and {len(technologies)-5} more")
    else:
        print("\n❌ No technologies found")


if __name__ == "__main__":
    main()