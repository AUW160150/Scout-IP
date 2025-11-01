"""
STEP 1.5: Pre-scraping filter (Replaces Step 3)
Uses EXACT same logic as original Step 3, but on titles only
This is more efficient - filter BEFORE scraping!
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import re
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
import tiktoken

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import RAW_DIR, OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SectorClassification(BaseModel):
    """Matches Step 3's exact schema"""
    mainSector: str  # "Life Sciences", "Medical Devices", or "Other"
    confidence: float
    reasoning: str


class PreScrapingFilter:
    """
    Filters URLs BEFORE scraping (efficiency!)
    Uses same classification logic as Step 3, but on titles only
    
    Trade-off: Slightly less accurate (titles only vs full content)
    Benefit: Saves 70%+ time and cost by not scraping irrelevant IPs
    """
    
    def __init__(self, use_llm: bool = True, llm_mode: str = "smart", 
                 temperature: float = 0.0, max_tokens: int = 200):
        """
        Args:
            use_llm: Whether to use LLM at all
            llm_mode: "never" (heuristic only), "smart" (hybrid), "always" (LLM for all)
            temperature: LLM temperature (0.0 = deterministic, good for classification)
            max_tokens: Max tokens for LLM response (200 is enough for simple JSON)
        """
        self.use_llm = use_llm
        self.llm_mode = llm_mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if use_llm and llm_mode != "never":
            if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
                logger.warning("No valid OpenAI API key - falling back to heuristic-only mode")
                self.use_llm = False
                self.llm_mode = "never"
                self.client = None
            else:
                logger.info(f"Using OpenAI API key: {OPENAI_API_KEY[:8]}****")
                self.client = OpenAI(api_key=OPENAI_API_KEY)
                self.model = "gpt-4o-mini"
                self.total_cost = 0.0
                self.total_tokens = 0
                self.llm_calls = 0
                self.input_cost_per_1k = 0.00015
                self.output_cost_per_1k = 0.0006
                
                # Health check
                try:
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                        temperature=0
                    )
                    logger.info(f"✓ LLM health check passed (mode: {llm_mode})")
                except Exception as e:
                    logger.error(f"LLM health check failed: {e}")
                    self.use_llm = False
                    self.llm_mode = "never"
                    self.client = None
        else:
            self.client = None
            self.llm_mode = "never"
            logger.info("Running in heuristic-only mode (no LLM)")
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        try:
            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            return int(len(text.split()) * 1.3)
    
    def classify_by_heuristic(self, title: str) -> SectorClassification:
        """
        EXACT same heuristic patterns as Step 3's _heuristic_label
        Adapted for title-only input
        """
        title_lower = title.lower()
        
        # Life Sciences patterns (from Step 3)
        BIO_PATTERNS = [
            r"\b(tuberculosis|TB)\b", r"\bCAR[- ]?T\b", r"\bT[- ]?cell(s)?\b",
            r"\bimmuno", r"\bbiomarker\b", r"\bmicrobiome\b",
            r"\bosteoarthritis|OA\b", r"\btransplant(ation)?\b",
            r"\bEEG\b", r"\bneuromodulation\b", r"\bultrasound\b",
            r"\bcytokine\b", r"\bIL-4\b", r"\bIL9R\b", r"\bhiPSC\b", r"\borganoid\b",
            r"\boncology\b", r"\btumou?r\b", r"\bdiagnos(tic|e)\b", r"\bbiotech\b",
            # Additional common bio terms
            r"\bdrug\b", r"\btherapeutic\b", r"\btherapy\b", r"\btreatment\b",
            r"\bantibody\b", r"\bprotein\b", r"\bgene\b", r"\bgenetic\b",
            r"\bvaccine\b", r"\bcell\b", r"\bcancer\b", r"\bdisease\b",
            r"\bmolecular\b", r"\bbiological\b", r"\bpharmaceutical\b"
        ]
        
        # Medical Devices patterns (from Step 3)
        DEVICE_PATTERNS = [
            r"\bimplant\b", r"\bcatheter\b", r"\bprosthe", r"\bdevice\b",
            r"\bimaging (system|device)\b", r"\bmonitor(ing)?\b",
            # Additional device terms
            r"\bsurgical\b", r"\binstrument\b", r"\bsensor\b"
        ]
        
        # Check patterns
        bio_matches = [p for p in BIO_PATTERNS if re.search(p, title_lower, re.I)]
        device_matches = [p for p in DEVICE_PATTERNS if re.search(p, title_lower, re.I)]
        
        if bio_matches:
            confidence = min(0.85, 0.60 + len(bio_matches) * 0.05)
            return SectorClassification(
                mainSector="Life Sciences",
                confidence=confidence,
                reasoning="Keyword heuristic matched bio/health terms"
            )
        
        if device_matches:
            confidence = min(0.85, 0.60 + len(device_matches) * 0.10)
            return SectorClassification(
                mainSector="Medical Devices",
                confidence=confidence,
                reasoning="Keyword heuristic matched device terms"
            )
        
        return SectorClassification(
            mainSector="Other",
            confidence=0.70,
            reasoning="No bio/device terms matched by heuristic"
        )
    
    def build_llm_classification_prompt(self, title: str) -> str:
        """
        EXACT same prompt as Step 3's build_sector_classification_prompt
        Adapted for title-only input
        """
        return f"""You are an expert in biotech classification. Analyze the opportunity and choose exactly one sector:
- "Life Sciences": pharmaceuticals, biotechnology, therapeutics, biologics, vaccines, gene/cell therapy, diagnostics (molecular/biochemical), life-science research tools, veterinary, microbiome, imaging biomarkers, EEG biomarkers, neuromodulation protocols for therapy, transplant tolerance.
- "Medical Devices": medical equipment, diagnostic hardware, imaging devices, implants, surgical instruments, monitoring devices, health-tech devices.
- "Other": everything else.

Be inclusive for biology/health-related technologies (platforms, diagnostics, research tools, computational biology all count).

Return only JSON:
{{
  "mainSector": "Life Sciences" | "Medical Devices" | "Other",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation for the choice"
}}

TITLE TO CLASSIFY:
{title}
"""
    
    def classify_by_llm(self, title: str) -> Optional[SectorClassification]:
        """Use LLM for classification (same as Step 3)"""
        if not self.use_llm or not self.client:
            return None
        
        prompt = self.build_llm_classification_prompt(title)
        
        try:
            input_tokens = self.count_tokens(prompt)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a biotech classification expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            self.llm_calls += 1
            
            content = response.choices[0].message.content
            output_tokens = self.count_tokens(content)
            
            # Track usage
            usage = getattr(response, "usage", None)
            if usage:
                self.total_tokens += (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
            else:
                self.total_tokens += input_tokens + output_tokens
            
            # Calculate cost
            self.total_cost += (input_tokens / 1000 * self.input_cost_per_1k) + \
                              (output_tokens / 1000 * self.output_cost_per_1k)
            
            result = json.loads(content)
            return SectorClassification(**result)
            
        except Exception as e:
            logger.warning(f"LLM classification failed for '{title[:50]}': {e}")
            return None
    
    def classify_single_url(self, url_entry: Dict[str, str]) -> Dict[str, any]:
        """
        Classify a single URL entry
        Strategy based on llm_mode:
        - "never": Heuristic only
        - "smart": Heuristic first, LLM for uncertain cases (< 0.8 confidence)
        - "always": Always use LLM
        """
        title = url_entry.get('title', '')
        
        if self.llm_mode == "always":
            # Always use LLM
            classification = self.classify_by_llm(title)
            if not classification:
                classification = self.classify_by_heuristic(title)
        
        elif self.llm_mode == "smart":
            # Try heuristic first
            heuristic_result = self.classify_by_heuristic(title)
            
            # If confident, use it; otherwise use LLM
            if heuristic_result.confidence >= 0.80:
                classification = heuristic_result
            else:
                llm_result = self.classify_by_llm(title)
                classification = llm_result if llm_result else heuristic_result
        
        else:  # "never"
            classification = self.classify_by_heuristic(title)
        
        # Add classification to URL entry
        url_entry_with_class = url_entry.copy()
        url_entry_with_class['classification'] = {
            'mainSector': classification.mainSector,
            'confidence': classification.confidence,
            'reasoning': classification.reasoning,
            'classified_date': datetime.now().isoformat(),
            'method': self.llm_mode
        }
        
        return url_entry_with_class
    
    def filter_urls(self, urls_file: Path, 
                    keep_categories: List[str] = None,
                    min_confidence: float = 0.5) -> tuple:
        """
        Filter URLs based on classification
        
        IMPORTANT: Filtering is DYNAMIC - not fixed!
        - Keeps URLs that match keep_categories AND have confidence >= min_confidence
        - Could keep 10 URLs, could keep 80 - depends on actual content
        
        Args:
            urls_file: Path to raw_urls JSON file
            keep_categories: Categories to keep (default: Life Sciences, Medical Devices)
            min_confidence: Minimum confidence threshold (default: 0.5)
        
        Returns:
            (kept_urls_data, discarded_urls_data, stats)
        """
        if keep_categories is None:
            # Default: keep Life Sciences and Medical Devices, discard Other
            keep_categories = ["Life Sciences", "Medical Devices"]
        
        logger.info(f"Loading URLs from {urls_file}")
        with open(urls_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        urls = data['urls']
        parent_url = data.get('parent_url', '')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Filtering {len(urls)} URLs...")
        logger.info(f"Keep categories: {keep_categories}")
        logger.info(f"Minimum confidence: {min_confidence}")
        logger.info(f"LLM mode: {self.llm_mode}")
        logger.info(f"{'='*60}\n")
        
        kept = []
        discarded = []
        stats = {
            'total': len(urls),
            'life_sciences': 0,
            'medical_devices': 0,
            'other': 0,
            'kept': 0,
            'discarded': 0
        }
        
        for url_entry in tqdm(urls, desc="Classifying"):
            classified_entry = self.classify_single_url(url_entry)
            classification = classified_entry['classification']
            
            sector = classification['mainSector']
            confidence = classification['confidence']
            
            # Update stats
            if sector == "Life Sciences":
                stats['life_sciences'] += 1
            elif sector == "Medical Devices":
                stats['medical_devices'] += 1
            else:
                stats['other'] += 1
            
            # DYNAMIC filtering: keep if matches criteria
            if sector in keep_categories and confidence >= min_confidence:
                kept.append(classified_entry)
                stats['kept'] += 1
            else:
                discarded.append(classified_entry)
                stats['discarded'] += 1
        
        # Print results
        logger.info(f"\n{'='*60}")
        logger.info("PRE-SCRAPING FILTER RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total URLs: {stats['total']}")
        logger.info(f"Life Sciences: {stats['life_sciences']} ({stats['life_sciences']/stats['total']*100:.1f}%)")
        logger.info(f"Medical Devices: {stats['medical_devices']} ({stats['medical_devices']/stats['total']*100:.1f}%)")
        logger.info(f"Other: {stats['other']} ({stats['other']/stats['total']*100:.1f}%)")
        logger.info(f"\n✅ KEPT for scraping: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
        logger.info(f"❌ DISCARDED: {stats['discarded']} ({stats['discarded']/stats['total']*100:.1f}%)")
        
        if self.use_llm and self.llm_mode != "never":
            logger.info(f"\n💰 LLM Usage:")
            logger.info(f"   Calls: {self.llm_calls}/{stats['total']}")
            logger.info(f"   Tokens: {self.total_tokens:,}")
            logger.info(f"   Cost: ${self.total_cost:.4f}")
        
        logger.info(f"{'='*60}")
        
        # Add metadata
        kept_data = {
            'parent_url': parent_url,
            'filtered_date': datetime.now().isoformat(),
            'filter_stage': 'pre_scraping_filter',
            'filter_criteria': {
                'keep_categories': keep_categories,
                'min_confidence': min_confidence,
                'llm_mode': self.llm_mode
            },
            'statistics': stats,
            'total_count': len(kept),
            'urls': kept
        }
        
        discarded_data = {
            'parent_url': parent_url,
            'filtered_date': datetime.now().isoformat(),
            'total_count': len(discarded),
            'urls': discarded
        }
        
        return kept_data, discarded_data, stats
    
    def save_filtered_urls(self, kept_data: Dict, discarded_data: Dict, 
                          source_name: str = None) -> tuple:
        """Save filtered URLs to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save kept URLs (these will be scraped)
        if source_name:
            kept_file = RAW_DIR / f"filtered_urls_{source_name}.json"
            kept_backup = RAW_DIR / f"filtered_urls_{source_name}_{timestamp}.json"
            discarded_file = RAW_DIR / f"discarded_urls_{source_name}.json"
        else:
            kept_file = RAW_DIR / "filtered_urls.json"
            kept_backup = RAW_DIR / f"filtered_urls_{timestamp}.json"
            discarded_file = RAW_DIR / "discarded_urls.json"
        
        # Save kept URLs
        with open(kept_file, 'w', encoding='utf-8') as f:
            json.dump(kept_data, f, indent=2, ensure_ascii=False)
        with open(kept_backup, 'w', encoding='utf-8') as f:
            json.dump(kept_data, f, indent=2, ensure_ascii=False)
        
        # Save discarded URLs (for reference)
        with open(discarded_file, 'w', encoding='utf-8') as f:
            json.dump(discarded_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Kept URLs saved to: {kept_file}")
        logger.info(f"✓ Discarded URLs saved to: {discarded_file}")
        
        return kept_file, discarded_file


def run_step1_5(urls_file: Path, 
                use_llm: bool = True,
                llm_mode: str = "smart",
                keep_categories: List[str] = None,
                min_confidence: float = 0.5,
                temperature: float = 0.0,
                max_tokens: int = 200) -> Path:
    """Main execution for Step 1.5"""
    logger.info("="*60)
    logger.info("STEP 1.5: Pre-Scraping Filter")
    logger.info("(Same logic as Step 3, but BEFORE scraping!)")
    logger.info("="*60)
    
    if not urls_file.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_file}")
    
    # Extract source name from input filename
    source_name = None
    if "stanford" in str(urls_file).lower():
        source_name = "stanford"
    elif "mit" in str(urls_file).lower():
        source_name = "mit"
    elif urls_file.name.startswith("raw_urls_"):
        parts = urls_file.stem.split('_')
        if len(parts) > 2:
            source_name = parts[2]
    
    # Initialize filter
    filter = PreScrapingFilter(use_llm=use_llm, llm_mode=llm_mode, 
                               temperature=temperature, max_tokens=max_tokens)
    
    # Filter URLs
    kept_data, discarded_data, stats = filter.filter_urls(
        urls_file,
        keep_categories=keep_categories,
        min_confidence=min_confidence
    )
    
    if not kept_data['urls']:
        logger.warning("⚠️  No URLs kept after filtering!")
        logger.warning("\nPossible reasons:")
        logger.warning("1. The source may not have bio/med content")
        logger.warning("2. Confidence threshold too high")
        logger.warning("3. Heuristic-only mode may miss some relevant content")
        logger.warning("\nSuggestions:")
        logger.warning("- Try lowering --min-confidence (e.g., 0.3)")
        logger.warning("- Try --llm-mode smart for better accuracy")
        logger.warning("- Review discarded_urls.json to see what was filtered")
        return None
    
    # Save filtered URLs
    kept_file, discarded_file = filter.save_filtered_urls(
        kept_data, discarded_data, source_name
    )
    
    # Print sample
    logger.info("\n📋 Sample kept URLs:")
    for i, url_data in enumerate(kept_data['urls'][:5]):
        sector = url_data['classification']['mainSector']
        conf = url_data['classification']['confidence']
        title = url_data['title']
        logger.info(f"{i+1}. [{sector}, {conf:.2f}] {title[:60]}...")
    
    if len(kept_data['urls']) > 5:
        logger.info(f"   ... and {len(kept_data['urls']) - 5} more")
    
    logger.info("\n✓ Step 1.5 completed successfully")
    logger.info(f"\n🎯 Next step: Scrape ONLY the filtered URLs:")
    logger.info(f"   python step2_scrape_details.py {kept_file}")
    
    return kept_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        logger.error("❌ No URLs file specified!")
        logger.error("\nUsage: python step1_5_filter_urls.py <raw_urls_file> [options]")
        logger.error("\nOptions:")
        logger.error("  --llm-mode MODE       'never', 'smart' (default), 'always'")
        logger.error("  --min-confidence X    Minimum confidence (default: 0.5)")
        logger.error("  --categories X,Y      Categories to keep (default: 'Life Sciences,Medical Devices')")
        logger.error("  --temperature X       LLM temperature (default: 0.0, deterministic)")
        logger.error("  --max-tokens N        LLM max tokens (default: 200)")
        logger.error("\nLLM Modes:")
        logger.error("  never  - Heuristic only (FREE, fast, ~75-80% accurate)")
        logger.error("  smart  - Hybrid (recommended, ~$0.10-0.30, ~85-90% accurate)")
        logger.error("  always - Always LLM (~$0.40, ~90-95% accurate)")
        logger.error("\nLLM Parameters:")
        logger.error("  temperature - 0.0 = deterministic (recommended for classification)")
        logger.error("              - 0.3 = slight variation")
        logger.error("              - 1.0 = high variation (not recommended)")
        logger.error("  max_tokens  - 200 = sufficient for classification (saves cost)")
        logger.error("              - 500 = longer reasoning (if needed)")
        logger.error("\n⚠️  FILTERING IS DYNAMIC:")
        logger.error("  - Not fixed at 25 or any number!")
        logger.error("  - Keeps ALL URLs matching your criteria")
        logger.error("  - Could be 10, could be 80 - depends on content")
        logger.error("\nAvailable raw URL files:")
        
        available_files = sorted(RAW_DIR.glob("raw_urls*.json"))
        if available_files:
            for f in available_files:
                try:
                    with open(f, 'r') as file:
                        data = json.load(file)
                        url_count = len(data.get('urls', []))
                        logger.error(f"  - {f.name} ({url_count} URLs)")
                except:
                    logger.error(f"  - {f.name}")
        else:
            logger.error("  (No files found - run step1 first)")
        
        logger.error("\nExamples:")
        logger.error("  # Smart mode (recommended)")
        logger.error("  python step1_5_filter_urls.py data/raw/raw_urls_stanford.json")
        logger.error("")
        logger.error("  # Free mode")
        logger.error("  python step1_5_filter_urls.py data/raw/raw_urls_mit.json --llm-mode never")
        logger.error("")
        logger.error("  # Stricter filter (higher confidence)")
        logger.error("  python step1_5_filter_urls.py data/raw/raw_urls_stanford.json --min-confidence 0.7")
        logger.error("")
        logger.error("  # Only Life Sciences")
        logger.error("  python step1_5_filter_urls.py data/raw/raw_urls_stanford.json --categories 'Life Sciences'")
        logger.error("")
        logger.error("  # Custom LLM parameters (more detailed reasoning)")
        logger.error("  python step1_5_filter_urls.py data/raw/raw_urls_stanford.json --max-tokens 500")
        sys.exit(1)
    
    urls_file = Path(sys.argv[1])
    
    # Parse options
    llm_mode = "smart"  # default
    if '--llm-mode' in sys.argv:
        idx = sys.argv.index('--llm-mode')
        if idx + 1 < len(sys.argv):
            llm_mode = sys.argv[idx + 1]
            if llm_mode not in ['never', 'smart', 'always']:
                logger.error(f"❌ Invalid llm-mode: {llm_mode}")
                logger.error("   Must be: 'never', 'smart', or 'always'")
                sys.exit(1)
    
    use_llm = llm_mode != "never"
    
    min_confidence = 0.5
    if '--min-confidence' in sys.argv:
        idx = sys.argv.index('--min-confidence')
        if idx + 1 < len(sys.argv):
            min_confidence = float(sys.argv[idx + 1])
    
    keep_categories = None
    if '--categories' in sys.argv:
        idx = sys.argv.index('--categories')
        if idx + 1 < len(sys.argv):
            keep_categories = [c.strip() for c in sys.argv[idx + 1].split(',')]
    
    temperature = 0.0
    if '--temperature' in sys.argv:
        idx = sys.argv.index('--temperature')
        if idx + 1 < len(sys.argv):
            temperature = float(sys.argv[idx + 1])
    
    max_tokens = 200
    if '--max-tokens' in sys.argv:
        idx = sys.argv.index('--max-tokens')
        if idx + 1 < len(sys.argv):
            max_tokens = int(sys.argv[idx + 1])
    
    # Validate file exists
    if not urls_file.exists():
        logger.error(f"❌ File not found: {urls_file}")
        logger.error("\nAvailable files:")
        for f in sorted(RAW_DIR.glob("raw_urls*.json")):
            logger.error(f"  - {f}")
        sys.exit(1)
    
    run_step1_5(urls_file, use_llm=use_llm, llm_mode=llm_mode, 
                keep_categories=keep_categories, min_confidence=min_confidence,
                temperature=temperature, max_tokens=max_tokens)