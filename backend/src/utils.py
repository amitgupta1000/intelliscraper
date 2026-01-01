# utils.py
import re
import json
from .logging_setup import logger
import asyncio
import time # Added time for cache timestamp
# Added readability for main content extraction
# Define SearchResult and PDF classes here for simplicity or ensure they are imported
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from .config_minimal import GCS_BUCKET
except Exception:
    GCS_BUCKET = ""

@dataclass
class SearchResult:
    """
    Dataclass to store individual search results.
    """
    url: str
    title: str
    snippet: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source
        }

# Utility function to safely format prompts with content that may contain curly braces
def safe_format(template: str, **kwargs: Any) -> str:
    """
    Safely format a template string, escaping any curly braces in the values.
    This prevents ValueError when content contains unexpected curly braces.
    """
    # Escape any curly braces in the values
    safe_kwargs = {k: v.replace('{', '{{').replace('}', '}}') if isinstance(v, str) else v
                  for k, v in kwargs.items()}
    return template.format(**safe_kwargs)

# Get current date in a readable format
from datetime import datetime
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


# Helper function to clean extracted text
def clean_extracted_text(text: str) -> str:
    """Cleans extracted text by removing extra whitespaces, image files, and boilerplate text."""
    if text is None:
        return ""
    # If incoming text contains HTML, try to extract visible text first
    try:
        from bs4 import BeautifulSoup
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            text = soup.get_text(separator='\n', strip=True)
    except Exception:
        # bs4 not available or parsing failed; fall back to regex removal
        text = re.sub(r'<[^>]+>', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove common boilerplate patterns (can be expanded)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)  # Remove HTML comments
    text = re.sub(r'<(script|style).*?>.*?</(script|style)>', '', text, flags=re.DOTALL | re.IGNORECASE) # Remove script and style tags
    text = re.sub(r'\[document\][^\]]*\]', '', text) # Remove patterns like [document]...
    text = re.sub(r'\(Source:.*?\)', '', text) # Remove source citations if present from previous steps

    # Remove navigation menus and footers (BBC/news style)
    boilerplate_patterns = [
        # BBC/news and generic navigation
        r'(Home\s+News\s+Sport\s+Business\s+Innovation\s+Culture\s+Arts\s+Travel\s+Earth\s+Audio\s+Video\s+Live)+',
        r'(Home\s+News\s+Israel-Gaza War\s+War in Ukraine\s+US & Canada\s+UK\s+UK Politics\s+England\s+N\. Ireland\s+N\. Ireland Politics\s+Scotland\s+Scotland Politics\s+Wales\s+Wales Politics\s+Africa\s+Asia\s+China\s+India\s+Australia\s+Europe\s+Latin America\s+Middle East\s+In Pictures\s+BBC InDepth\s+BBC Verify\s+Sport\s+Business\s+Executive Lounge\s+Technology of Business\s+Future of Business\s+Innovation\s+Technology\s+Science & Health\s+Artificial Intelligence\s+AI v the Mind\s+Culture\s+Film & TV\s+Music\s+Art & Design\s+Style\s+Books\s+Entertainment News\s+Arts\s+Arts in Motion\s+Travel\s+Destinations\s+Africa\s+Antarctica\s+Asia\s+Australia and Pacific\s+Caribbean & Bermuda\s+Central America\s+Europe\s+Middle East\s+North America\s+South America\s+World’s Table\s+Culture & Experiences\s+Adventures\s+The SpeciaList\s+Earth\s+Natural Wonders\s+Weather & Science\s+Climate Solutions\s+Sustainable Business\s+Green Living\s+Audio\s+Podcast Categories\s+Radio\s+Audio FAQs\s+Video\s+BBC Maestro\s+Live\s+Live News\s+Live Sport\s+Home\s+News\s+Sport\s+Business\s+Innovation\s+Culture\s+Arts\s+Travel\s+Earth\s+Audio\s+Video\s+Live\s+Weather\s+Newsletters)+',
        # Social media and newsletter
        r'Follow (us )?on (Facebook|Twitter|Instagram|LinkedIn|YouTube)',
        r'Subscribe to our newsletter',
        r'Sign up for our newsletter',
        r'Get updates',
        r'Enter your email',
        r'Subscribe',
        # Copyright/legal
        r'All rights reserved',
        r'© \d{4}',
        r'Copyright \d{4}',
        r'Terms and Conditions',
        r'Legal Disclaimer',
        r'Site Map',
        r'Terms of Use',
        r'Subscription Terms',
        r'Privacy Policy',
        r'Cookies Policy',
        r'Privacy Notice',
        r'Accessibility Statement',
        r'Manage Preferences',
        r'Do Not Sell My Info',
        r'About the BBC',
        r'BBC Shop',
        r'BritBox',
        r'Read the BBC In your own language',
        r'International Business',
        r'Artificial intelligence',
        r'Technology of Business',
        r'More from the BBC',
        r'British Broadcasting Corporation',
        # Contact/support
        r'Contact us',
        r'Customer Service',
        r'Support',
        r'Help Center',
        r'FAQ',
        # Navigation
        r'Back to top',
        r'Next article',
        r'Previous article',
        r'Related articles',
        r'Recommended for you',
        # Miscellaneous
        r'Advertise with us',
        r'Sponsored content',
        r'Advertisement',
        r'AdChoices',
        r'Powered by',
        # App/download
        r'Download our app',
        r'Get the app',
        r'Available on the App Store',
        r'Get it on Google Play',
        # Section headers
        r'Latest News',
        r'Trending',
        r'Most Read',
        r"Editor's Picks",
        r'Top Stories',
        r'Popular Now',
        # Language selectors
        r'Select Language',
        r'Choose Language',
        r'English',
        r'Español',
        r'Français',
        r'Deutsch',
        r'中文',
        r'日本語',
        r'한국어',
        r'Русский',
        r'Português',
        r'Italiano',
        r'हिन्दी',
        r'العربية',
        r'ภาษาไทย',
        r'Türkçe',
        r'Polski',
        r'Nederlands',
        r'Ελληνικά',
        r'Українська',
        r'עברית',
        r'فارسی',
        r'বাংলা',
        r'தமிழ்',
        r'తెలుగు',
        r'ગુજરાતી',
        r'ਪੰਜਾਬੀ',
        r'മലയാളം',
        r'සිංහල',
        r'ಕನ್ನಡ',
        r'मराठी',
        r'ଓଡ଼ିଆ',
        r'اردو',
        r'தமிழ்',
        r'తెలుగు',
        r'ગુજરાતી',
        r'ਪੰਜਾਬੀ',
        r'മലയാളം',
        r'සිංහල',
        r'ಕನ್ನಡ',
        r'मराठी',
        r'ଓଡ଼ିଆ',
        r'اردو',
        # BBC/news repeated
        r'\bRelated\b',
        r'\bShare\b',
        r'\bSave\b',
        r'\bGetty Images\b',
        r'\bAudio FAQs\b',
        r'\bLive News\b',
        r'\bLive Sport\b',
        r'\bWeather\b',
        r'\bNewsletters\b',
        r'\bBBC Maestro\b',
        r'\bBBC Verify\b',
        r'\bBBC InDepth\b',
        r'\bBBC Shop\b',
        r'\bBritBox\b',
        r'\bBBC\b',
        r'\bAudio\b',
        r'\bVideo\b',
        r'\bLive\b',
        r'\bWeather & Science\b',
        r'\bClimate Solutions\b',
        r'\bGreen Living\b',
        r'\bPodcast Categories\b',
        r'\bRadio\b',
        r'\bEntertainment News\b',
        r'\bArt & Design\b',
        r'\bFilm & TV\b',
        r'\bBooks\b',
        r'\bStyle\b',
        r'\bMusic\b',
        r'\bArts in Motion\b',
        r'\bDestinations\b',
        r'\bWorld’s Table\b',
        r'\bCulture & Experiences\b',
        r'\bAdventures\b',
        r'\bThe SpeciaList\b',
        r'\bNatural Wonders\b',
        r'\bExecutive Lounge\b',
        r'\bFuture of Business\b',
        r'\bAI v the Mind\b',
        r'\bScience & Health\b',
        r'\bSustainable Business\b',
        r'\bIn Pictures\b',
        r'\bSport\b',
        r'\bBusiness\b',
        r'\bInnovation\b',
        r'\bCulture\b',
        r'\bArts\b',
        r'\bTravel\b',
        r'\bEarth\b',
        r'\bNews\b',
        r'\bHome\b',
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)

    # Remove repeated lines (simple deduplication)
    lines = text.split('. ')
    seen = set()
    deduped_lines = []
    for line in lines:
        l = line.strip()
        if l and l.lower() not in seen:
            deduped_lines.append(l)
            seen.add(l.lower())
    text = '. '.join(deduped_lines)

    return text


def safe_format_url(u: str) -> str:
    """Normalize and sanitize a URL string for stable use.

    Steps:
    - ensure scheme
    - lowercase scheme/host
    - remove default ports
    - strip fragment
    - remove common tracking query params
    - strip trailing slash (except root)
    """
    if not u or not isinstance(u, str):
        return u
    try:
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
        p = urlparse(u)
        scheme = p.scheme.lower() if p.scheme else 'http'
        netloc = p.netloc.lower()
        # remove default ports
        if netloc.endswith(':80') and scheme == 'http':
            netloc = netloc[:-3]
        if netloc.endswith(':443') and scheme == 'https':
            netloc = netloc[:-4]
        # strip fragment
        fragment = ''
        # remove tracking params
        pairs = parse_qsl(p.query, keep_blank_values=True)
        TRACKING_PREFIXES = ('utm_', 'fbclid', 'gclid', 'mc_cid', 'mc_eid')
        filtered = [(k, v) for (k, v) in pairs if not any(k.startswith(pref) for pref in TRACKING_PREFIXES)]
        query = urlencode(sorted(filtered))
        path = p.path or ''
        if path.endswith('/') and path != '/':
            path = path.rstrip('/')
        norm = urlunparse((scheme, netloc, path, '', query, fragment))
        return norm
    except Exception:
        return u.strip()


def html_to_markdown(html: str) -> str:
    """Convert HTML to markdown when possible, fallback to cleaned text.

    Tries `markdownify` or `html2text` if installed; otherwise extracts text via BeautifulSoup.
    """
    if not html:
        return ''
    # First, try to extract the main article HTML using readability-lxml
    try:
        from readability import Document
        try:
            doc = Document(html)
            summary_html = doc.summary()
            # If summary is empty, fall back to the original html
            if summary_html and len(summary_html.strip()) > 20:
                html = summary_html
        except Exception:
            # readability failed for this document; continue with original html
            pass
    except Exception:
        # readability not installed; continue
        pass

    # Prefer markdownify
    try:
        from markdownify import markdownify as mdify
        # use ATX-style headings and preserve lists
        return mdify(html, heading_style='ATX', bullets='-')
    except Exception:
        pass
    # Fallback to html2text
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        return h.handle(html)
    except Exception:
        pass
    # Last resort: extract visible text
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        return soup.get_text(separator='\n', strip=True)
    except Exception:
        # Strip tags naively
        return re.sub(r'<[^>]+>', '', html)


def upload_text_to_gcs(bucket_name: str, destination: str, content: str, content_type: str = 'text/plain') -> str:
    """Upload text to GCS and return the gs:// path. Raises informative error if GCS client unavailable."""
    try:
        from google.cloud import storage as gcs_lib
    except Exception as e:
        raise RuntimeError('google.cloud.storage not installed: ' + str(e))
    client = gcs_lib.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_string(content, content_type=content_type)
    logger.info(f"Uploaded to GCS: gs://{bucket_name}/{destination}")
    return f'gs://{bucket_name}/{destination}'


def save_text_to_gcs(content: str, filename_prefix: str = 'scraped', bucket: Optional[str] = None) -> Optional[str]:
    """Convenience: compute filename and upload content to GCS if configured.

    Returns gs:// path or None if upload not performed.
    """
    bucket_name = bucket or GCS_BUCKET
    if not bucket_name:
        logger.debug('GCS bucket not configured; skipping upload')
        return None
    # create a filename with timestamp and short hash
    try:
        import hashlib
        ts = datetime.now().strftime('%Y%m%dT%H%M%SZ')
        h = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
        dest = f"{filename_prefix}/{h}_{ts}.md"
        return upload_text_to_gcs(bucket_name, dest, content, content_type='text/markdown')
    except Exception as e:
        logger.warning(f"Failed to upload to GCS: {e}")
        return None


def save_report_to_text(report_content: str, filename: str) -> str:
    """Saves the report content to Firestore only."""
    try:
        from google.cloud import firestore
        db = firestore.Client()
        # Add a timestamp to filename if not already present (YYYYMMDD_HHMMSS)
        import re
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not re.search(r"_\d{8}_\d{6}(?:\.|$)", filename):
            # insert before extension if present
            if '.' in filename:
                parts = filename.rsplit('.', 1)
                filename_with_ts = f"{parts[0]}_{ts}.{parts[1]}"
            else:
                filename_with_ts = f"{filename}_{ts}"
        else:
            filename_with_ts = filename

        # Use filename_with_ts as document id for uniqueness
        doc_ref = db.collection("report_files").document(filename_with_ts)
        doc_ref.set({
            "filename": filename_with_ts,
            "content": report_content,
            "saved_at": datetime.now().isoformat()
        })
        logger.info(f"Report saved to Firestore: {filename_with_ts}")
        return filename_with_ts  # Return filename for reference
    except Exception as e:
        logger.warning(f"Could not save report to Firestore: {e}")
        return ""


def format_research_report(report_content: str) -> str:
    """
    Format a research report with proper markdown structure and readability improvements.
    
    Args:
        report_content (str): The raw report content to format
        
    Returns:
        str: Formatted report with improved structure and readability
    """
    if not report_content or not report_content.strip():
        return report_content

    # Split into lines for processing
    lines = report_content.split('\n')
    formatted_lines = []
    
    last_was_empty = False
    for i, line in enumerate(lines):
        line = line.strip()
        # Skip multiple consecutive empty lines
        if not line:
            if not last_was_empty:
                formatted_lines.append('')
                last_was_empty = True
            continue
        last_was_empty = False
        # Format main headings (# Title)
        if line.startswith('# '):
            if formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            formatted_lines.append('')
            continue
        # Format sub-headings (## Title)
        if line.startswith('## '):
            if formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            formatted_lines.append('')
            continue
        # Format sub-sub-headings (### Title)
        if line.startswith('### '):
            if formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            formatted_lines.append('')
            continue
        # Format bullet points
        if line.startswith('- ') or line.startswith('* '):
            if formatted_lines and not formatted_lines[-1].startswith(('- ', '* ')) and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            continue
        # Format numbered lists
        if re.match(r'^\d+\. ', line):
            if formatted_lines and not re.match(r'^\d+\. ', formatted_lines[-1] or '') and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            continue
        # Regular paragraph text
        formatted_lines.append(line)
    
    # Join lines back together
    formatted_content = '\n'.join(formatted_lines)
    
    # Clean up excessive whitespace (more than 2 consecutive empty lines)
    formatted_content = re.sub(r'\n{4,}', '\n\n\n', formatted_content)
    
    # Ensure report starts and ends cleanly
    formatted_content = formatted_content.strip()
    # Prepend a generated TOC if headings exist
    try:
        toc = generate_toc(formatted_content, max_level=3)
        if toc:
            formatted_content = toc + '\n\n' + formatted_content
    except Exception:
        pass

    return formatted_content


def _slugify_heading(text: str) -> str:
    """Create a GitHub-style anchor slug from a heading text."""
    s = text.strip().lower()
    # remove code ticks
    s = re.sub(r'`+', '', s)
    # remove non-alphanumeric characters except spaces and hyphens
    s = re.sub(r"[^a-z0-9\s-]", '', s)
    s = re.sub(r'\s+', '-', s)
    s = re.sub(r'-{2,}', '-', s)
    s = s.strip('-')
    return s


def generate_toc(markdown: str, max_level: int = 3) -> str:
    """Generate a markdown table-of-contents from headings in the given markdown.

    - Scans for ATX-style headings (#, ##, ###)
    - Includes headings up to `max_level`
    - Produces nested bullet list with links to anchors
    """
    if not markdown:
        return ''
    toc_lines = []
    for line in markdown.splitlines():
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if not m:
            continue
        level = len(m.group(1))
        if level > max_level:
            continue
        title = m.group(2).strip()
        if not title:
            continue
        anchor = _slugify_heading(title)
        indent = '  ' * (level - 1)
        toc_lines.append(f"{indent}- [{title}](#{anchor})")
    if not toc_lines:
        return ''
    toc = ['## Table of contents', ''] + toc_lines + ['']
    return '\n'.join(toc)


def enhance_report_readability(report_content: str) -> str:
    """
    Enhance report readability with additional formatting improvements.
    
    Args:
        report_content (str): The report content to enhance
        
    Returns:
        str: Enhanced report with improved readability
    """
    if not report_content or not report_content.strip():
        return report_content
    
    content = report_content
    
    # Add proper spacing around citations
    content = re.sub(r'(\[\d+\])(?=[A-Za-z])', r'\1 ', content)  # Space after citation if followed by letter
    content = re.sub(r'([A-Za-z])(\[\d+\])', r'\1 \2', content)  # Space before citation if preceded by letter
    
    # Improve sentence structure
    content = re.sub(r'\.([A-Z])', r'. \1', content)  # Ensure space after periods
    content = re.sub(r',([A-Za-z])', r', \1', content)  # Ensure space after commas
    content = re.sub(r';([A-Za-z])', r'; \1', content)  # Ensure space after semicolons
    content = re.sub(r':([A-Za-z])', r': \1', content)  # Ensure space after colons
    
    # Fix multiple spaces
    content = re.sub(r' {2,}', ' ', content)
    
    # Ensure proper paragraph breaks
    content = re.sub(r'\.([A-Z][a-z])', r'.\n\n\1', content)  # Add paragraph breaks after sentences that end paragraphs
    
    # Format the content with the main formatting function
    content = format_research_report(content)
    
    return content


logger.info("utils.py loaded with utility functions.")
