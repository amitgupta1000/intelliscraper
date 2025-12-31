# utils.py
import re
import json
from .logging_setup import logger
import asyncio
import time # Added time for cache timestamp

# Define SearchResult and PDF classes here for simplicity or ensure they are imported
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import REPORT_FILENAME_TEXT

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
    # Remove extra whitespaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove HTML tags (a more general approach)
    text = re.sub(r'<[^>]+>', '', text)

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


# Helper function to rank URLs
from rank_bm25 import BM25Okapi # Assuming rank_bm25 is installed

def rank_urls(query: str, urls: List[str], relevant_contexts: Dict[str, Dict[str, str]]) -> List[str]:
    """Ranks URLs based on their relevance to the query using BM25."""
    if not urls or not relevant_contexts or not query:
        return urls # Return original order if ranking not possible

    # Extract content from the new dictionary structure
    corpus = []
    for url in urls:
        context_data = relevant_contexts.get(url, {})
        if isinstance(context_data, dict):
            content = context_data.get('content', '')
        else:
            # Handle backward compatibility with old string format
            content = context_data if isinstance(context_data, str) else ''
        corpus.append(content)
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    if not any(tokenized_corpus): # Check if corpus is empty after tokenization
        return urls # Return original order

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    if not tokenized_query:
         return urls # Return original order

    scores = bm25.get_scores(tokenized_query)
    # Pair scores with URLs and sort in descending order of scores
    scored_urls = sorted(zip(scores, urls), reverse=True)
    ranked_urls = [url for score, url in scored_urls]

    return ranked_urls

# Function to save report to text

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

    # Enhance the conclusion section before further formatting
    report_content = enhance_conclusion_section(report_content)

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
    
    return formatted_content


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


# Replace all '## Conclusion' with a bold section header for conclusion
import re

def enhance_conclusion_section(text: str) -> str:
    # Replace markdown '## Conclusion' with bolded 'Conclusion' section
    return re.sub(r'## Conclusion\s*', '\n**Conclusion**\n', text)

# Example usage in formatting pipeline:
# formatted_text = enhance_conclusion_section(generated_text)


logger.info("utils.py loaded with utility functions.")
