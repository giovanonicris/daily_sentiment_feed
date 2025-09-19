# 7/25/25 - CG adds Debug mode for easier debugging and testing
# 9/1/25 - CG optimizes GitHub Actions to do the ff: parallel processing, reduct CSV size, limit rates
# 9/9/25 - CG removes file splitting, optimizes for power bi, reduces csv size, keeps full summaries
# 9/11/25 - CG keeps source_url, populates with domain, limits to 3 google news pages
# 9/11/25 - CG adds quality scoring logic, removes relative file paths
# 9/11/25 - CG fixes syntax error in calculate_quality_score

import requests
import random
import re
import time
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser
from newspaper import Article, Config
import datetime as dt
import nltk
from googlenewsdecoder import new_decoderv1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import chardet
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import cProfile
import csv
import sys

# IMPORTANT!!!
# DEBUG MODE SETTINGS - CHANGE THIS TO False WHEN RUNNING IN PROD
DEBUG_MODE = False
MAX_SEARCH_TERMS = 2 if DEBUG_MODE else None
MAX_ARTICLES_PER_TERM = 3 if DEBUG_MODE else 20
SKIP_ARTICLE_PROCESSING = True if DEBUG_MODE else False

# global config
RISK_ID_COL = "EMERGING_RISK_ID"

# original decoder function from your working script
def process_encoded_search_terms(term):
    """decode encoded search terms from the csv file"""
    try:
        encoded_number = int(term)
        byte_length = (encoded_number.bit_length() + 7) // 8
        byte_rep = encoded_number.to_bytes(byte_length, byteorder='little')
        decoded_text = byte_rep.decode('utf-8')
        return decoded_text
    except (ValueError, UnicodeDecodeError, OverflowError):
        return None

# DEBUG META INFO
print("*" * 50)
print(f"DEBUG_MODE: {DEBUG_MODE}")
if DEBUG_MODE:
    print(f"   - Limited to {MAX_SEARCH_TERMS} search terms")
    print(f"   - Max {MAX_ARTICLES_PER_TERM} articles per term")
    print(f"   - Skip article processing: {SKIP_ARTICLE_PROCESSING}")
print(f"Script started at: {dt.datetime.now()}")
print(f"Working directory: {os.getcwd()}")
print(f"Script file location: {os.path.abspath(__file__)}")
print("*" * 50)

# set dates for today and yesterday
now = dt.date.today()
yesterday = now - dt.timedelta(days=1)

# check and download nltk resources
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        print(f"Downloading missing NLTK resource: {resource}")
        nltk.download(resource)

# create a list of random user agents
user_agent_list = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'
]

config = Config()
user_agent = random.choice(user_agent_list)
config.browser_user_agent = user_agent
config.enable_image_fetching = False  # disable image fetching for speed
config.request_timeout = 10 if DEBUG_MODE else 20
header = {'User-Agent': user_agent}

# set up requests session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# load existing dataset to avoid duplicate fetching
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
main_csv_path = os.path.join(output_dir, 'emerging_risks_online_sentiment.csv')
encoded_search_terms_csv = os.path.join(script_dir, 'EmergingRisksListEncoded.csv')

print("*" * 50)
print("EMERGING RISK NEWS PROCESSOR")
print(f"Script directory: {script_dir}")
print(f"Output directory: {output_dir}")
print(f"Main CSV path: {main_csv_path}")
print(f"Output directory exists: {os.path.exists(output_dir)}")
print("*" * 50)

# skip existing links check in debug mode for speed
if DEBUG_MODE:
    existing_links = set()
    print("DEBUG: Skipping existing links check for faster testing")
else:
    if os.path.exists(main_csv_path):
        existing_df = pd.read_csv(main_csv_path, usecols=lambda x: 'LINK' in x, encoding="utf-8")
        existing_links = set(existing_df["LINK"].str.lower().str.strip().tolist())
        print(f"Loaded {len(existing_links)} existing links from CSV")
    else:
        existing_links = set()
        print("No existing CSV found - starting fresh")

def load_search_terms(encoded_csv_path, risk_id_col):
    # Load and decode search terms from CSV - ORIGINAL LOGIC
    try:
        usecols = [risk_id_col, 'SEARCH_TERM_ID', 'ENCODED_TERMS']
        df = pd.read_csv(f'data/{encoded_csv_path}', encoding='utf-8', usecols=usecols)
        df[risk_id_col] = pd.to_numeric(df[risk_id_col], downcast='integer', errors='coerce')
        
        # ORIGINAL DECODING LOGIC
        df['SEARCH_TERMS'] = df['ENCODED_TERMS'].apply(process_encoded_search_terms)
        
        print(f"Loaded {len(df)} search terms from {encoded_csv_path}")
        valid_terms = df['SEARCH_TERMS'].dropna()
        print(f"Valid search terms ({len(valid_terms)}): {valid_terms.head().tolist()}")
        
        return df
    except FileNotFoundError:
        print(f"ERROR!!! data/{encoded_csv_path} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading data/{encoded_csv_path}: {e}")
        sys.exit(1)

def get_google_news_articles(search_term, session, existing_links, max_articles, now, yesterday):
    # original working RSS-based google news search
    articles = []
    article_count = 0
    
    # iterate over first 3 pages (10 results per page)
    for page in range(3):
        start = page * 10
        try:
            time.sleep(0.5)  # rate limit to avoid 429 errors
            url_start = 'https://news.google.com/rss/search?q={'
            url_end = '}%20when%3A1d'
            req = session.session.get(f"{url_start}{search_term}{url_end}&start={start}", headers=session.get_random_headers())
            req.raise_for_status()
            
            # Parse RSS feed
            soup = BeautifulSoup(req.content, 'xml')
            items = soup.find_all('item')
            
            print(f"    - Page {page+1}: found {len(items)} potential articles")
            
            for item in items:
                # Decode the Google News encoded URL
                try:
                    encoded_url = item.link.text.strip()
                    decoded_url = new_decoderv1(encoded_url)
                    
                    if not isinstance(decoded_url, str):
                        continue
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"    - URL decode error: {e}")
                    continue
                
                # Extract title and source
                title_elem = item.title.text.strip() if item.title else None
                source_elem = item.source.text.strip() if item.source else None
                
                if not title_elem or not source_elem:
                    continue
                
                title_text = title_elem
                source_text = source_elem
                
                # Basic filtering
                if len(title_text) < 10:
                    continue
                
                # Extract domain from source for filtering
                parsed_url = urlparse(decoded_url)
                domain_name = parsed_url.netloc.lower()
                
                if not any(domain_name.endswith(ext) for ext in ('.com', '.edu', '.org', '.net')):
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url} (Invalid domain extension)")
                    continue
                
                # Load source lists for filtering
                try:
                    whitelist_df = pd.read_csv('filter_in_sources.csv', encoding='utf-8')
                    blacklist_df = pd.read_csv('filter_out_sources.csv', encoding='utf-8')
                    whitelist = set(whitelist_df['SOURCE_NAME'].str.lower().str.strip().tolist())
                    blacklist = set(blacklist_df['SOURCE_NAME'].str.lower().str.strip().tolist())
                except FileNotFoundError:
                    print("WARNING: Could not load source filter files, skipping source filtering")
                    whitelist = set()
                    blacklist = set()
                
                # Check if source should be filtered
                source_domain = domain_name.replace('www.', '')
                if any(black_source in source_text.lower() or black_source in source_domain for black_source in blacklist):
                    if DEBUG_MODE:
                        print(f"Skipping article from {source_text} (Blacklisted source)")
                    continue
                
                if whitelist and not any(white_source in source_text.lower() or white_source in source_domain for white_source in whitelist):
                    if DEBUG_MODE:
                        print(f"Skipping article from {source_text} (Not in whitelist)")
                    continue
                
                if "/en/" in decoded_url:
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url} (Detected translated article)")
                    continue
                
                if decoded_url.lower().strip() in existing_links:
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url} (Already exists)")
                    continue
                
                try:
                    published_date = parser.parse(item.pubDate.text).date()
                except (ValueError, TypeError):
                    published_date = None
                    if DEBUG_MODE:
                        print(f"WARNING! Date Error: {item.pubDate.text}")
                
                # fix regex pattern for Python 3.12+
                regex_pattern = re.compile(r'(https?):((|(\\\\))+[\w\d:#@%;$()~_?\+-=\\\.&]*)')
                domain_search = regex_pattern.search(str(item.source))
                
                articles.append({
                    'url': decoded_url,
                    'title': title_text,
                    'html': None  # will fetch during processing
                })
                article_count += 1
                print(f"    - Added article: '{title_text[:50]}...' from {source_text}")
                
            if article_count >= max_articles:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Request error for term {search_term[:30]}... on page {page+1}: {e}")
            break
    
    print(f"  - found {len(articles)} new articles")
    return articles

def process_articles_batch(articles, config, analyzer, search_term, whitelist, risk_id, existing_links):
    # Process in parallel for optimization...
    processed = []
    
    def process_single_article(article_data):
        # handle single article processing
        try:
            url = article_data['url']
            title = article_data['title']
            
            # skip if already processed (double check)
            if url.lower().strip() in existing_links:
                return None
            
            # download and parse article
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            # extract content
            summary = article.summary if article.summary else article.text[:500]
            
            # skip empty content
            if not summary or len(summary.strip()) < 50:
                return None
            
            # sentiment analysis
            sentiment = analyzer.polarity_scores(title + " " + summary)
            
            # quality scoring
            quality_scores = calculate_quality_score(
                title, summary, url, [search_term], whitelist
            )
            
            # only keep articles with decent quality score
            if quality_scores['total_score'] >= 2:
                return {
                    'RISK_ID': risk_id,  # proper risk id mapping
                    'TITLE': title,
                    'LINK': url,
                    'PUBLISHED_DATE': article.publish_date or dt.datetime.now(),
                    'SUMMARY': summary[:1000],  # truncate for CSV size
                    'SENTIMENT_COMPOUND': sentiment['compound'],
                    'SOURCE_URL': url,
                    'QUALITY_SCORE': quality_scores['total_score'],
                    # add individual score components
                    **{f'SCORE_{k.upper()}': v for k, v in quality_scores.items() if k != 'total_score'}
                }
            else:
                print(f"  - skipped low quality article: score {quality_scores['total_score']}")
                return None
                
        except Exception as e:
            print(f"  - error processing article {url}: {e}")
            return None
    
    # process with threading (limit to 3 concurrent)
    if articles:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = executor.map(process_single_article, articles)
            processed = [r for r in results if r is not None]
    
    return processed

def calculate_quality_score(title, summary, url, search_terms, whitelist):
    """calculate quality score for articles based on multiple factors"""
    score = {
        'relevance': 0,
        'source_reputation': 0,
        'content_length': 0,
        'freshness': 0,
        'total_score': 0
    }
    
    # 1. Relevance scoring (search term matching)
    title_lower = title.lower()
    summary_lower = summary.lower()
    terms_lower = [term.lower() for term in search_terms]
    
    title_matches = sum(1 for term in terms_lower if term in title_lower)
    summary_matches = sum(1 for term in terms_lower if term in summary_lower)
    
    relevance_score = min(title_matches + summary_matches * 0.5, 2)
    score['relevance'] = relevance_score
    
    # 2. Source reputation scoring
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower().replace('www.', '')
        
        # load whitelist for bonus points
        if whitelist:
            if any(white_domain in domain for white_domain in whitelist):
                score['source_reputation'] = 2
            else:
                score['source_reputation'] = 1
        else:
            # basic domain reputation based on common patterns
            reputable_domains = ['nytimes', 'wsj', 'reuters', 'bloomberg', 'bbc', 'cnn', 'apnews', 'forbes']
            if any(rep_domain in domain for rep_domain in reputable_domains):
                score['source_reputation'] = 2
            else:
                score['source_reputation'] = 1
                
    except:
        score['source_reputation'] = 0
    
    # 3. Content length scoring
    content_length = len(summary)
    if content_length > 1000:
        score['content_length'] = 2
    elif content_length > 300:
        score['content_length'] = 1
    else:
        score['content_length'] = 0
    
    # 4. Freshness scoring (within last 24 hours = max score)
    try:
        # This would ideally use the article's publish date
        # For now, assume recent articles get full points
        score['freshness'] = 1
    except:
        score['freshness'] = 0
    
    # Calculate total score (weighted)
    score['total_score'] = (
        score['relevance'] * 1.5 +
        score['source_reputation'] * 1.2 +
        score['content_length'] * 0.8 +
        score['freshness'] * 0.5
    )
    
    return score

def process_emerging_articles(search_terms_df, session, existing_links, analyzer, whitelist):
    # this is the MAIN processing loop for emerging articles
    print(f"Processing {len(search_terms_df)} search terms...")
    
    all_articles = []
    
    # set dates for search (last 24 hours)
    # NOTE!! for backfilling, change to last 7 days
    now = dt.date.today()
    yesterday = now - dt.timedelta(days=1)
    
    # process each search term
    for idx, row in search_terms_df.iterrows():
        # quick exit for debug mode!
        if DEBUG_MODE and len(all_articles) >= 5:
            print("DEBUG: Early exit after 5 articles")
            break
            
        search_term = row['SEARCH_TERMS']  # use DECODED term
        risk_id = row[RISK_ID_COL]
        
        print(f"Processing search term {idx + 1}/{len(search_terms_df)} (ID: {risk_id})")
        
        # Get Google News articles
        articles = get_google_news_articles(search_term, session, existing_links, MAX_ARTICLES_PER_TERM, now, yesterday)
        
        if not articles:
            print(f"  - No new articles found for this term")
            continue
        
        # IMPORTANT FOR OPTIMIZATION: process articles in parallel
        processed_articles = process_articles_batch(articles, config, analyzer, search_term, whitelist, risk_id, existing_links)
        
        all_articles.extend(processed_articles)
        print(f"  - Processed {len(processed_articles)} articles")
        
        # rate limiting every 5 terms to ease load on Google
        if idx % 5 == 0 and idx > 0:
            print("  - rate limiting pause...")
            time.sleep(random.uniform(2, 5))
    
    return all_articles

def main():
    # config
    RISK_TYPE = "emerging"
    ENCODED_CSV = "EmergingRisksListEncoded.csv"
    OUTPUT_CSV = "emerging_risks_online_sentiment.csv"
    
    # process time start
    print("*" * 50)
    start_time = dt.datetime.now()
    print(f"EMERGING RISK NEWS - Started: {start_time}")
    print(f"Processing type: {RISK_TYPE}")
    print("*" * 50)
    
    # setup analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # load data
    search_terms_df = load_search_terms(ENCODED_CSV, RISK_ID_COL)
    
    # limit for debug mode
    if MAX_SEARCH_TERMS:
        search_terms_df = search_terms_df.head(MAX_SEARCH_TERMS)
        print(f"DEBUG: Limited to first {MAX_SEARCH_TERMS} search terms")
    
    # load whitelist sources
    try:
        whitelist_df = pd.read_csv('filter_in_sources.csv', encoding='utf-8')
        whitelist = set(whitelist_df['SOURCE_NAME'].str.lower().str.strip().tolist())
        print(f"Loaded {len(whitelist)} whitelist sources")
    except FileNotFoundError:
        print("WARNING: Could not load whitelist - using empty set")
        whitelist = set()
    
    # process articles
    print("Starting article processing...")
    summary = process_emerging_articles(search_terms_df, session, existing_links, analyzer, whitelist)
    
    # Create final dataframe
    if summary:
        final_df = pd.DataFrame(summary)
        
        # Apply quality scoring to existing data if needed
        if not final_df.empty and 'SEARCH_TERMS' not in final_df.columns:
            # For existing data, we might need to re-calculate scores
            print("Re-calculating quality scores for existing data...")
            
            # Load source lists
            try:
                blacklist_df = pd.read_csv('filter_out_sources.csv', encoding='utf-8')
                blacklist = set(blacklist_df['SOURCE_NAME'].str.lower().str.strip().tolist())
            except FileNotFoundError:
                blacklist = set()
            
            # Extract search terms from the dataframe (assuming they're stored somewhere)
            # For simplicity, we'll use a placeholder approach
            def get_search_terms_from_row(row):
                # This is a simplified approach - in production you'd want to map back to original terms
                return [row.get('TITLE', '').lower()]  # Use title as fallback
            
            # Apply quality scoring
            score_breakdown = final_df.apply(
                lambda row: calculate_quality_score(
                    row.get('TITLE', ''),
                    row.get('SUMMARY', ''),
                    row.get('SOURCE_URL', ''),
                    get_search_terms_from_row(row),
                    whitelist
                ),
                axis=1
            )
            
            # convert the series of dictionaries to separate columns
            score_df = pd.DataFrame(score_breakdown.tolist())
            
            # add all scoring columns to the final dataframe
            for col in score_df.columns:
                final_df[f'SCORE_{col.upper()}'] = score_df[col]
            
            # add total score column
            final_df['QUALITY_SCORE'] = final_df['SCORE_TOTAL_SCORE']
        
        # Filter by quality score
        if 'QUALITY_SCORE' in final_df.columns:
            high_quality_df = final_df[final_df['QUALITY_SCORE'] >= 2].copy()
            print(f"Filtered to {len(high_quality_df)} high-quality articles (score >= 2)")
            final_df = high_quality_df
        else:
            print("WARNING: No quality scores calculated - keeping all articles")
        
    else:
        final_df = pd.DataFrame()
        print("No articles processed")

print("*" * 50)
print(f"Processed {len(summary)} articles")
print(f"Final DataFrame shape: {final_df.shape}")
print(f"Final DataFrame columns: {final_df.columns.tolist()}")
if len(final_df) > 0:
    print("Sample of final data:")
    print(final_df.head(2))
else:
    print("WARNING!!! Final DataFrame is empty!")
print(f"\nQuality Score Statistics:")
if 'QUALITY_SCORE' in final_df.columns:
    print(f"Mean score: {final_df['QUALITY_SCORE'].mean():.2f}")
    print(f"Score distribution:")
    print(final_df['QUALITY_SCORE'].value_counts().sort_index())
    print(f"\nScoring Component Statistics:")
    scoring_cols = [col for col in final_df.columns if col.startswith('SCORE_') and col != 'SCORE_TOTAL_SCORE']
    for col in scoring_cols:
        print(f"{col}: Mean = {final_df[col].mean():.2f}, Non-zero = {(final_df[col] != 0).sum()}")
print("*" * 50)

# load existing data and combine
if os.path.exists(main_csv_path):
    existing_main_df = pd.read_csv(main_csv_path, parse_dates=['PUBLISHED_DATE'], encoding='utf-8')
    print(f"Loaded existing CSV with {len(existing_main_df)} records")
else:
    existing_main_df = pd.DataFrame()
    print("No existing CSV found - starting fresh")

# DEBUG BEFORE SAVING
print("*" * 50)
if not final_df.empty:
    print(f"Saving {len(final_df)} new records")
else:
    print("WARNING!!! No new records to save!")
print("*" * 50)

combined_df = pd.concat([existing_main_df, final_df], ignore_index=True).drop_duplicates(subset=['TITLE', 'LINK', 'PUBLISHED_DATE'])

# rolling 4-month window
cutoff_date = dt.datetime.now() - dt.timedelta(days=4 * 30)
combined_df['PUBLISHED_DATE'] = pd.to_datetime(combined_df['PUBLISHED_DATE'], errors='coerce')

if combined_df['PUBLISHED_DATE'].isna().any():
    print("Warning: Some rows have invalid PUBLISHED_DATE values.")

# separate current and old data
current_df = combined_df[combined_df['PUBLISHED_DATE'] >= cutoff_date].copy()
old_df = combined_df[combined_df['PUBLISHED_DATE'] < cutoff_date].copy()

# DEBUG AFTER COMBINING DATA
print("*" * 50)
print(f"Combined DataFrame shape: {combined_df.shape}")
print(f"Current DataFrame shape (after filtering): {current_df.shape}")
print("*" * 50)

# save current data
current_df.sort_values(by='PUBLISHED_DATE', ascending=False).to_csv(main_csv_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
file_size = os.path.getsize(main_csv_path)
print(f"Updated main CSV with {len(current_df)} records.")
print(f"File size: {file_size} bytes")

# DEBUG VERIFY FILE
print("*" * 50)
if os.path.exists(main_csv_path):
    file_size = os.path.getsize(main_csv_path)
    print(f"✓ Output file exists at: {main_csv_path}")
    print(f"✓ File size: {file_size} bytes")
    if file_size > 0:
        print("Preview file:")
        try:
            preview_df = pd.read_csv(main_csv_path).head(2)
            print(preview_df)
        except Exception as e:
            print(f"Could not preview file: {e}")
    else:
        print("File is empty!!!")
else:
    print("ERROR!!! Output file not created!")
print(f"Script completed at: {dt.datetime.now()}")
print("*" * 50)

# archive old data
if not DEBUG_MODE and not old_df.empty:
    old_df = old_df.sort_values(by='PUBLISHED_DATE')
    archive_path = os.path.join(output_dir, 'emerging_risks_sentiment_archive.csv')
    old_df.to_csv(archive_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
    print(f"Archived {len(old_df)} records to {archive_path}.")
elif DEBUG_MODE:
    print("DEBUGGING - skipping archival process")

if __name__ == '__main__':
    main()