# ENTERPRISE RISK NEWS
# uses shared utilities for common functionality; this includes the debug mode

import datetime as dt
import random
import time
import re
import csv
import requests
from pathlib import Path
from newspaper import Article, Config
from googlenewsdecoder import new_decoderv1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser
import sys

# GLOBAL CONSTANTS
RISK_ID_COL = "ENTERPRISE_RISK_ID" # makes sure it matches the CSV column
SEARCH_DAYS = 7  # look back this many days for news articles; edit to change

# decoding logic (retained from original script but made a fx)
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

# IMPORTANT!! Import shared utilities from utils.py
from utils import (
    ScraperSession, setup_nltk, load_existing_links, setup_output_dir,
    save_results, print_debug_info, DEBUG_MODE,
    MAX_ARTICLES_PER_TERM, MAX_SEARCH_TERMS, load_source_lists, 
    calculate_quality_score
)

# This is the main fx that orchestrates the entire process.
def main():
    # config
    RISK_TYPE = "enterprise"
    ENCODED_CSV = "EnterpriseRisksListEncoded.csv"
    OUTPUT_CSV = "enterprise_risks_online_sentiment.csv"
    
    # process time start for reference
    print("*" * 50)
    start_time = dt.datetime.now()
    print_debug_info("EnterpriseRiskNews", RISK_TYPE, start_time)
    
    # setup NLTK and session etc.
    setup_nltk()
    session = ScraperSession()
    analyzer = SentimentIntensityAnalyzer()
    
    # load data
    output_path = setup_output_dir(OUTPUT_CSV)
    existing_links = load_existing_links(output_path)
    search_terms_df = load_search_terms(ENCODED_CSV, RISK_ID_COL)
    
    # only limit search terms in debug mode
    if DEBUG_MODE and MAX_SEARCH_TERMS:
        search_terms_df = search_terms_df.head(MAX_SEARCH_TERMS)
        print(f"DEBUG: Limited to first {MAX_SEARCH_TERMS} search terms")
    
    # load whitelist, paywalled, and credibility sources
    whitelist, paywalled, credibility_map = load_source_lists()
    
    # process articles
    articles_df = process_enterprise_articles(search_terms_df, session, existing_links, analyzer, whitelist, paywalled, credibility_map)
    
    # save results
    if not articles_df.empty:
        record_count = save_results(articles_df, output_path, RISK_TYPE)
        print(f"Completed: {record_count} total records")
    else:
        print("WARNING!!! No articles processed!")
    
    # end time for reference
    print(f"Completed at: {dt.datetime.now()}")
    print("*" * 50)

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
        
        # filter out rows with invalid search terms
        valid_df = df.dropna(subset=['SEARCH_TERMS'])
        if valid_df.empty:
            print("ERROR!!! No valid search terms after decoding!")
            sys.exit(1)
        return valid_df
    except FileNotFoundError:
        print(f"ERROR!!! data/{encoded_csv_path} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading data/{encoded_csv_path}: {e}")
        sys.exit(1)

def process_enterprise_articles(search_terms_df, session, existing_links, analyzer, whitelist, paywalled, credibility_map):
    # this is the MAIN processing loop for enterprise articles
    print(f"Processing {len(search_terms_df)} search terms...")
    
    all_articles = []
    
    # setup newspaper config
    config = Config()
    user_agent = random.choice(session.user_agents)
    config.browser_user_agent = user_agent
    config.enable_image_fetching = False  # faster without images!
    config.request_timeout = 10 if DEBUG_MODE else 20
    
    # set dates for search (using SEARCH_DAYS global constant)
    now = dt.date.today()
    yesterday = now - dt.timedelta(days=SEARCH_DAYS)
    
    # process each search term
    for idx, row in search_terms_df.iterrows():
        # quick exit for debug mode!
        if DEBUG_MODE and len(all_articles) >= 5:
            print("DEBUG: Early exit after 5 articles")
            break
            
        search_term = row['SEARCH_TERMS']  # use DECODED term
        risk_id = row[RISK_ID_COL]
        
        if pd.isna(search_term):
            print(f"  -> Skipping invalid search term for risk ID {risk_id}")
            continue
            
        print(f"Processing search term {idx + 1}/{len(search_terms_df)} (ID: {risk_id}) - '{search_term[:50]}...'")
        
        # Get Google News articles
        articles = get_google_news_articles(search_term, session, existing_links, MAX_ARTICLES_PER_TERM, now, yesterday, whitelist, paywalled, credibility_map)
        
        if not articles:
            print(f"  -> No new articles found for this term")
            continue
        
        # IMPORTANT FOR OPTIMIZATION: process articles in parallel
        processed_articles = process_articles_batch(articles, config, analyzer, search_term, whitelist, risk_id, existing_links)
        
        all_articles.extend(processed_articles)
        print(f"  -> Processed {len(processed_articles)} articles")
        
        # rate limiting every 5 terms to ease load on Google
        if idx % 5 == 0 and idx > 0:
            print("  -> rate limiting pause...")
            time.sleep(random.uniform(2, 5))
    
    # Create final dataframe
    if all_articles:
        df = pd.DataFrame(all_articles)
        print(f"Total articles collected: {len(df)}")
        return df
    else:
        print("No articles to process")
        return pd.DataFrame()

def get_google_news_articles(search_term, session, existing_links, max_articles, now, yesterday, whitelist, paywalled, credibility_map):
    # from original logic, fetch articles from Google News RSS
    articles = []
    article_count = 0
    
    # iterate over first 3 pages (10 results per page)
    for page in range(3):
        start = page * 10
        try:
            time.sleep(0.5)  # rate limit - this avoids 429 errors encountered previously
            url_start = 'https://news.google.com/rss/search?q='
            url_end = f'%20when%3A{SEARCH_DAYS}d'  # use SEARCH_DAYS global constant
            req = session.session.get(f"{url_start}{search_term}{url_end}&start={start}", headers=session.get_random_headers())
            req.raise_for_status()
            
            # Parse RSS feed
            soup = BeautifulSoup(req.content, 'xml')
            items = soup.find_all('item')
            
            print(f"    - Page {page+1}: found {len(items)} potential articles")
            
            for item_idx, item in enumerate(items):
                # Decode the Google News encoded URL - FIXED VERSION
                try:
                    encoded_url = item.link.text.strip()
                    decoded_result = new_decoderv1(encoded_url)
                    
                    # FIXED: Handle both string and dict responses from the decoder
                    if isinstance(decoded_result, dict):
                        if decoded_result.get('status') and 'decoded_url' in decoded_result:
                            decoded_url = decoded_result['decoded_url']
                        else:
                            if DEBUG_MODE:
                                print(f"    - Skipping: bad dict format: {decoded_result}")
                            continue
                    elif isinstance(decoded_result, str):
                        decoded_url = decoded_result
                    else:
                        if DEBUG_MODE:
                            print(f"    - Skipping: unexpected decode type: {type(decoded_result)}")
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
                
                # Extract domain from URL for filtering
                parsed_url = urlparse(decoded_url)
                domain_name = parsed_url.netloc.lower().replace('www.', '')
                
                if not any(domain_name.endswith(ext) for ext in ('.com', '.edu', '.org', '.net')):
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url[:50]}... (Invalid domain extension: {domain_name})")
                    continue
                
                # removed whitelist check to include all articles from first 3 pages
                # DOMAIN-BASED WHITELIST CHECK
                # source_is_whitelisted = False
                # if not whitelist:
                #     source_is_whitelisted = True
                # else:
                #     # Check if the actual domain matches any whitelist entry
                #     for white_source in whitelist:
                #         white_lower = white_source.lower().strip()
                #         if white_lower == domain_name or white_lower in domain_name:
                #             source_is_whitelisted = True
                #             break
                # if not source_is_whitelisted:
                #     if DEBUG_MODE:
                #         print(f"Skipping '{title_text[:50]}...' from {source_text} (domain: {domain_name} not in whitelist)")
                #     continue
                
                if "/en/" in decoded_url:
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url[:50]}... (Translated article)")
                    continue
                
                # removed existing_links check to handle in process_articles_batch
                # if decoded_url.lower().strip() in existing_links:
                #     if DEBUG_MODE:
                #         print(f"Skipping {decoded_url[:50]}... (Already exists)")
                #     continue
                
                try:
                    published_date = parser.parse(item.pubDate.text).date()
                except (ValueError, TypeError):
                    published_date = None
                    if DEBUG_MODE:
                        print(f"WARNING! Date Error: {item.pubDate.text}")
                
                # fix regex pattern for Python 3.12+
                regex_pattern = re.compile(r'(https?):((|(\\\\))+[\w\d:#@%;$()~_?\+-=\\\.&]*)')
                domain_search = regex_pattern.search(str(item.source))
                
                # add google index for article position (page-based + item position)
                google_index = page * 10 + item_idx + 1
                
                # check if domain is paywalled
                is_paywalled = domain_name in paywalled
                
                # set credibility type (default to Relevant Article)
                credibility_type = credibility_map.get(domain_name, 'Relevant Article')
                
                articles.append({
                    'url': decoded_url,
                    'title': title_text,
                    'html': None,  # will fetch during processing
                    'google_index': google_index,
                    'paywalled': is_paywalled,
                    'credibility_type': credibility_type
                })
                article_count += 1
                print(f"    - Added article: '{title_text[:50]}...' from {source_text} (domain: {domain_name}, index: {google_index}, paywalled: {is_paywalled}, credibility: {credibility_type})")
                
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
    seen_urls = set()  # DEDUP LAYER - track urls for this search term
    seen_titles = set()  # DEDUP LAYER - track titles for this search term
    
    def process_single_article(article_data):
        # handle single article processing
        try:
            url = article_data['url']
            title = article_data['title']
            google_index = article_data.get('google_index', 0)  # get index from article to see the sort order
            is_paywalled = article_data.get('paywalled', False)
            credibility_type = article_data.get('credibility_type', 'Relevant Article')
            
            # deduplicate by url and title for this search term
            url_key = url.lower().strip()
            title_key = title.lower().strip()[:100]  # limit title length for comparison
            if url_key in seen_urls or title_key in seen_titles:
                if DEBUG_MODE:
                    print(f"  - Skipping duplicate: '{title[:50]}...' ({url[:50]}...)")
                return None
            seen_urls.add(url_key)
            seen_titles.add(title_key)
            
            # removed existing_links check to handle in save_results - DEDUP LOGIC HANDLED IN utils.py
            # if url.lower().strip() in existing_links:
            #     return None
            
            # PRE-FILTER: Skip known problematic URL patterns from manual review
            problematic_patterns = [
                '/video/', '/videos/', '/watch/',
                'wsj.com/subscriptions', 'bloomberg.com/newsletters',
                'reuters.com/video', 'reuters.com/graphics'
            ]
            
            if any(pattern in url.lower() for pattern in problematic_patterns):
                if DEBUG_MODE:
                    print(f"  - Skipping problematic URL: {title[:50]}... ({url[:50]}...)")
                return None
            
            # download and parse article
            article = Article(url, config=config)
            article.download()
            
            # Check if download succeeded - FIXED: Use try/except instead of download_exception
            if not article.html or article.html.strip() == '':
                if DEBUG_MODE:
                    print(f"  - Download failed for '{title[:50]}...' (empty HTML)")
                return None
                
            article.parse()
            
            # extract content
            summary = article.summary if article.summary else article.text[:500]
            
            # skip empty content
            if not summary or len(summary.strip()) < 50:
                if DEBUG_MODE:
                    print(f"  - Empty content for '{title[:50]}...'")
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
                    'GOOGLE_INDEX': google_index,  # google news position for this article
                    'TITLE': title,
                    'LINK': url,
                    'PUBLISHED_DATE': article.publish_date or dt.datetime.now(),
                    'SUMMARY': summary[:1000],  # truncate for CSV size
                    'SENTIMENT_COMPOUND': sentiment['compound'],
                    'SOURCE_URL': url,
                    'PAYWALLED': is_paywalled,
                    'CREDIBILITY_TYPE': credibility_type,
                    'QUALITY_SCORE': quality_scores['total_score'],
                    # add individual score components
                    **{f'SCORE_{k.upper()}': v for k, v in quality_scores.items() if k != 'total_score'}
                }
            else:
                if DEBUG_MODE:
                    print(f"  - skipped low quality article '{title[:50]}...': score {quality_scores['total_score']}")
                return None
                
        except Exception as e:
            if DEBUG_MODE:
                print(f"  - error processing article '{title[:50] if 'title' in locals() else 'Unknown'}...': {e}")
            return None
    
    # process with threading (limit to 3 concurrent)
    if articles:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = executor.map(process_single_article, articles)
            processed = [r for r in results if r is not None]
    
    return processed

if __name__ == '__main__':
    main()