# enterprise risk news
# uses shared utilities for common functionality

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

# global config
RISK_ID_COL = "ENTERPRISE_RISK_ID"

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

# import shared utilities
from utils import (
    ScraperSession, setup_nltk, load_existing_links, setup_output_dir,
    save_results, print_debug_info, DEBUG_MODE,
    MAX_ARTICLES_PER_TERM, MAX_SEARCH_TERMS, load_source_lists, 
    calculate_quality_score
)

def main():
    # config
    RISK_TYPE = "enterprise"
    ENCODED_CSV = "EnterpriseRisksListEncoded.csv"
    OUTPUT_CSV = "enterprise_risks_online_sentiment.csv"
    
    # process time start
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
    
    # limit for debug mode
    if MAX_SEARCH_TERMS:
        search_terms_df = search_terms_df.head(MAX_SEARCH_TERMS)
        print(f"DEBUG: Limited to first {MAX_SEARCH_TERMS} search terms")
    
    # load whitelist sources
    whitelist = load_source_lists()
    
    # process articles
    articles_df = process_enterprise_articles(search_terms_df, session, existing_links, analyzer, whitelist)
    
    # save results
    if not articles_df.empty:
        record_count = save_results(articles_df, output_path, RISK_TYPE)
        print(f"Completed: {record_count} total records")
    else:
        print("WARNING: No articles processed!")
    
    # end time
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
        
        return df
    except FileNotFoundError:
        print(f"ERROR!!! data/{encoded_csv_path} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading data/{encoded_csv_path}: {e}")
        sys.exit(1)

def process_enterprise_articles(search_terms_df, session, existing_links, analyzer, whitelist):
    # this is the MAIN processing loop for enterprise articles
    print(f"Processing {len(search_terms_df)} search terms...")
    
    all_articles = []
    
    # setup newspaper config
    config = Config()
    user_agent = random.choice(session.user_agents)
    config.browser_user_agent = user_agent
    config.enable_image_fetching = False  # faster without images!
    config.request_timeout = 10 if DEBUG_MODE else 20
    
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
        
        print(f"Processing search term {idx + 1}/{len(search_terms_df)} (ID: {risk_id}) - '{search_term[:30]}...'")
        
        # Get Google News articles
        articles = get_google_news_articles(search_term, session, existing_links, MAX_ARTICLES_PER_TERM, now, yesterday, whitelist)
        
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
    
    # Create final dataframe
    if all_articles:
        df = pd.DataFrame(all_articles)
        print(f"Total articles collected: {len(df)}")
        return df
    else:
        print("No articles to process")
        return pd.DataFrame()

def get_google_news_articles(search_term, session, existing_links, max_articles, now, yesterday, whitelist):
    # original working RSS-based google news search
    articles = []
    article_count = 0
    
    # iterate over first 3 pages (should be 10 results per page)
    for page in range(3):
        start = page * 10
        try:
            time.sleep(0.5)  # rate limit to avoid 429 errors
            url_start = 'https://news.google.com/rss/search?q='
            url_end = '%20when%3A1d'
            req = session.session.get(f"{url_start}{search_term}{url_end}&start={start}", headers=session.get_random_headers())
            soup = BeautifulSoup(req.text, 'xml')
            
            items = soup.find_all("item")
            print(f"    - Page {page+1}: found {len(items)} RSS items")
            
            for item in items:
                if article_count >= max_articles:
                    print(f"DEBUGGING - stopping at {max_articles} articles for term '{search_term[:30]}...'")
                    break
                
                title_text = item.title.text.strip() if item.title else ''
                encoded_url = item.link.text.strip() if item.link else ''
                source_text = item.source.text.strip().lower() if item.source else ''
                
                if not title_text or not encoded_url:
                    continue
                
                source_url = urlparse(encoded_url).netloc  # extract domain for SOURCE_URL
                
                interval_time = 5
                decoded_url = new_decoderv1(encoded_url, interval=interval_time)
                
                if decoded_url.get("status"):
                    decoded_url = decoded_url['decoded_url'].strip().lower()
                    parsed_url = urlparse(decoded_url)
                    domain_name = parsed_url.netloc.lower()
                    
                    if not any(domain_name.endswith(ext) for ext in ('.com', '.edu', '.org', '.net', '.gov', '.co')):
                        if DEBUG_MODE:
                            print(f"Skipping {decoded_url} (Invalid domain extension)")
                        continue
                    
                    # # whitelist check (instead of filtered_sources)
                    # if source_text not in whitelist:
                    #     if DEBUG_MODE:
                    #         print(f"Skipping article from {source_text} (Not in whitelist)")
                    #     continue
                    
                    # if "/en/" in decoded_url:
                    #     if DEBUG_MODE:
                    #         print(f"Skipping {decoded_url} (Detected translated article)")
                    #     continue
                    
                    if decoded_url in existing_links:
                        if DEBUG_MODE:
                            print(f"Skipping {decoded_url} (Already exists)")
                        continue
                    
                    try:
                        published_date = parser.parse(item.pubDate.text).date()
                    except (ValueError, TypeError):
                        published_date = None
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
                else:
                    print(f"    - Decode error: {decoded_url.get('message', 'Unknown error')}")
                
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

if __name__ == '__main__':
    main()