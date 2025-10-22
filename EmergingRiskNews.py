# EMERGING RISK NEWS
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
from keybert import KeyBERT

# GLOBAL CONSTANTS
RISK_ID_COL = "EMERGING_RISK_ID" # makes sure it matches the CSV column
SEARCH_DAYS = 30  # look back this many days for news articles; edit to change

# decoding logic (retained from original script but made a fx)
def process_encoded_search_terms(term):
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
    calculate_quality_score, get_source_name
)

# This is the main fx that orchestrates the entire process.
def main():
    # config
    RISK_TYPE = "emerging"
    ENCODED_CSV = "EmergingRisksListEncoded.csv"
    OUTPUT_CSV = "emerging_risks_online_sentiment.csv"
    
    # process time start for reference
    print("*" * 50)
    start_time = dt.datetime.now()
    print_debug_info("EmergingRiskNews", RISK_TYPE, start_time)
    
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
    articles_df = process_emerging_articles(search_terms_df, session, existing_links, analyzer, whitelist, paywalled, credibility_map)
    
    # save results
    if not articles_df.empty:
        record_count = save_results(articles_df, output_path, RISK_TYPE)
        print(f"Completed: {record_count} total records")
    else:
        print("WARNING!!! No articles processed!!")
    
    # end time for reference
    print(f"Completed at: {dt.datetime.now()}")
    print("*" * 50)

def load_search_terms(encoded_csv_path, risk_id_col):
    # load and decode search terms from CSV - ORIGINAL LOGIC
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
            print("ERROR!!! No valid search terms after decoding!!")
            sys.exit(1)
        return valid_df
    except FileNotFoundError:
        print(f"ERROR!!! data/{encoded_csv_path} not found!!")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading data/{encoded_csv_path}: {e}")
        sys.exit(1)

def process_emerging_articles(search_terms_df, session, existing_links, analyzer, whitelist, paywalled, credibility_map):
    # this is the MAIN processing loop for emerging articles
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
        search_term_id = row['SEARCH_TERM_ID']  # capture search term id
        
        if pd.isna(search_term):
            print(f"  ---Skipping invalid search term for risk ID {risk_id}")
            continue
            
        print(f"Processing search term {idx + 1}/{len(search_terms_df)} (ID: {risk_id}, SEARCH_TERM_ID: {search_term_id}) - '{search_term[:50]}...'")
        
        # Get Google News articles
        articles = get_google_news_articles(search_term, session, existing_links, MAX_ARTICLES_PER_TERM, now, yesterday, whitelist, paywalled, credibility_map)
        
        if not articles:
            print(f"  - No new articles found for this term")
            continue

        # just checking...for debug, DELETE LATER!
        if articles and DEBUG_MODE:
            print("Sample article keys:", articles[0].keys() if isinstance(articles[0], dict) else "Not a dict")
            print("Sample source value:", articles[0].get('source', 'No source key') if articles else 'No articles')
        
        # IMPORTANT FOR OPTIMIZATION: process articles in parallel
        processed_articles = process_articles_batch(articles, config, analyzer, search_term, whitelist, risk_id, search_term_id, existing_links)
        
        all_articles.extend(processed_articles)
        print(f"  ---Processed {len(processed_articles)} articles")
        
        # rate limiting every 5 terms to ease load on Google
        if idx % 5 == 0 and idx > 0:
            print("  ---rate limiting pause...")
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
    
    # TECH_DEBT! Changed to 5 pages for the first full run
    # iterate over first 5 pages (10 results per page)
    for page in range(3):
        start = page * 10
        try:
            time.sleep(0.5)  # rate limit - this avoids 429 errors encountered previously
            url_start = 'https://news.google.com/rss/search?q='
            url_end = f'%20when%3A{SEARCH_DAYS}d'  # use SEARCH_DAYS global constant
            req = session.session.get(f"{url_start}{search_term}{url_end}&start={start}", headers=session.get_random_headers())
            req.raise_for_status()
            
            # parse RSS feed
            soup = BeautifulSoup(req.content, 'xml')
            items = soup.find_all('item')
            
            print(f"    ---Page {page+1}: found {len(items)} potential articles")
            
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
                            #if DEBUG_MODE:
                            print(f"    --Skipping: bad dict format: {decoded_result}") # this handles when status is False or missing
                            continue
                    elif isinstance(decoded_result, str):
                        decoded_url = decoded_result
                    else:
                        #if DEBUG_MODE:
                        print(f"    ---Skipping: unexpected decode type: {type(decoded_result)}")
                        continue
                        
                except Exception as e:
                    #if DEBUG_MODE:
                    print(f"    ---URL decode error: {e}") # if decode failed, then we skip
                    continue
                
                # extract title and source
                title_elem = item.title.text.strip() if item.title else None
                source_elem = item.source.text.strip() if item.source else None
                
                if not title_elem or not source_elem:
                    continue
                
                title_text = title_elem
                source_text = source_elem
                
                # basic filtering
                if len(title_text) < 10:
                    continue
                
                # extract domain from URL for filtering
                parsed_url = urlparse(decoded_url)
                full_domain = parsed_url.netloc.replace('www.', '')
                
                # FILTER SERIES for reliable TLDs (.com, .edu, .org, .net, .gov) and exclude international paths
                # FILTER #1 = Reliable TLDs only
                valid_tlds = ('.com', '.edu', '.org', '.net', '.gov', '.co', '.news', '.info', '.biz')
                if not any(full_domain.endswith(ext) for ext in valid_tlds):
                    if DEBUG_MODE:
                        print(f"    - Skipping: invalid domain extension: {full_domain}")
                    continue
                # FILTER #2 = No international paths/subdomains
                if re.search(r'\.[a-z]{2}$|\.[a-z]{2}\.[a-z]{2}$', full_domain.lower()):
                    # if DEBUG_MODE:
                    print(f"Skipping {decoded_url[:50]}... (International path or subdomain: {parsed_url.path or full_domain})")
                    continue
                # FILTER #3 = No translated to English articles
                if "/en/" in decoded_url.lower():
                    # if DEBUG_MODE:
                    print(f"Skipping {decoded_url[:50]}... (Translated article)")
                    continue
                
                try:
                    published_date = parser.parse(item.pubDate.text).date()
                except (ValueError, TypeError):
                    published_date = None
                    if DEBUG_MODE:
                        print(f"WARNING! Date Error: {item.pubDate.text}")
                
                # use regex to extract source domain
                regex_pattern = re.compile(r'(https?):((|(\\\\))+[\w\d:#@%;$()~_?\+-=\\\.&]*)')
                domain_search = regex_pattern.search(str(item.source))
                
                # add google index for article position (page-based + item position)
                google_index = page * 10 + item_idx + 1
                
                # check if domain is paywalled
                is_paywalled = full_domain.lower() in paywalled
                
                # set credibility type (default to Relevant Article)
                credibility_type = credibility_map.get(full_domain.lower(), 'Relevant Article')
                
                articles.append({
                    'url': decoded_url,
                    'title': title_text,
                    'html': None,  # will fetch during processing
                    'google_index': google_index,
                    'paywalled': is_paywalled,
                    'credibility_type': credibility_type
                })
                print(f"    - Added article: '{title_text[:50]}...' from {source_text} (domain: {get_source_name(decoded_url)}, full_domain: {full_domain}, index: {google_index}, paywalled: {is_paywalled}, credibility: {credibility_type})")

            article_count += 1    
            if article_count >= max_articles:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"SPOTTED REQUEST ERROR - term {search_term[:30]}... on page {page+1}: {e}")
            break
    
    print(f"  ---found {len(articles)} new articles")
    return articles

def process_articles_batch(articles, config, analyzer, search_term, whitelist, risk_id, search_term_id, existing_links): #STID to delete later!
    # process in parallel for optimization...
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
                    print(f"  ---Skipping duplicate: '{title[:50]}...' ({url[:50]}...)")
                return None
            seen_urls.add(url_key)
            seen_titles.add(title_key)
            
            # removed existing_links check to handle in save_results - DEDUP LOGIC HANDLED in utils.py
            # if url.lower().strip() in existing_links:
            #     return None
            
            # PRE-FILTER: Skip known problematic URL patterns from manual review
            # Add as needed based on result review
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
            
            # check if download succeeded - FIXED: Use try/except instead of download_exception
            if not article.html or article.html.strip() == '':
                if DEBUG_MODE:
                    print(f"  ---Download failed for '{title[:50]}...' (empty HTML)")
                return None

            #parse article, extract keywords    
            article.parse()
            keywords = article.keywords if article.keywords else []
            # KEYWORD EXTRACT FALLBACK - use KeyBERT if no keywords found using newspaper lib
            if not keywords and article.text:
                kw_model = KeyBERT()
                keywords = kw_model.extract_keywords(article.text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
                keywords = [kw[0] for kw in keywords]  # extract keyword strings
            if DEBUG_MODE:
                print(f"    - Extracted keywords for '{title[:50]}...': {keywords}")
                print(f"    - Article text length: {len(article.text) if article.text else 0} chars")
            
            # extract content
            summary = article.summary if article.summary else article.text[:500]
            
            # skip empty content
            if not summary or len(summary.strip()) < 50:
                if DEBUG_MODE:
                    print(f"  ---Empty content for '{title[:50]}...'")
                return None
            
            # sentiment analysis
            sentiment = analyzer.polarity_scores(title + " " + summary)
            sentiment_category = 'Negative' if sentiment['compound'] <= -0.05 else 'Positive' if sentiment['compound'] >= 0.05 else 'Neutral'
            
            # quality scoring
            quality_scores = calculate_quality_score(
                title, summary, url, [search_term], whitelist
            )
            
            # include all articles, keeping quality score for review
            print(f"DEBUG: Assigning SEARCH_TERM_ID={search_term_id} to article '{title[:50]}...' (RISK_ID={risk_id})") #STID to delete later!

            # final formatting before write
            source_name = get_source_name(url).capitalize()
            
            publish_date = article.publish_date or dt.datetime.now()
            formatted_publish_date = pd.to_datetime(publish_date).strftime('%Y-%m-%d %H:%M:%S')

            return {
                'RISK_ID': risk_id,  # proper risk id mapping
                'SEARCH_TERM_ID': search_term_id, #STID to delete later!
                'GOOGLE_INDEX': google_index,  # google news position for this article
                'TITLE': title,
                'LINK': url,
                'PUBLISHED_DATE': formatted_publish_date,
                'SUMMARY': summary[:500],  # truncate for CSV size
                'KEYWORDS': ', '.join(keywords) if keywords else '',
                'SENTIMENT_COMPOUND': sentiment['compound'],
                'SENTIMENT': sentiment_category,
                'SOURCE': source_name,
                'SOURCE_URL': url,
                'PAYWALLED': is_paywalled,
                'CREDIBILITY_TYPE': credibility_type,
                'QUALITY_SCORE': quality_scores['total_score'],
                # add individual score components
                **{f'SCORE_{k.upper()}': v for k, v in quality_scores.items() if k != 'total_score'},
            }
                
        except Exception as e:
            if DEBUG_MODE:
                print(f"  ---error processing article '{title[:50] if 'title' in locals() else 'Unknown'}...': {e}")
            return None
    
    # process with threading (limit to 3 concurrent to avoid overload)
    if articles:
        with ThreadPoolExecutor(max_workers=1) as executor:
            results = executor.map(process_single_article, articles)
            processed = [r for r in results if r is not None]
    
    return processed

if __name__ == '__main__':
    main()