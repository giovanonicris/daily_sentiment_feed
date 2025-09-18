# enterprise risk news
# uses shared utilities for common functionality

import datetime as dt
import random
import time
import re
import csv
from pathlib import Path
from newspaper4k import Article, Config
from googlenewsdecoder import new_decoderv1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import pandas as pd

# import shared utilities
from utils import (
    ScraperSession, setup_nltk, load_existing_links, setup_output_dir,
    load_search_terms, save_results, print_debug_info, DEBUG_MODE,
    MAX_ARTICLES_PER_TERM, MAX_SEARCH_TERMS, load_source_lists, 
    calculate_quality_score
)

def main():
    # config
    RISK_TYPE = "enterprise"
    ENCODED_CSV = "EnterpriseRisksListEncoded.csv"
    OUTPUT_CSV = "enterprise_risks_online_sentiment.csv"
    RISK_ID_COL = "ENTERPRISE_RISK_ID"
    
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
            
        search_term = row['ENCODED_TERMS']
        risk_id = row[RISK_ID_COL]
        
        print(f"Processing search term {idx + 1}/{len(search_terms_df)} (ID: {risk_id})")
        
        # Get Google News articles
        articles = get_google_news_articles(search_term, session, existing_links, MAX_ARTICLES_PER_TERM, now, yesterday)
        
        if not articles:
            print(f"  - No new articles found for this term")
            continue
        
        # IMPORTANT FOR OPTIMIZATION: process articles in parallel
        processed_articles = process_articles_batch(articles, config, analyzer, search_term, whitelist, risk_id)
        
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

def get_google_news_articles(search_term, session, existing_links, max_articles, now, yesterday):
    # extract articles from Google News using decoder
    articles = []
    try:
        print(f"  - searching: {search_term[:50]}...")
        
        # googlenewsdecoder
        gn = new_decoderv1.GoogleNewsDecoder()
        gn.set_search(search_term)
        gn.set_date(now.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d"))
        gn.set_pages(3)  # limit to 3 pages as in original
        
        # get search results
        results = gn.get_results()
        
        # KEY LOOP! Processes each result
        for result in results:
            url = result.get('link', '').strip()
            title = result.get('title', '').strip()
            
            # skip duplicates and invalid urls
            if not url or not title or url.lower().strip() in existing_links:
                continue
                
            # basic validation
            if len(title) < 10 or len(url) < 10:
                continue
            
            articles.append({
                'url': url,
                'title': title,
                'html': None  # will fetch during processing
            })
            
            # respect article limit
            if len(articles) >= max_articles:
                break
                
        print(f"  - found {len(articles)} new articles")
        return articles
        
    except Exception as e:
        print(f"  - error searching google news: {e}")
        return []

def process_articles_batch(articles, config, analyzer, search_term, whitelist, risk_id):
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