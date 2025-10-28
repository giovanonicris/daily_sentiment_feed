import time
import random
from googlenewsdecoder import new_decoderv1  # assumes you pip installed this already

# real encoded google news urls from rss (macroeconomic downturn, recent)
sample_urls = [
    "https://news.google.com/rss/articles/CBMisAFBVV95cUxPenRYbVVidzhLTkVVbXN2VHd4VjY3ZGlZWjBFa0NCQ2xPcEhneHlWZVhud3p2cUVMcWNsdkEzLTlpcmVxREZBbDBkOGFEa0wySEx4YkdoMkdzT3N6NXRnQ19waFNqM3BCdkE1bmczY1pHNVNxQkxoRnpkMm15cERnSklGLTc0ZlZvMzFxeEEtalR1UXF3R0VWNjZCWXFqMUk0VERZWHdpaHplWTdFdmlldQ?oc=5",
    "https://news.google.com/rss/articles/CBMixwFBVV95cUxORzRKNldHRmh3VDlNckN0UzRmTHZGQXZzLWstSjRjTU9KU2loWUhYWWI2X2dSOVZ4V0wyclkxcWV4LWZCUHVqRjIzLUFtR2pwd2xHNUo1RDRCSU11VDJVSkRNQkF3Q3pCbjFJOHBGQVJ5MUMxS0wzZE5wRnRHYUozeEdxanViVm1tMzh5T1ZTeWVJT0YtdC15X2xFaTB4dTlmTlFoYWlybXk1VklOZ19WNWZVR3ltLW05WlJpbWMycjE3QS1JQ1l3?oc=5",
    "https://news.google.com/rss/articles/CBMiXkFVX3lxTE5nMmdtb0NRTkNvNVczVE5VVERfTjhSbFhaZFR0VUE4LTNTUXhtNFpWZ0JZZVh0UXF4XzMwNWFTdlBLNC1NcFhvY3NIYWpPdnhQMktCWkJZT1FtZVp4Z2c?oc=5",
    "https://news.google.com/rss/articles/CBMitAFBVV95cUxPQXRFUW1jcHd2QmJFR0RQenJmNi15MktKcXY3WWlVVS1QUHp3VWYyUXMzNGtLZzQ2MU5DTXU0Z3dYY2Qta19kSEFyQWdWNE45NDNyRkdCMmUwY2x0ZXhzTHBrNWpQdmxBcDVUSDJPX1JjVUVVVm81UmU1YXBqNVQzdUJFMjdnajVaM2VGcDNOWW5vUEZheGxaVWlUMXNFdWU2S05UWjQyWkNUQ0ZmZ1VmaVJZbm4?oc=5",
    "https://news.google.com/rss/articles/CBMiiAFBVV95cUxNVThUZ2diSUNrLTZIVF92bUtEaUhNTF8yemRQcHlDSXBVTGJUZlZXTXFYWUlsaUZMUnBIQS1NcWxlNjh3TTB2YzFQYXIzVUlpRGpHdlZTaWpUN1ZpemhOTmtHZXRQOFJ4SEZ1alQ3Q1ZVdzkyTlR1TDJpbjR1ZW5ya3VjR1JMUkV4?oc=5"
]

print(f"Testing {len(sample_urls)} decodes...")

success_count = 0
for i, encoded_url in enumerate(sample_urls):
    try:
        print(f"Decode {i+1}: {encoded_url[:50]}...")
        decoded = new_decoderv1(encoded_url)
        
        # handle both str and dict (like in get_google_news_articles)
        if isinstance(decoded, dict):
            if decoded.get('status') and 'decoded_url' in decoded:
                print(f"  SUCCESS: {decoded['decoded_url'][:50]}...")
                success_count += 1
            else:
                print(f"  FAIL: {decoded.get('message', 'Bad dict format')}")
        elif isinstance(decoded, str):
            print(f"  SUCCESS: {decoded[:50]}...")
            success_count += 1
        else:
            print(f"  WEIRD: {type(decoded)}")
        
        # aggressive delay: 3-5s random per decode
        sleep_time = random.uniform(3, 5)
        print(f"  Sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)
        
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"Done: {success_count}/{len(sample_urls)} successes")