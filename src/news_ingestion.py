import feedparser
import ssl

# Fix for potential SSL fetch errors in some environments
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

def fetch_energy_news(limit=5):
    """
    Fetches the latest energy news headlines for Germany/Europe.
    Returns a list of strings (Headline - Source).
    """
    # Using Google News RSS for 'Germany Power Market'
    rss_url = "https://news.google.com/rss/search?q=Germany+Energy+Market+Prices&hl=en-US&gl=US&ceid=US:en"
    
    try:
        feed = feedparser.parse(rss_url)
        headlines = []
        for entry in feed.entries[:limit]:
            # Clean up title (Google News often adds ' - Source' at the end)
            title = entry.title
            pub_date = entry.published
            headlines.append(f"- {title} ({pub_date})")
            
        if not headlines:
            return ["No recent news found via RSS."]
            
        return headlines
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return [f"Error fetching news. Using system default context."]

if __name__ == "__main__":
    # Test
    news = fetch_energy_news()
    print("Latest Energy News:")
    for n in news:
        print(n)
