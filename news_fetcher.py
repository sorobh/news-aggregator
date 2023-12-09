import feedparser

def fetch_news_from_rss(rss_urls, max_articles_per_feed=4):
    news_items = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_articles_per_feed]:  # Limit the number of articles
            news_items.append({
                'title': entry.title,
                'summary': 'Summary will be generated later',  # Placeholder for later summary generation
                'link': entry.link
            })
    return news_items

# Example usage
# news_items = fetch_news_from_rss(NEWS_SOURCES)
