import os
import json
import feedparser
from groq import Groq

class AIAnalyst:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Groq client: {e}")
        else:
            print("Warning: GROQ_API_KEY not found in environment variables.")

    def fetch_news(self, query="German Power Market", limit=5):
        """
        Fetches news from Google News RSS feed.
        """
        print(f"Fetching news for '{query}'...")
        encoded_query = query.replace(" ", "%20")
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(rss_url)
            news_items = [entry.title for entry in feed.entries[:limit]]
            return news_items
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def analyze_sentiment(self, news_items):
        """
        Analyzes the sentiment of provided news items using Groq (Llama3-70b/8b).
        """
        if not news_items:
            return {"sentiment_score": 0, "reasoning": "No news found."}
        
        if not self.client:
            return {"sentiment_score": 0, "reasoning": "Groq client not initialized (missing API key)."}

        prompt = f"""
        Analyze the sentiment of the following news headlines regarding the European/German Power Market:
        {json.dumps(news_items)}

        Provide a JSON response with:
        1. "sentiment_score": A float between -1.0 (Bearish/Oversupply) and 1.0 (Bullish/Shortage).
        2. "reasoning": A concise summary of why.
        
        JSON params only.
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert energy market analyst. Output valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-70b-8192",
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = chat_completion.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return {"sentiment_score": 0, "reasoning": "Error during AI analysis."}

if __name__ == "__main__":
    # Test
    ai = AIAnalyst()
    headlines = ai.fetch_news("German Energy Market")
    print("Headlines:", json.dumps(headlines, indent=2))
    
    if headlines:
        analysis = ai.analyze_sentiment(headlines)
        print("Analysis:", json.dumps(analysis, indent=2))
