import requests
from bs4 import BeautifulSoup
from transformers import GPT2Tokenizer
import openai

def truncate_text(text, max_length=1024):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

def fetch_full_article(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text(strip=True) for para in paragraphs])
        return article_text
    except Exception as e:
        print(f"Error fetching article: {e}")
        return ""

def generate_summary_and_topic(article_text):
    try:
        truncated_text = truncate_text(article_text)
        prompt = (
        f"Article:\n{truncated_text}\n\n"
        "Please provide a concise, informative summary of the article above in 2-3 sentences. "
        "Then, identify the main theme of the article in a few keywords. Format your response as follows:\n\n"
        "[Summary]\n"
        "Your summary here.\n\n"
        "[Main Theme]\n"
        "Keywords here."
        )
                
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=200
        )
        response_text = response.choices[0].text.strip()
        #print("Model Response:", response_text)

        # Extract summary and topic from response
        parts = response_text.split("\n")
        summary = parts[1] if len(parts) > 0 else "Summary not available."
        topic = parts[4] if len(parts) > 2 else "Unknown"

        return summary.strip(), topic.strip()
    except Exception as e:
        print(f"Error generating summary and topic: {e}")
        return "Summary not available.", "Unknown"

def preprocess_news_items(news_items):
    preprocessed_items = []
    for item in news_items:
        full_text = fetch_full_article(item['link'])
        if full_text:
            summary, topic = generate_summary_and_topic(full_text)
        else:
            summary, topic = item['summary'], 'General'
        preprocessed_items.append({
            'title': item['title'],
            'text': summary,
            'source': item.get('link', 'Unknown Source'),
            'topic': topic
        })
    return preprocessed_items
