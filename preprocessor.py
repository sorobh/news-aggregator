import requests
from bs4 import BeautifulSoup
from transformers import GPT2Tokenizer
import openai

# Truncate text function
def truncate_text(text, max_length=1024):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]  # Truncate to the max_length
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

# Fetch full article function
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


# Generate summary function
def generate_summary(article_text):
    try:
        truncated_text = truncate_text(article_text)  # Truncate text if necessary
        prompt = f"Summarize the following article:\n\n{truncated_text}"

        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=150  # Adjust as needed
        )
        summary = response.choices[0].text.strip()
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Summary not available."


# Preprocess news items function
def preprocess_news_items(news_items):
    preprocessed_items = []
    for item in news_items:
        full_text = fetch_full_article(item['link'])
        summary = generate_summary(full_text) if full_text else item['summary']
        preprocessed_items.append({
            'title': item['title'],  # Include the title
            'text': summary,
            'source': item.get('link', 'Unknown Source')
        })
    return preprocessed_items


'''
# Not using for now
# Generate categories function
def generate_categories(news_items):
    titles = [item['title'] for item in news_items]
    concatenated_titles = ' '.join(titles)

    prompt = ("Identify the main themes or topics from the following news titles: "
              + concatenated_titles
              + ". List the themes in a few words for each, separated by commas.")

    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=100  # Adjust as needed
        )
        themes = response.choices[0].text.strip()
        return themes.split(',')
    except Exception as e:
        print(f"Error generating themes: {e}")
        return []

# Example usage
# categories = generate_categories(news_items)
'''