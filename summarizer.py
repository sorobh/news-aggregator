import openai
from apikey import apikey
from sentence_transformers import SentenceTransformer, util

openai.api_key = apikey

def summarize_news(news_content):
    # Instruct the model to group related news and summarize in consolidated bullet points
    prompt_text = [
        "Summarize the following news items as though a single journalist is telling daily highlights.\n"
        "Group all news items related to a single topic in a single paragraph so that there's no repetition of topics. \n"
        "If a topic combines 3 sources, then it should have 3 citations that link to those 3 sources. These linked citations should be placed at the end of the paragraph in markdown format and should only include sources from filtered_news\n"
        "Start new topic on new lines to indicate the previous topic has ended:\n\n"
        + news_content
    ]
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_text,
        max_tokens=900,  # Increased max_tokens to allow for a bit more detail
        temperature=0.2  # Adjust temperature if needed to balance creativity and coherence
    )
    return response.choices[0].text.strip()


# Instantiate the model (this can be done outside the function for efficiency)
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_citations(articles, sources, topics):
    # Generate embeddings for all topics
    topic_embeddings = model.encode(topics, convert_to_tensor=True)

    citations = []
    grouped_topics = {}
    for i, article in enumerate(articles):
        # Check if 'summary' key exists, otherwise use 'text' as a fallback
        if 'summary' in article and article['summary']:
            current_summary = article['summary']
        elif 'text' in article and article['text']:
            current_summary = article['text']
        else:
            current_summary = 'Summary not available.'
        
        # Check if 'topic' key exists, otherwise set it to 'Unknown'
        if 'topic' in article:
            current_topic = article['topic']
        else:
            current_topic = 'Unknown'

        # Find the most similar existing topic group
        current_embedding = topic_embeddings[i]
        max_similarity, max_index = 0, -1
        for topic_index, topic_embedding in grouped_topics.items():
            similarity = util.pytorch_cos_sim(current_embedding, topic_embedding).item()
            if similarity > max_similarity:
                max_similarity = similarity
                max_index = topic_index

        # Check if the article fits in an existing topic group
        if max_similarity > 0.9:  # Threshold for considering as the same topic
            current_topic = max_index
        else:
            current_topic = i
            grouped_topics[i] = current_embedding

        # Add source to sources dictionary if not present
        if article['source'] not in sources:
            sources[article['source']] = len(sources) + 1
        ref_number = sources[article['source']]

        # Create and add citation link
        citation_link = f"[{ref_number}]({article['source']})"
        
        # Ensure that there are enough elements in the citations list
        while len(citations) <= current_topic:
            citations.append("")

        # Generate the bullet points with citationscl
        citations[current_topic] += f"• {current_summary} {citation_link}"

    return citations
