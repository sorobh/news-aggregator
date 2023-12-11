import streamlit as st
from news_fetcher import fetch_news_from_rss
from preprocessor import preprocess_news_items
from summarizer import summarize_news, generate_citations
import config
import apikey
from pinecone_operations import (
    connect_to_pinecone, disconnect_from_pinecone, 
    create_or_connect_index, insert_embeddings, retrieve_documents, convert_string_embeddings_to_numeric_embeddings
)
from embedding_generator import generate_embedding

# Streamlit application
def main():
    connect_to_pinecone(apikey.pinecone_key, apikey.pinecone_env)
    print("Connection to Pinecone established")

    st.title("Daily News Summary")
    st.subheader("Categories: Tech, World, Politics, Sports, Culture")
    user_query = st.text_input("Enter the news category of interest")

    if st.button("Summarize Today's News"):
        with st.spinner('Fetching and summarizing news...'):
            # Fetch, and preprocess
            news_items = fetch_news_from_rss(config.NEWS_SOURCES)
            preprocessed_news = preprocess_news_items(news_items)

            # Generate embeddings for each news item
            news_embeddings = [generate_embedding(article['text']) for article in preprocessed_news]
            numeric_embeddings = convert_string_embeddings_to_numeric_embeddings(news_embeddings)
            topics = [article['topic'] for article in preprocessed_news]  # Extract topics from preprocessed news

            # Create or Connect to Pinecone index
            index_name = 'news'
            dimension = len(numeric_embeddings[0])
            index = create_or_connect_index(index_name, dimension)

            # Insert embeddings into Pinecone
            article_ids = list(range(len(numeric_embeddings)))
            insert_embeddings(index, article_ids, numeric_embeddings)

            # Generate query embedding and retrieve related documents
            query_embedding = generate_embedding(user_query)
            retrieved_ids = retrieve_documents(index, query_embedding, top_k=10)

            # Convert retrieved string IDs to integers
            retrieved_ids_int = [int(i) for i in retrieved_ids]

            # Filter preprocessed news items based on retrieved IDs
            filtered_news = [preprocessed_news[id] for id in retrieved_ids_int if id < len(preprocessed_news)]

            # Check for out-of-range IDs
            if len(filtered_news) < len(retrieved_ids_int):
                print("Warning: Some retrieved IDs are out of range for preprocessed news.")

            # Initialize sources dictionary for citations
            sources = {}

            # Generate citations for the filtered news items
            citations = generate_citations(filtered_news, sources, topics)

            # Display the news with citations and summaries
            if citations:
                # Generate summaries for the filtered news items
                summaries = [summarize_news(article['text']) for article in filtered_news]
                st.subheader("Summary")
                for i, (citation, summary) in enumerate(zip(citations, summaries)):
                    st.markdown(f"{i+1}. {summary}\n\n{citation}", unsafe_allow_html=True)
            else:
                st.write("No relevant news articles found for summarization.")

    disconnect_from_pinecone()
    print("Disconnecting from Pinecone")

if __name__ == "__main__":
    main()