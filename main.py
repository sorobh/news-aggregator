import ipdb;
import streamlit as st
from news_fetcher import fetch_news_from_rss
from preprocessor import preprocess_news_items
from summarizer import summarize_news
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

            # Create or Connect to Pinecone index
            index_name = 'news'
            dimension = len(numeric_embeddings[0])
            index = create_or_connect_index(index_name, dimension)
            print(f"Index object created for: {index_name}, Index object: {index}")

            # Insert embeddings into Pinecone
            article_ids = list(range(len(numeric_embeddings)))
            insert_embeddings(index, article_ids, numeric_embeddings)

            # Generate query embedding and retrieve related documents
            query_embedding = generate_embedding(user_query)
            retrieved_ids = retrieve_documents(index, query_embedding, top_k=10)

            # Convert retrieved string IDs to integers
            retrieved_ids_int = [int(i) for i in retrieved_ids]

            # Initialize related news content list
            related_news = []
            sources = {}  # To collect unique source links with reference numbers

            # Initialize citation links for the entire group
            group_citations = []

            for i, article_id in enumerate(retrieved_ids_int):
                if article_id < len(preprocessed_news):
                    article = preprocessed_news[article_id]
                    summary = summarize_news(article['text'])

                    # Add source link to sources dict if not already present
                    if article['source'] not in sources:
                        sources[article['source']] = len(sources) + 1  # Assign a new reference number

                    ref_number = sources[article['source']]

                    # Create a clickable citation with square brackets
                    citation_link = f"[{ref_number}]({article['source']})"
                    group_citations.append(citation_link)

                    # Append summary without individual citation to related news
                    summary_without_link = f"• {summary}"
                    related_news.append(summary_without_link)
                else:
                    print(f"Warning: Retrieved ID {article_id} is out of range for preprocessed news.")

            # Combine group citations and add them to the end of the bullet point
            if group_citations:
                group_citations_content = ' '.join(group_citations)
                summary_with_group_citation = f"{summary_without_link} {group_citations_content}"
                related_news[-1] = summary_with_group_citation  # Replace the last item in the list with the summary with group citation

            # Summarize and display the news
            if related_news:
                news_content = '\n'.join(related_news)
                st.subheader("Summary")
                st.markdown(news_content, unsafe_allow_html=True)  # Use markdown to render hyperlinks
            else:
                st.write("No relevant news articles found for summarization.")

    disconnect_from_pinecone()
    print("Disconnecting from Pinecone")

if __name__ == "__main__":
    main()