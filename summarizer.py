import openai
from apikey import apikey

openai.api_key = apikey

def summarize_news(news_content):
    # Instruct the model to group related news and summarize in consolidated bullet points
    prompt_text = [
        "Summarize the following news items into bullet points as though a single jounrlaist is writing in a concise manner and a formal tone.\n"
        "Group all related news items in a single bullet so that there's no repetition of topics. \n"
        "Each bullet point should start on a separate line indicating it's a new topic.\n"
        "Ensure each bullet ends with citation number(s) that are linked. If the bullet point combines 3 articles, then it should have 3 citations that link to those 3 articles. Citations to be placed at the end of individual bullet points in markdown format and not at the bottom as footnotes:\n\n"
        + news_content
    ]
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_text,
        max_tokens=900,  # Increased max_tokens to allow for a bit more detail
        temperature=0.2  # Adjust temperature if needed to balance creativity and coherence
    )
    return response.choices[0].text.strip()

