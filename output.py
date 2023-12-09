def save_summary_to_file(summary, filename="daily_news_summary.txt"):
    with open(filename, 'w') as file:
        file.write(summary)

# Example usage
# save_summary_to_file(summary)
