import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import re


# 1. Load the sentiment dataset
df = pd.read_csv("3) Sentiment dataset.csv")

print("First 5 rows:")
print(df.head())


# 2. Check dataset information
print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# 3. Remove duplicate rows
df = df.drop_duplicates()


# 4. Remove rows with missing text
df = df.dropna(subset=["Text"])


# 5. Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["Cleaned_Text"] = df["Text"].apply(clean_text)


# 6. Sentiment analysis using TextBlob
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


df["Predicted_Sentiment"] = df["Cleaned_Text"].apply(get_sentiment)


# 7. Display sentiment results
print("\nSentiment Analysis Results:")
print(
    df[["Text", "Sentiment", "Predicted_Sentiment"]].head(10)
)


# 8. Sentiment distribution
sentiment_counts = df["Predicted_Sentiment"].value_counts()

print("\nPredicted Sentiment Distribution:")
print(sentiment_counts)


# 9. Visualize sentiment distribution
plt.figure(figsize=(8, 5))

sentiment_counts.plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Texts")
plt.xticks(rotation=0)
plt.grid(axis="y")

plt.tight_layout()
plt.show()


# 10. Word frequency analysis
all_text = " ".join(df["Cleaned_Text"])

words = all_text.split()

word_counts = Counter(words)

print("\nTop 20 Most Frequent Words:")

for word, count in word_counts.most_common(20):
    print(word, ":", count)


# 11. Generate Word Cloud
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(all_text)


plt.figure(figsize=(12, 6))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title("Word Cloud of Sentiment Dataset")

plt.tight_layout()
plt.show()


# 12. Save the processed dataset
df.to_csv("sentiment_results.csv", index=False)

print("\n--------------------------------")
print("Sentiment Analysis Completed!")
print("--------------------------------")
print("Processed dataset saved as: sentiment_results.csv")