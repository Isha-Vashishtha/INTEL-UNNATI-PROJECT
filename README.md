
# INTEL-UNNATI-PROJECT
# This project aims to explore consumer sentiment toward Intel products through the use of online surveys. By pre-processing and analyzing survey data, the project seeks to uncover insights into customer satisfaction, common themes, and overall emotional dynamics The survey includes natural language processing (NLP) techniques diversity, sentiment analysis modeling, and data visualization to provide a comprehensive understanding of customer feedback

The main features of this project are:

Pre-processing data:

Refining and generalizing review writing.
Detailing interruptions and correcting spelling errors.
Translation of non-English research into English.
Emotional Analysis:

Using TextBlob and VADER for emotional scoring.
List the positive, negative, and neutral emotions in the reviews.
Image credits:

Creating word clouds to highlight common words in reviews.
Long-term cognitive development planning.
Imagine the distribution of the emotion across product lines and geographic locations.
Clustering and Feature Analysis:

Use of TF-IDF vectorization and K-Means clustering to identify common themes in studies.
Analyzing mentions of specific technical features such as battery life, screen quality, performance and camera.
Suggestions:

To deliver actionable insights based on sentiment analysis to help improve Intel’s products and customer satisfaction.
By automating sentiment analysis of online reviews, this business helps customers identify pain points and areas of excellence, guide product development and make strategic decisions.


#Table Of Contents
- Introduction
- Literature Review
- Data Collection
- Data Preprocessing
- Sentiment Analysis Methodology
- Implementation
- Result and Discussion
- Conclusion


#USAGE STEPS:
1. Load the dataset
  import pandas as pd
  df = pd.read_csv('/content/reviews_combinedfirst.csv')
  print(df.head())  # Display the first few rows of the dataframe

2. Preprocessing Reviews
 
def preprocess_text(text):  # Function to preprocess text
    # Preprocessing steps
    # ...

df['Cleaned_Review'] = df['Review'].apply(preprocess_text)
print(df[['Review', 'Cleaned_Review']].head())

3. Sentiment Analysis
   1. Using TextBlob
      from textblob import TextBlob


def perform_sentiment_analysis(text_data):# Function to perform sentiment analysis using TextBlob
    sentiment_scores = []
    for review in text_data:
        if isinstance(review, str):
            blob = TextBlob(review)
            sentiment_scores.append(blob.sentiment.polarity)
        else:
            sentiment_scores.append(None)
    return sentiment_scores

df['TextBlob_Sentiment_Score'] = perform_sentiment_analysis(df['Cleaned_Review'])

  2. Using VADER
     from nltk.sentiment.vader import SentimentIntensityAnalyzer


def perform_vader_sentiment_analysis(text_data):  # Function to perform sentiment analysis using VADER
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for review in text_data:
        if isinstance(review, str):
            scores = sid.polarity_scores(review)
            sentiment_scores.append(scores['compound'])
        else:
            sentiment_scores.append(None)
    return sentiment_scores

df['VADER_Sentiment_Score'] = perform_vader_sentiment_analysis(df['Cleaned_Review'])
df['Sentiment_Category'] = df['VADER_Sentiment_Score'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))

4. Using Visualization
    a.Sentiment Over Time
    import matplotlib.pyplot as plt


def plot_sentiment_over_time(df):  # Function to plot sentiment over time
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')
    sentiment_over_time = df['VADER_Sentiment_Score'].resample('M').mean()
    sentiment_over_time.plot()
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.title('Sentiment Over Time')
    plt.show()

plot_sentiment_over_time(df)

b. Word Cloud
   from wordcloud import WordCloud


def plot_wordcloud(text, title):  # Function to plot word cloud
    if isinstance(text, str):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()
    else:
        print(f"No valid text data for '{title}' word cloud.")


positive_reviews = df[df['Sentiment_Category'] == 'Positive'] # Generate word clouds for positive, negative, and neutral reviews
negative_reviews = df[df['Sentiment_Category'] == 'Negative']
neutral_reviews = df[df['Sentiment_Category'] == 'Neutral']

plot_wordcloud(' '.join(positive_reviews['Cleaned_Review'].dropna()), 'Word Cloud of Positive Reviews')
plot_wordcloud(' '.join(negative_reviews['Cleaned_Review'].dropna()), 'Word Cloud of Negative Reviews')
plot_wordcloud(' '.join(neutral_reviews['Cleaned_Review'].dropna()), 'Word Cloud of Neutral Reviews')

5. Clustering Reviews
    from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns

# Ensure no NaN values in 'Cleaned_Review' column
df['Cleaned_Review'].fillna('', inplace=True)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Cleaned_Review'])

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Plot clusters
sns.countplot(x='Cluster', data=df)
plt.title('Cluster Counts')
plt.show()

# Print top terms per cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(3):
    print(f"Cluster {i}: ", end='')
    for ind in order_centroids[i, :10]:
        print(f'{terms[ind]} ', end='')
    print()




