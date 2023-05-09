import pandas as pd
import praw
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import nltk
import plotly.io as pio
import plotly.graph_objs as go
import plotly.subplots as sp
import os
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from functions import preprocess_text, union_dataframes, sentiment_analysis
from pprint import pprint
from bertopic import BERTopic

# Uncomment if it is the first time running the script
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('vader_lexicon')


print('*****************')
print('Start Code')
print('*****************', end="\n\n")

# Create a directory with the current date
today = datetime.datetime.now().strftime('%Y-%m-%d')
output_directory = f'output/{today}'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

##################################################
# Post / Comment Scraper
##################################################

# define unique user agent, as per reddit API rule guidelines
user_agent = "windows:mdsa:v1.0 (by /u/datasci_guy)"

# initiate reddit instance
reddit = praw.Reddit(user_agent= user_agent,
                     client_id="P3uTH6wSXQJID55D18Y9jQ", client_secret="9rkhOaTSzcuybjAXgRlAW8mdv5swgQ")

# subreddits to scrape
subreddits = ["Calgary","Edmonton","vancouver","VictoriaBC","saskatoon","Winnipeg","toronto","ottawa","Hamilton","halifax"]

# create an empty DataFrame to hold all the data
city_df = pd.DataFrame(columns=["title", "content", "id", "date", "city"])
comment_df = pd.DataFrame(columns=["post_id", "body", "comment_id", "comment_score", 'date'])

for subreddit_name in subreddits:
    print(subreddit_name + "Subreddit")
    # get the subreddit object
    subreddit = reddit.subreddit(subreddit_name)
    
    # get the top 50 posts of the week
    top_posts = subreddit.top(limit=50, time_filter='week')
    
    # create a DataFrame for this subreddit's posts and comments
    df = pd.DataFrame(columns=["title", "content", "id", "date"])
    df_comment = pd.DataFrame(columns=["post_id", "body", "comment_id", "comment_score", 'date'])
    
    # loop through each post and add its data to the DataFrame
    post_num = 1
    for post in top_posts:
        counter = 0
        new_df = pd.DataFrame({"title":post.title, "content":post.selftext, "id":post.id, 
                               "date":datetime.datetime.utcfromtimestamp(post.created_utc)}, index=[0])
        df = pd.concat([df, new_df], ignore_index=True)
        print("Scraping Post " + str(post_num) + "/50")
        post_num += 1

        # loop through the top 10 comments of each post and add their data to the comment DataFrame
        for comment in post.comments:
            if counter == 10:
                break
            else:
                new_df_comment = pd.DataFrame({"post_id":post.id, "body":comment.body, "comment_id":comment.id, 
                               'comment_score':comment.score, "date":datetime.datetime.utcfromtimestamp(comment.created_utc)}, index=[0])
                df_comment = pd.concat([df_comment, new_df_comment], ignore_index=True)
            counter += 1
            print("Scraping Comment " + str(counter) + "/10")

    # add the city column to the post DataFrame
    df["city"] = subreddit_name
    
    # concatenate the post and comment DataFrames for this subreddit to the main DataFrames
    city_df = pd.concat([city_df, df])
    comment_df = pd.concat([comment_df, df_comment])

# reset the index of the main post DataFrame and combine the title and content columns into a single text column
city_df.reset_index(drop=True, inplace=True)
comment_df.reset_index(drop=True, inplace=True)
city_df['text'] = city_df['title'] + city_df['content']

# Deleting rows with no data
city_df = city_df[city_df['id'].str.len().isin([6, 7])]
comment_df = comment_df[comment_df['post_id'].str.len().isin([6, 7])]

print('*****************')
print('Reddit API Done')
print('*****************', end="\n\n")


##################################################
# Sentiment Analysis
##################################################

print('*****************')
print('Performing Sentiment Analysis')
print('*****************', end="\n\n")

# Clean data
city_df['text'] = city_df['text'].fillna(value='')
city_df['text'] = city_df['text'].astype('str')
city_df['city'] = city_df['city'].str.lower()
city_df.dropna(subset=['city'], inplace=True)
city_df['city'] = city_df['city'].map({'victoriabc':'victoria'}).fillna(city_df['city'])
comment_df['body'] = comment_df['body'].fillna(value='')
comment_df['body'] = comment_df['body'].astype('str')

# Generating a column that only has dates
city_df['date_8d'] = pd.to_datetime(city_df['date'])
city_df['date_8d'] = city_df['date_8d'].dt.date
comment_df['date_8d'] = pd.to_datetime(comment_df['date'])
comment_df['date_8d'] = comment_df['date_8d'].dt.date

# Get text ready for sentiment Analysis
city_df['cleaned_text'] = city_df['text'].apply(preprocess_text)
comment_df['cleaned_text'] = comment_df['body'].apply(preprocess_text)

# Sentiment Analysis
city_df[['neg', 'neu', 'pos', 'compound']] = city_df['cleaned_text'].apply(sentiment_analysis).apply(pd.Series)
comment_df[['neg', 'neu', 'pos', 'compound']] = comment_df['cleaned_text'].apply(sentiment_analysis).apply(pd.Series)

# Append historic data with weekly data
city_df = union_dataframes(city_df, 'Data/posts.csv', "id")
comment_df = union_dataframes(comment_df, 'Data/comments.csv', "comment_id")

# Write data into a csv file
comment_df.to_csv('Data/comments.csv', index=False)
city_df.to_csv('Data/posts.csv', index=False)
print('*****************')
print('Sentiment Analysis Done')
print('*****************', end="\n\n")

##################################################
# Topic Modelling
##################################################

print('*****************')
print('Performing Topic Modeling')
print('*****************', end="\n\n")

########################
# Pre-Processing
########################

# Load the data
post_df = pd.read_csv('./Data/posts.csv', encoding='ISO-8859-1')

# Create a list of stop words from the NLTK library
stop_words = stopwords.words('english')

# Add custom stop words to the list
custom_stop_words = ['calgari', 'edmonton', 'vancouv', 'victoria', 'saskatoon', 'winnipeg', 'toronto', 'ottawa', 'hamilton', 'halifax', 'https']

# Extend the stop words list with the custom stop words
stop_words.extend(custom_stop_words)


def remove_stopwords(text, stop_words):
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(cleaned_words)


# Remove float datatypes
post_df['cleaned_text'] = post_df['cleaned_text'].astype(str)

# Remove stopwords from the text
post_df['cleaned_text'] = post_df['cleaned_text'].apply(lambda x: remove_stopwords(x, stop_words))

########################
# Model Fit
########################

# Create BERTopic instance
topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)

# Fit the model to our data
topics, probabilities = topic_model.fit_transform(post_df['cleaned_text'])

# Assign topics to DataFrame
post_df['topic'] = topics

# Save the model
topic_model.save("my_bertopic_model")

########################
# If code below fails, run pre-fit_topic_model.py
########################

########################
# Topic Visualizing
########################

# Get the top 10 most common topics for each subreddit
top_topics_per_subreddit = post_df.groupby(["city", "topic"]).size().reset_index(name="count")
top_topics_per_subreddit = top_topics_per_subreddit.groupby("city").apply(lambda x: x.nlargest(10, "count")).reset_index(drop=True)

# Create a pivot table with the city as index, topic as columns, and count as values
pivot_table = top_topics_per_subreddit.pivot_table(index='city', columns='topic', values='count', fill_value=0)

# Visualize Topics
topic_figure = topic_model.visualize_topics()
pio.write_html(topic_figure, file=f"{output_directory}/intertopic_distance.html")  # Save as an interactive HTML file

# Generate hierarchical topics
hierarchical_topics = topic_model.hierarchical_topics(post_df['cleaned_text'])

# Visualize hierarchy
hierarchy_figure = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
pio.write_html(hierarchy_figure, file=f"{output_directory}/topic_hierarchy_tree.html")  # Save as an interactive HTML file

# Get the topic information
topic_info = topic_model.get_topic_info()

# Save topic information to a CSV file
topic_info.to_csv(f"{output_directory}/topic_info.csv", index=False)

# Save the new posts.csv
post_df.to_csv('./Data/posts.csv')

# Create a pivot table with the city as index, topic as columns, and count as values
pivot_table = top_topics_per_subreddit.pivot_table(index='city', columns='topic', values='count', fill_value=0)

########################
# City-Topic Chart
########################

# Get topic words
def get_topic_words(topic_model, topic_id):
    topic_words = topic_model.get_topic(topic_id)
    words = ', '.join([word for word, _ in topic_words])
    return words

# Convert pivot table to long format
long_df = pivot_table.unstack().reset_index(name='count')

# Filter the long_df DataFrame to exclude topics containing the word "nan"
long_df = long_df[~long_df['topic']==-1]

# Create a trace for each topic
traces = []
for city in long_df['city'].unique():
    city_topics = long_df[long_df['city'] == city].sort_values('count', ascending=False).head(10)['topic']
    for topic in city_topics:
        topic_words = get_topic_words(topic_model, topic)
        traces.append(
            go.Bar(
                x=[city],
                y=long_df[(long_df['topic'] == topic) & (long_df['city'] == city)]['count'],
                name=f'Topic {topic}',
                hovertext=[topic_words] * len(long_df[(long_df['topic'] == topic) & (long_df['city'] == city)]['count']),
                text=[f'{topic}'] * len(long_df[(long_df['topic'] == topic) & (long_df['city'] == city)]['count']),
                hovertemplate='Topic %{text}: %{hovertext}<br>Count %{y}<extra></extra>',
                showlegend=False
            )
        )


# Create layout and figure
layout = go.Layout(
    title='Top 10 Most Common Topics by City',
    xaxis=dict(title='City'),
    yaxis=dict(title='Topic Frequency'),
    barmode='stack',
    height=800
)
fig = go.Figure(data=traces, layout=layout)

# Save figure as interactive HTML
pio.write_html(fig, file="topics_by_city_stacked_bar.html")

########################
# Topic WordClouds
########################

top_n_words = 20  # number of words to display in each word cloud

# Get the top 20 topics and their most frequent words
topic_words = topic_model.get_topic_freq()
top_20_topics = topic_words[topic_words['Topic'].isin(post_df['topic'].unique())].iloc[1:21]['Topic'].tolist()
# Modify the top_20_topics list to exclude the topics containing the word "nan"
#top_20_topics = [t for t in top_20_topics if t not in nan_topics]
topic_word_dict = {}

for topic in top_20_topics:
    words = topic_model.get_topic(topic)[:top_n_words]
    word_dict = {word: importance for word, importance in words}
    topic_word_dict[topic] = word_dict

# Create an array of word clouds for each topic
fig, axs = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle('Word Clouds of Top 20 Topics', fontsize=30)

for i, ax in enumerate(axs.flat):
    if i < len(top_20_topics):
        topic = top_20_topics[i]
        words = topic_word_dict[topic]
        wc = WordCloud(background_color='white', width=800, height=400)
        wc.generate_from_frequencies(words)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f'Topic {topic}', fontsize=20)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()

# Save the WordCloud plot
plt.savefig(f"{output_directory}/topic_word_clouds.png", dpi=300)

########################
# Topic Sentiment
########################

# Load DF's
post_df = pd.read_csv('./Data/posts.csv')
comment_df = pd.read_csv('./Data/comments.csv')

# Clean comment DF
comment_df = comment_df[~comment_df['post_id'].str.contains('Ã')]
comment_df = comment_df.dropna(subset=['post_id','body'])
comment_df = comment_df[~comment_df['body'].str.contains('Ã')]
comment_df = comment_df[comment_df['compound'] != 0]
# group by post_id, calc mean
comment_sent = comment_df.groupby('post_id').agg(compound_avg=('compound','mean'))

# Reset index
comment_sent = comment_sent.reset_index()

# Merge post_df with comment_sent on 'post_id'
result_df = post_df.merge(comment_sent, left_on='id', right_on='post_id', how='left')

# Rename 'compound_avg' column to 'comment_sent'
result_df.rename(columns={'compound_avg': 'comment_sent'}, inplace=True)

# Calculate difference between post sentiment and comment sentiment, find mean
result_df['sent_diff'] = result_df['compound'] - result_df['comment_sent']
result_df['sent_diff'].mean()

topic_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Get the top words for each topic
top_words = 5
topic_titles = []
for topic in topic_list:
    words = topic_model.get_topic(topic)[:top_words]
    topic_words = ', '.join([word for word, _ in words])
    topic_titles.append(f'Topic {topic}: {topic_words}')

# Create a subplot with 20 rows and 1 column
fig = sp.make_subplots(rows=20, cols=1, subplot_titles=topic_titles, vertical_spacing=0.03)

# Create a box and whisker plot for each topic from 0 to 19
for i, topic in enumerate(topic_list):
    # Filter to only posts with the current topic
    topic_posts = result_df[post_df['topic'] == topic]

    # Create the box and whisker plot and add it to the subplot
    for city in topic_posts['city'].unique():
        city_data = topic_posts[topic_posts['city'] == city]
        fig.add_trace(go.Box(y=city_data['comment_sent'], x=city_data['city'], name=city, showlegend=False), row=i + 1, col=1)

# Update the layout
fig.update_layout(height=4000, width=1000, title_text='Box and Whisker Plots for Topics by City')

# Save the figure
fig.write_html(f"{output_directory}/city_topic_sentiment_boxAndWhisker.html")

# save results to CSV
result_df.to_csv('post_df_with_comment_sens.csv')

print('*****************')
print('All Tasks Completed.')
print('*****************', end="\n\n")
input("Close?")