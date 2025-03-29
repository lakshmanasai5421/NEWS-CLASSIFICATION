import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string

# Load dataset
df = pd.read_csv('C:/xampp/htdocs/fake-news-detection/dataset/fake_or_real_news.csv')

# Check dataset balance
print("Before Balancing:")
print(df['label'].value_counts())

# Visualize Fake vs Real News Count
plt.figure(figsize=(6,4))
sns.countplot(x=df['label'])
plt.title("Fake vs Real News Count")
plt.xlabel("News Type (0 = Real, 1 = Fake)")
plt.ylabel("Count")
plt.show()

# Word Cloud for Fake News
fake_text = ' '.join(df[df['label'] == 1]['text'])
fake_wc = WordCloud(width=800, height=400, background_color='black').generate(fake_text)

plt.figure(figsize=(10,5))
plt.imshow(fake_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Fake News")
plt.show()

# Word Cloud for Real News
real_text = ' '.join(df[df['label'] == 0]['text'])
real_wc = WordCloud(width=800, height=400, background_color='white').generate(real_text)

plt.figure(figsize=(10,5))
plt.imshow(real_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Real News")
plt.show()
