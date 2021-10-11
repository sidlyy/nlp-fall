#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from collections import defaultdict
from pathlib import Path

## Loadable Files (I have shared with you some loadable files for you to test out)
## Silk RoadTopics - Sheet1
## C:/Users/Sid/Downloads/Silk RoadTopics - Sheet1.csv
## silkroadconvert.csv // FAIL
## 

## MAIN FILE: C:\Users\Sid\Desktop\Full CSV.csv
## copyable: C:/Users/Sid/Desktop/Full CSV.csv

## USING A SINGLE CSV
#papers = pd.read_csv('C:/Users/Sid/Desktop/Full CSV.csv')
#type(papers)


## USING A DATAFRAME
#Full text files
#my_dir_path = 'C:/Users/Sid//text_files_no_dupes'

#Individual subjects
#my_dir_path = 'C:/Users/Sid//Desktop/Cryptocurrency'
#my_dir_path = 'C:/Users/Sid//Desktop/Cybercrime'
#my_dir_path = 'C:/Users/Sid//Desktop/Darknet'
#my_dir_path = 'C:/Users/Sid//Desktop/Encryption & Tor'
#my_dir_path = 'C:/Users/Sid//Desktop/Hackers'
#my_dir_path = 'C:/Users/Sid//Desktop/Adult'
#my_dir_path = 'C:/Users/Sid//Desktop/Privacy'
#my_dir_path = 'C:/Users/Sid//Desktop/Black markets'

#Single video
my_dir_path = 'C:/Users/Sid//Desktop//Single video'


results = defaultdict(list)
for file in Path(my_dir_path).iterdir():
    with open(file, "r", encoding='latin-1') as file_open:
        results["file_name"].append(file.name)
        results["text"].append(file_open.read())
df = pd.DataFrame(results)
print(df.shape)
print(df)

papers = df["text"]

papers = papers.to_frame()
type(papers)


# In[2]:


papers.head()


# In[3]:


## run pip install WordCloud if you have not had this installed already! ##

# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(papers["text"].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width = 800, height = 800)
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# In[ ]:





# In[4]:


# These are the imports required to run our NLP libraries. The LDA model Gensim, the NLTK (a toolkit designed for natural language)
# and necessary details from these libraries, including a corpus. 
# A Corpus is a collection of documents, in this case, we are downloading the corpus of stopwords. 
# Stopwords are those words that do not provide any useful information to decide in which category a text should be classified.
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#### TODO: STOPWORDS ####
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'um', 'like', 'know', 'uh', 'think', 'thing', 
                   'gonna', 'want', 'txt', 'go', 'see', 'okay', 'going', 'one', 'get', 'dark', 'really', 'would', 'well', 'also', 
                   'lot', 'people', 'got', 'web', 'good', 'kind', 'let', 'oh', 'right', 'things', 
                   'kinds', 'yeah', 'us ', 'usually', 'way', 'even', 'much', 'actually', 'us', 'say', 'back', 'take', 'little', 
                   'would', 'well', 'no', 'basically', 'stuff', 'should', 'could', 'from', 'subject', 're', 'edu', 'use', 'um', 'like', 'know', 'uh', 'think', 'thing', 
                   'gonna', 'want', 'txt', 'go', 'see', 'okay', 'going', 'one', 'get', 'dark', 'really', 'would', 'well', 'also', 
                   'lot', 'people', 'got', 'web', 'good', 'kind', 'let', 'oh', 'right', 'things', 
                   'kinds', 'yeah', 'us ', 'usually', 'way', 'even', 'much', 'actually', 'us', 'say', 'back', 'take', 'little', 
                   'could', 'should', 'well', 'no', 'something', 'make', 'look', 'using', 'find', 'need', 'different', 'search', 'first', 'time', 'basically',
                   'used', 'maybe', 'many', 'might'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
data = papers.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])


# In[5]:


# The corpora module is Gensim's way of assigning a dictionary to their integer IDs for NLP tasks.
# In Gensim documentation: This module implements the concept of a Dictionary – a mapping between words and their integer ids.
import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])


# In[6]:


# The LDA model is complete after running this cell. 
# Here, we build an LDA model using Gensim's command and supply the necessary variables: the corpus, the id mappings,
# and the number of topics we have selected for it to generate.

# Each topic has a keyword, and we will print the generated topic keywords below with another command. 

# Increasing or decreasing the number of topics can correspondingly change the coherence score. 
# Coherence is likely to increase until a certain number of topics is selected, and then drop off. This number is 
# supposed to be generated by the cell at the very end, which plots the coherence scores on a Y axis and the number of topics
# on the x. It runs the model through each number of topics and records the scores to do this - which means
# it will likely take a long time to run for large document sets (corpus size).  

from pprint import pprint
# number of topics
num_topics = 200
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)


#If you would like to see all the top topics and their salient terms, which is done by the visualization normally, uncomment this. 
top_topics = lda_model.top_topics(corpus)
pprint(top_topics)

# Print the Keyword in the X topics
#pprint(lda_model.print_topics())
#doc_lda = lda_model[corpus]


# In[7]:


# This visualization I found online and is useful in navigating the complexity of the different topics and the relevant salient terms.
# You can adjust the relevance metric, 

#Visualization Cell
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pickle

# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)


# In[ ]:


LDAvis_prepared


# In[ ]:


#Coherence Score
from gensim.models import CoherenceModel

coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:


#Perplexity Score
print('\nPerplexity: ', lda_model.log_perplexity(corpus))


# In[ ]:


## This is failing for unknown reasons, it's supposed to automate us improving the model but I cannot get it to work. Unfinished as of 8/12/2021
import matplotlib as plt

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=40, step=6)
# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
# Select the model and print the topics
optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# In[ ]:




