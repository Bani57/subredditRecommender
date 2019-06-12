import numpy as np
import html
import re
from emoji_utils import *
from gensim.models import word2vec
import logging
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk import TextCollection
from nltk import pos_tag
from nltk import download

download('stopwords')
download('wordnet')

stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

special_tokens = ('URL', 'SUBREDDIT', 'USER', 'NUM', '.', ',', '?', '!', ':', '-', '+', '\n', '\r')


def get_words_from_reddit_text(text, case_sensitive=False):
    text = html.unescape(text)
    if not case_sensitive:
        text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
    text = re.sub(r'u/\w+', ' USER ', text)
    text = re.sub(r'r/\w+', ' SUBREDDIT ', text)
    text = re.sub(r'\d+(\.\d+)?', ' NUM ', text)
    emojis = get_unicode_emoji_from_text(text)
    emojis_concat = ''.join(emojis)
    for emoji in emojis:
        text = re.sub(emoji, ' ' + emoji + ' ', text)
    text = re.sub(r'[^\w\s.,?!:\-+\'' + emojis_concat + ']', ' ', text)
    text = re.sub(r'(.{1,5})\1+', r'\1\1', text)
    text = re.sub(r'\.', ' . ', text)
    text = re.sub(',', ' , ', text)
    text = re.sub(r'\?', ' ? ', text)
    text = re.sub('!', ' ! ', text)
    text = re.sub(':', ' : ', text)
    text = re.sub('-', ' - ', text)
    text = re.sub(r'\+', ' + ', text)
    text = re.sub(r'\'', ' \'', text)
    text = re.sub('\n', ' \n ', text)
    text = re.sub('\r', ' \r ', text)
    words = text.split(" ")
    words = [word for word in words if word != '']
    return words


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def post_process_words(words):
    processed = []
    for word in words:
        if word not in special_tokens and word not in stops:
            word_pos_tag = pos_tag([word, ])[0][1]
            processed.append(lemmatizer.lemmatize(word, get_wordnet_pos(word_pos_tag)))
        elif word in special_tokens:
            processed.append(word)

    return processed


def get_keywords_for_vocabularies(vocabularies):
    collection = TextCollection(list(vocabularies.values()))
    keywords = {}
    batch_size = len(vocabularies) // 20
    for index, (name, vocabulary) in enumerate(vocabularies.items()):
        if index % batch_size == 0:
            print("Processing vocabulary #" + str(index) + "...")
        vocabulary_words = list(set(vocabulary))
        num_tokens = len(vocabulary_words)
        if num_tokens > 1000:
            tokens_freq = FreqDist(vocabulary)
            first_keyword = 50
            vocabulary_words = tokens_freq.most_common(first_keyword + 1000)
            vocabulary_words = vocabulary_words[first_keyword:]
            vocabulary_words = [word for word, _ in vocabulary_words]
        tf_idfs = {word: collection.tf_idf(word, vocabulary) for word in vocabulary_words}
        tf_idf_values = list(tf_idfs.values())
        tf_idfs = sorted(tf_idfs.items(), key=lambda x: x[1], reverse=True)
        mean_tf_idf = np.mean(tf_idf_values)
        median_tf_idf = np.median(tf_idf_values)
        threshold = max((mean_tf_idf, median_tf_idf))
        keyword_list = [word for word, tf_idf_value in tf_idfs if tf_idf_value >= threshold]
        if len(keyword_list) > 30:
            keyword_list = keyword_list[:30]
        keywords[name] = keyword_list
    return keywords


def load_word2vec_google_model():
    return word2vec.Word2VecKeyedVectors.load_word2vec_format('models/word2vec_google.bin', binary=True)


def get_average_word2vec_keyword_embeddings(word2vec_model, keywords):
    keywords_embeddings = [list(word2vec_model.get_vector(keyword)) for keyword in keywords
                           if keyword in word2vec_model]
    if len(keywords_embeddings) > 0:
        features = list(np.mean(keywords_embeddings, axis=0))
    else:
        features = [0, ] * 300
    return features
