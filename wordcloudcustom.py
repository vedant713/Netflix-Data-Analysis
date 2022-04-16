import pandas as pd
import matplotlib.pyplot as plt
import numpy
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
from PIL import Image
import random
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import seaborn as sns
import requests
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from collections import Counter
def ngrams_count(corpus, ngram_range=(1, 1), n=-1, cached_stopwords=stopwords.words('english')):
    """
    Applies text transformation for counting words in a n_gram range
    
    Parameters
    ----------
    :param corpus: text list to be analysed [type: list or pd.Series]
    :param ngram_range: ngrams to be extracted from corpus [type: tuple, default=(1, 1)]
    :param n: limits the returning of only the top N ngrams [type: int, default=-1]
        *in case of n=-1, all ngrams will be returned
    :param cached_stopwords: stopwords to be used on filtering words 
        *[type: list, default=stopwords.words('english')]
        
    Return
    ------
    :return df_count: DataFrame with columns "ngram" and "count" [type: pd.DataFrame]
    
    Application
    -----------
    df_count = ngrams_count(corpus=df['text_attribute'])
    """
    
    # Using CountVectorizer to build a bag of words using the given corpus
    vectorizer = CountVectorizer(stop_words=cached_stopwords, ngram_range=ngram_range).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    
    # Summing words and generating a frequency list
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    total_list = words_freq[:n]
    
    # Returning a DataFrame with the ngrams count
    return pd.DataFrame(total_list, columns=['ngram', 'count'])
def generate_wordcloud(corpus, ngram_range=(1, 1), n=-1, cached_stopwords=stopwords.words('english'),
                       **kwargs):
    """
    Applies a ngram count and generates a wordcloud object using a counter dictionary
    
    Parameters
    ----------
    :param corpus: text list to be analysed [type: list or pd.Series]
    :param ngram_range: ngrams to be extracted from corpus [type: tuple, default=(1, 1)]
    :param n: limits the returning of only the top N ngrams [type: int, default=-1]
        *in case of n=-1, all ngrams will be returned
    :param cached_stopwords: stopwords to be used on filtering words 
        *[type: list, default=stopwords.words('english')]
    :param **kwargs: additional parameters
        :arg width: wordcloud width [type: int, default=1280]
        :arg height: wordcloud height [type: int, default=720]
        :arg random_state: random seed for word positioning [type: int, defualt=42]
        :arg colormap: colormap for wordcloud chart [type: string, default='viridis']
        :arg background_color: wordcloud background color [type: string, default='white']
        :arg mask: either an internet image url or an image array for using as mask
            *[type: string or array, default=None]
        
    Return
    ------
    :return wordcloud: WordCloud object [type: wordcloud.WordCloud]
    
    Application
    -----------
    wordcloud = generate_wordcloud(corpus=df['text_attribute'])
    """
    
    # Generating a DataFrame with ngrams count
    df_count = ngrams_count(corpus=corpus, ngram_range=ngram_range, n=n, 
                            cached_stopwords=cached_stopwords)
    
    # Transforming the ngram count into a dictionary
    words_dict = {w: c for w, c in df_count.loc[:, ['ngram', 'count']].values}
    
    # Extracting kwargs for generating a wordcloud
    width = kwargs['width'] if 'width' in kwargs else 1280
    height = kwargs['height'] if 'height' in kwargs else 720
    random_state = kwargs['random_state'] if 'random_state' in kwargs else 42
    colormap = kwargs['colormap'] if 'colormap' in kwargs else 'viridis'
    background_color = kwargs['background_color'] if 'background_color' in kwargs else 'white'
    
    # Creating a mask if applicable
    mask = kwargs['mask'] if 'mask' in kwargs else None
    try:
        if type(mask) == str and mask is not None:
            # Requesting the image url using requests and transforming it using PIL
            img = Image.open(requests.get(mask, stream=True).raw)
            mask_array = np.array(img)
            
            # If mask array is a 3-dimensional array, transformes it into a 2-dimensional
            if len(mask_array.shape) == 3:
                mask_array = mask_array[:, :, -1]
            
            # Creating a transformarion mask and changing pixels on it
            transf_mask = np.ndarray((mask_array.shape[0], mask_array.shape[1]), np.int32)
            for i in range(len(mask_array)):
                transf_mask[i] = [255 if px == 0 else 0 for px in mask_array[i]]

        # If mask argument is already given as an array
        else:
            transf_mask = mask
            
    except Exception as e:
        # Error on requesting or preparing the mask - wordcloud will be generated without it
        print(f'Error on requesting or preparing mask. WordCloud will be generated without mask')
        transf_mask = None
        
    
    # Generating wordcloud
    wordcloud = WordCloud(width=width, height=height, random_state=random_state, colormap=colormap, 
                          background_color=background_color, mask=transf_mask).generate_from_frequencies(words_dict)
    
    return wordcloud
def plot_wordcloud(corpus, ngram_range=(1, 1), n=-1, cached_stopwords=stopwords.words('english'),
                   **kwargs):
    """
    Generates a ngram count and a wordcloud object for plotting a custom wordcloud chart
    
    Parameters
    ----------
    :param corpus: text list to be analysed [type: list or pd.Series]
    :param ngram_range: ngrams to be extracted from corpus [type: tuple, default=(1, 1)]
    :param n: limits the returning of only the top N ngrams [type: int, default=-1]
        *in case of n=-1, all ngrams will be returned
    :param cached_stopwords: stopwords to be used on filtering words 
        *[type: list, default=stopwords.words('english')]
    :param **kwargs: additional parameters
        :arg width: wordcloud width [type: int, default=1280]
        :arg height: wordcloud height [type: int, default=720]
        :arg random_state: random seed for word positioning [type: int, defualt=42]
        :arg colormap: colormap for wordcloud chart [type: string, default='viridis']
        :arg background_color: wordcloud background color [type: string, default='white']
        :arg mask: either an internet image url or an image array for using as mask
            *[type: string or array, default=None]
        :arg figsize: figure dimension [type: tuple, default=(20, 17)]
        :arg ax: matplotlib axis in case of external figure defition [type: mpl.Axes, default=None]
        :arg title: chart title [type: string, default=f'Custom WordCloud Plot']
        :arg size_title: title size [type: int, default=18]
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved 
            [type: string, default='wordcloud.png']
        
    Return
    ------
    This function returns nothing besides the plot of a custom wordcloud
    
    Application
    -----------
    plot_wordcloud(corpus=df['text_attribute'])
    """
    
    # Extracting kwargs for generating a wordcloud
    width = kwargs['width'] if 'width' in kwargs else 1280
    height = kwargs['height'] if 'height' in kwargs else 720
    random_state = kwargs['random_state'] if 'random_state' in kwargs else 42
    colormap = kwargs['colormap'] if 'colormap' in kwargs else 'viridis'
    background_color = kwargs['background_color'] if 'background_color' in kwargs else 'white'
    mask = kwargs['mask'] if 'mask' in kwargs else None
    
    # Generating a pre configured wordcloud
    wordcloud = generate_wordcloud(corpus=corpus, ngram_range=ngram_range, n=n, 
                                   cached_stopwords=cached_stopwords, width=width, height=height,
                                   random_state=random_state, colormap=colormap, 
                                   background_color=background_color, mask=mask)
    
    # Extracting kwargs for figure plotting
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (20, 17)
    ax = kwargs['ax'] if 'ax' in kwargs else None
    title = kwargs['title'] if 'title' in kwargs else f'Custom WordCloud Plot'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 18
    
    # Creating figure and plotting wordcloud
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)  
    ax.imshow(wordcloud)
    ax.axis('off')
    ax.set_title(title, size=size_title, pad=20)
    
    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'wordcloud.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)