import os
from datetime import datetime
import json
import pandas as pd
import tomotopy as tp
import numpy as np
import codecs
import re
import little_mallet_wrapper as lmw


def cleaning_docs(_df, _clean_docs_file):
    _clean_docs_d = {}  # dictionary of documents to perform topic modeling on, here documents are posts' clean sentences
    for index, row in _df.iterrows():  # iterating over posts
        clean_doc_st = lmw.process_string(row['text'])
        if clean_doc_st and 'bot' not in row['author'] and 'Bot' not in row['author']:  # exclude posts authored by bots
            clean_doc_l = [t for t in clean_doc_st.split(' ') if t != 'didn' and t != 'doesn']
            if len(clean_doc_l) > 5 and 'bot' not in clean_doc_l \
                    and 'torrent' not in clean_doc_l:  # exclude posts that are less than 5 words
                # add post id and clean post to the dictionary as a key,value pair
                # the clean - tokenized and lemmatized - posts are our documents
                _clean_docs_d[row['id']] = [clean_doc_l, row['text']]  # row['url'] in case you need url

    with open(_clean_docs_file, 'w') as jsonfile:  # creating a file with the dict of documents to topic model
        json.dump(_clean_docs_d, jsonfile)

    return _clean_docs_d


def perform_tm(_doc_ids, _clean_docs, _num_topics, _rm_top, _topwords_file, _saved_model):
    # setting and loading the LDA model
    lda_model = tp.LDAModel(k=_num_topics,  # number of topics in the model
                            min_df=3,  # remove words that occur in less than n documents
                            rm_top=_rm_top)  # remove n most frequent words
    vocab = set()
    for doc in _clean_docs:
        lda_model.add_doc(doc)  # adding document to the model
        vocab.update(doc)  # adding tokens in the document to the vocabulary
    print('Num docs:{}'.format(len(lda_model.docs)))
    print("Vocabulary Size: {}".format(len(list(vocab))))
    print('Removed Top words: ', lda_model.removed_top_words)

    iterations = 10
    for i in range(0, 100, iterations):  # train model 10 times with 10 iterations at each training = 100 iterations
        lda_model.train(iterations)
        print(f'Iteration: {i}\tLog-likelihood: {lda_model.ll_per_word}')
    lda_model.save(_saved_model, full=True)

    # TOP WORDS
    num_top_words = 25  # number of top words to print for each topic
    with open(_topwords_file, "w", encoding="utf-8") as file:
        file.write(
            f"\nTopics in LDA model: {_num_topics} topics {_rm_top} removed top words\n\n")  # write settings of the model in file
        topic_individual_words = []
        for topic_number in range(0, _num_topics):  # for each topic number in the total number of topics
            topic_words = ' '.join(  # string of top words in the topic
                word for word, prob in lda_model.get_topic_words(topic_id=topic_number,
                                                                 top_n=num_top_words))  # get_topic_words is a tomotopy function that returns a dict of words and their probabilities
            topic_individual_words.append(topic_words.split(' '))  # append list of the topic's top words for later
            file.write(f"Topic {topic_number}\n{topic_words}\n\n")  # write topic number and top words in file

    # TOPIC DISTRIBUTIONS
    topic_distributions = [list(doc.get_topic_dist()) for doc in
                           lda_model.docs]  # list of lists of topic distributions for each document, get_topic_dist() is a tomotopy function
    topic_results = []
    for topic_distribution in topic_distributions:  # list of dicts of documents' topic distributions to convert into pandas' dataframe
        topic_results.append({'topic_distribution': topic_distribution})
    df = pd.DataFrame(topic_results,
                      index=_doc_ids)  # df where each row is the list of topic distributions of a document, s_ids are the ids of the sentences
    column_names = [f"Topic {number}" for number, topic in
                    enumerate(topic_individual_words)]  # create list of column names from topic numbers and top words
    df[column_names] = pd.DataFrame(df['topic_distribution'].tolist(),
                                    index=df.index)  # df where topic distributions are not in a list and match the list of column names
    df = df.drop('topic_distribution', axis='columns')  # drop tentativo topic distributions' column
    dominant_topic = np.argmax(df.values, axis=1)  # get dominant topic for each document
    df['dominant_topic'] = dominant_topic  # add column for the dominant topic in the document

    return df


def main(subreddit, _dataset_path, tomo_folder):
    
    df = pd.read_pickle(_dataset_path) # read data as df
    
    clean_docs_file = os.path.join(tomo_folder, f'{subreddit}_clean.json')  # file with clean documents
    if not os.path.exists(clean_docs_file):  # if clean documents file doesn't exist, executes data cleaning
        start = datetime.now()
        print("Data Cleaning...")
        clean_docs_dict = cleaning_docs(df, clean_docs_file)
        print(f'{str(datetime.now())}________________{str(datetime.now() - start)}\n')  # print timing of data cleaning
    else:
        with open(clean_docs_file) as json_file:
            clean_docs_dict = json.load(json_file)
    doc_ids = [doc_id for doc_id in clean_docs_dict.keys()]  # get list of document ids for later
    print(len(doc_ids))
    clean_docs = [sent_url[0] for sent_url in clean_docs_dict.values()]  # get list of clean documents for later
    og_docs = [[sent_url[1]] for sent_url in clean_docs_dict.values()]  # get list of original documents for later
    # doc_urls = [sent_url[2] for sent_url in clean_docs_dict.values()]  # get list of document urls for later

    for num_topics in [5]:  # for number of topics - for loops to run multiple models with different settings with one execution
        for rm_top in [10]: #, 20, 30]:  # for number of most frequent words to remove
            topwords_file = os.path.join(tomo_folder,
                                         f'{subreddit}-{num_topics}_{rm_top}.txt')  # path for top words file
            docterm_file = os.path.join(tomo_folder,
                                        f'{subreddit}-{num_topics}_{rm_top}.pkl')  # path for doc-topic matrix file
            saved_model = os.path.join(tomo_folder,
                                       f'{subreddit}-{num_topics}_{rm_top}.bin')  # path for model file

            if not os.path.exists(topwords_file) or not os.path.exists(docterm_file):  # if result files don't exist, performs topic modeling
                start = datetime.now()
                print("Performing Topic Modeling...")
                lda_dtm = perform_tm(doc_ids, clean_docs, num_topics, rm_top, topwords_file, saved_model)
                print(len(lda_dtm))
                lda_dtm['og_doc'] = og_docs  # add original docs to doc-topic df
                # lda_dtm['url'] = doc_urls  # add urls of the posts of the sentences to matrix
                lda_dtm.to_pickle(docterm_file, protocol=4)  # convert doc-topic df in csv file
                print(f'{str(datetime.now())}____Topic modeling {num_topics}, {rm_top} time:____{str(datetime.now() - start)}\n')  # print timing of topic modeling


if __name__ == '__main__':
    level = 'parags'
    dataset_path = os.path.join('data', f'{subreddit}_{level}.pkl')  # replace with path of your dataset in pkl format
    output_folder = os.path.join('output', 'topic_modeling', level)  # replace with path for your tomo results
    if not os.path.exists(output_folder):  # create result folder if it doesn't exist
        os.makedirs(output_folder)
        
    main('endo+endometriosis', dataset_path, output_folder)  # name of the subreddit file
