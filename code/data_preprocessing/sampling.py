import os
import json
import pandas as pd
import numpy as np
import Levenshtein


def find_annotated(path, id_label, df):

    annotated_idx = []
    ids = pd.read_csv(path)['id'].tolist()
    for id in ids:
        annotated_idx.append(df.loc[df[id_label] == id].index.values[0])

    return annotated_idx


def find_duplicates(df):  # function to find duplicated posts in the data

    prev_post = ''
    map_dict = {}  # dict of authors' posts
    dup = []  # list of duplicates' indexes for removal from dataframe
    for index, row in df.iterrows():  # iterate over posts
        author = row['author']
        post = row['selftext']

        # if author info is available we compare each post with previous ones by the same author
        # we compare/calculate the similarity between the posts using the Levenshtein distance
        if author != '[deleted]':
            if author in map_dict.keys():
                flag = 0
                idx = 0
                while idx < len(map_dict[author]) and flag == 0:
                    lev = Levenshtein.ratio(post, map_dict[author][idx])
                    if lev > 0.99:
                        dup.append(index)
                        flag = 1
                    idx += 1
                if flag == 0:
                    map_dict[author].append(post)
            else:
                map_dict[author] = [post]

        # if author info is not available we compare each post with the preceding one chronologically
        else:
            lev = Levenshtein.ratio(row['selftext'], prev_post)
            if lev > 0.99:
                dup.append(index)

        prev_post = post

    return dup


def split_paragraphs(df):

    paragraphs = []
    for index, row in df.iterrows():  # iterate over posts
        par_n = 0
        for paragraph in row['selftext'].split('\n\n'):  # split post in paragraphs
            if len(paragraph.split(' ')) > 5:  # keep paragraphs that are longer than 5 words
                par_id = f"{row['concat_id']}_{par_n}"  # create new id from post id
                par_d = {'id': par_id,  # dict with paragraph' info
                         'text': paragraph,
                         'date': row['time'].split(' ')[0],
                         'yy': row['yy'],
                         'url': row['url']}
                paragraphs.append(par_d)
                par_n += 1  # paragraphs counter

    par_df = pd.DataFrame(paragraphs)  # transform dict into dataframe

    return par_df


def sampling(df, size):

    init = True
    ratio = size / len(df)  # ratio of docs to sample based on size of the sample and size of the corpus
    for year in df.yy.unique():  # for each year in the corpus
        sam = df.loc[df['yy'] == year].sample(frac=ratio)  # we randomly sample that ratio of docs from all the docs in that year
        if init:
            samp = sam
            init = False
        else:
            samp = samp.append(sam)  # we aggregate each year's sample into the overall sample

    if len(samp) < size:  # if length of the overall sample < the desired size
        sam = df.sample(n=size - len(samp))  # we sample a few more docs at random
        samp = samp.append(sam)
    elif len(samp) > size:  # if length of the overall sample > the desired size
        samp = samp[:size]  # we discard the exceeding docs

    samp['yy'] = samp['yy'].astype(np.int64)

    return samp


def main(subreddit, level, sample_size, label):

    annotated_path = os.path.join('labeling', 'annotated-data', f'{label}.csv')

    reddit_df = pd.read_csv(os.path.join('data', f'{subreddit}.csv'), index_col=0)  # og data
    print(f'Number of posts and comments: {len(reddit_df)}')

    dupes = find_duplicates(reddit_df)  # find duplicates
    reddit_df.drop(dupes, inplace=True)  # removing duplicates
    print(f'Number of duplicates: {len(dupes)}, Number of posts after removing duplicates: {len(reddit_df)}')

    if level == 'posts':  # if sampling at the post level, we use the same dataset
        df = reddit_df
        annotated = find_annotated(annotated_path, 'concat_id', df)

    elif level == 'parags':  # if sampling at the parags level
        par_reddit_df = split_paragraphs(reddit_df)  # we split the og dataset into a parags dataset
        print(f'Number of paragraphs: {len(par_reddit_df)}')
        df = par_reddit_df
        df.to_csv(os.path.join('data', 'endo+endometriosis_parags.csv'))
        annotated = find_annotated(annotated_path, 'id', df)

    df.drop(annotated, inplace=True)  # removing already annotated docs
    print(f'Number of annotated: {len(annotated)}, Number of posts after removing annotated: {len(df)}')

    sample = sampling(df, sample_size)  # sampling
    for index, row in sample.iterrows():  # save as json list
        post_d = {'id': row['id'], 'text': row['text'], 'url': row['url']}
        sample_json = os.path.join('data', 'samples', f'sample_{label}_{level}_{sample_size}.jsonl')
        with open(sample_json, 'a', encoding="utf-8") as jsonfile:
            json.dump(post_d, jsonfile)
            jsonfile.write('\n')
    print(f'Check sample size: {len(sample)}')


if __name__ == '__main__':
    # SETTINGS
    _subreddit = 'endo+endometriosis'
    _level = 'parags'  #posts, sentences
    _sample_size = 5000
    _label = 'negligence'
    main(_subreddit, _level, _sample_size, _label)
