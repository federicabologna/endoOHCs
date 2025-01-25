import os
import glob
import pandas as pd


def combine_one_subreddit(_subreddit, _subreddit_csv_path):  # creating csv with all of a subreddit's posts and comments

    df_d = {'author': [], 'id': [], 'type': [], 'text': [],   # create a dictionary
            'url': [], 'link_id': [], 'parent_id': [], 'flair': [],
            'subreddit': [], 'created_utc': []}

    for target_type in ['posts', 'comments']:
        files_directory_path = os.path.join('data', _subreddit, target_type)
        all_target_type_files = glob.glob(os.path.join(files_directory_path, "*.csv"))

        for f in all_target_type_files:
            # df = pd.read_pickle(f)
            df = pd.read_csv(f, index_col=0, header=0)
            if target_type == 'posts':
                if 'link_flair_text' in df.columns:
                    df.fillna(value={'link_flair_text': ''}, inplace=True)
                for index, row in df.iterrows():
                    df_d['author'].append(row['author'])
                    df_d['id'].append(f"{row['subreddit']}_{row['id']}_post")  # id of the post, 'Endo_xyz123_post'
                    df_d['type'].append('post')
                    df_d['text'].append(row['selftext'])  # textual content of the post
                    df_d['url'].append(row['url'])  # url of the post
                    df_d['link_id'].append('N/A')
                    df_d['parent_id'].append('N/A')
                    if 'link_flair_text' in df.columns:
                        df_d['flair'].append(row['link_flair_text'])
                    else:
                        df_d['flair'].append('N/A')
                    df_d['subreddit'].append(row['subreddit'])
                    df_d['created_utc'].append(row['created_utc'])  # utc time stamp of the post


            elif target_type == 'comments':
                for index, row in df.iterrows():
                    df_d['author'].append(row['author'])
                    df_d['id'].append(f"{row['subreddit']}_{row['id']}_comment")
                    df_d['type'].append('comment')
                    df_d['text'].append(row['body'])  # textual content of the comment
                    df_d['url'].append(f"http://www.reddit.com/r/Endo/comments/{row['link_id'].split('_')[1]}/")  # url of the post
                    df_d['link_id'].append(row['link_id'])
                    df_d['parent_id'].append(row['parent_id'])
                    df_d['flair'].append('N/A')
                    df_d['subreddit'].append(row['subreddit'])
                    df_d['created_utc'].append(row['created_utc'])  # utc time stamp of the post

    subreddit_df = pd.DataFrame.from_dict(df_d)
    print(_subreddit_csv_path, len(subreddit_df))
    subreddit_df.sort_values('created_utc', inplace=True, ignore_index=True)
    subreddit_df['time'] = pd.to_datetime(subreddit_df['created_utc'], unit='s').apply(lambda x: x.to_datetime64())
    subreddit_df.to_pickle(_subreddit_csv_path, protocol=4)


def combine_multiple_subreddits(_subreddits, _combined_csv_path): # creating csv with all subreddits' posts and comments

    all_subreddits_files = [os.path.join('data', _subreddit, f'{_subreddit}.pkl') for _subreddit in _subreddits]
    combined_df = pd.concat((pd.read_pickle(f) for f in all_subreddits_files), axis=0,
                            ignore_index=True)
    print(_combined_csv_path, len(combined_df))
    combined_df.sort_values('created_utc', inplace=True, ignore_index=True)
    combined_df.to_pickle(_combined_csv_path, protocol=4)


def split_paragraphs(_combined_csv_path, _parags_csv_path):

    paragraphs = []
    df = pd.read_pickle(_combined_csv_path)
    for index, row in df.iterrows():  # iterate over posts
        par_n = 0
        for paragraph in row['text'].split('\n\n'):  # split post in paragraphs
            if len(paragraph.split(' ')) > 5:  # keep paragraphs that are longer than 5 words
                par_id = f"{row['id']}_{par_n}"  # create new id from post id
                par_d = {'author': row['author'],
                         'id': par_id,  # dict with paragraph' info
                         'text': paragraph,
                         'type': row['type'],
                         'url': row['url'],
                         'created_utc': row['created_utc'],
                         'link_id': row['link_id'],
                         'parent_id': row['parent_id'],
                         'flair': row['flair'],
                         'subreddit': row['subreddit'],
                         }
                paragraphs.append(par_d)
                par_n += 1  # paragraphs counter

    par_df = pd.DataFrame(paragraphs)  # transform dict into dataframe
    print(_parags_csv_path, len(par_df))
    par_df.to_pickle(_parags_csv_path, protocol=4)

    return par_df


def main():

    subreddits = ['endo', 'endometriosis']
    combine_one = True
    combine_multiple = True
    paragraphs = True
    combined_names = '+'.join([_subreddit for _subreddit in subreddits])

    if combine_one:
        for subreddit in subreddits:
            subreddit_csv_path = os.path.join('data', subreddit, f'{subreddit}.pkl')
            if not os.path.exists(subreddit_csv_path):
                combine_one_subreddit(subreddit, subreddit_csv_path)

    if combine_multiple:
        combined_csv_path = os.path.join('data', f'{combined_names}.pkl')
        if not os.path.exists(combined_csv_path):
            flag = True
            for subreddit in subreddits:
                subreddit_csv_path = os.path.join('data', subreddit, f'{subreddit}.pkl')
                if not os.path.exists(subreddit_csv_path):
                    flag = False
            if flag:
                combine_multiple_subreddits(subreddits, combined_csv_path)

    if paragraphs:
        parags_csv_path = os.path.join('data', f'{combined_names}_parags.pkl')
        if not os.path.exists(parags_csv_path):
            combined_csv_path = os.path.join('data', f'{combined_names}.pkl')
            if os.path.exists(combined_csv_path):
                split_paragraphs(combined_csv_path, parags_csv_path)


if __name__ == '__main__':
    main()
