import json
import os

import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pydriller import RepositoryMining

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, 'data')


def not_meaningful(text):
    if text is '':
        return True
    # elif re.match(r'^\s*#', text):
    #     return True, False
    # elif re.match(r'^\s*""".*"""', text):
    #     return True, False
    # elif re.match(r'^\s*"""$', text):       # """
    #     if comment_started:
    #         return True, False
    # elif re.match(r'^\s*""".+$', text):  # """ blah blah blah
    #     pass
    # elif re.match(r'^\s*.+"""$', text):  # blah blah blah"""
    #     pass
    else:
        return False


def data_extractor(project_name):
    df = pd.read_csv(os.path.join(data_path, 'openstack_links.csv'))
    df = df[df['Project'] == project_name]
    all_ids = list(pd.concat([df['HashId'], df['FixHashId']]).unique())
    iterator = RepositoryMining(os.path.join(BASE_DIR, 'nova'),
                                only_commits=all_ids,
                                only_modifications_with_file_types=['.py']).traverse_commits()

    allowed_types = ['py']
    with open(os.path.join(data_path, 'added.txt'), 'a') as a_file:
        with open(os.path.join(data_path, 'deleted.txt'), 'a') as d_file:
            for commit in iterator:
                a_file.write('***&%$#@ commit {} @#$%&***\n'.format(commit.hash))
                d_file.write('***&%$#@ commit {} @#$%&***\n'.format(commit.hash))
                for m in commit.modifications:
                    if len(m.filename.split('.')) == 1 or m.filename.split('.')[1] not in allowed_types:
                        continue
                    a_file.write('***&%$#@ file {} @#$%&***\n'.format(m.new_path))
                    d_file.write('***&%$#@ file {} @#$%&***\n'.format(m.new_path))

                    for added in m.diff_parsed['added']:
                        if not_meaningful(added[1]):
                            with open(os.path.join(data_path, 'skipped.txt'), 'a') as s_file:
                                s_file.write('{}\n'.format(added[1]))
                            continue
                        a_file.write('{}\n'.format(added[1]))

                    for deleted in m.diff_parsed['deleted']:
                        if not_meaningful(deleted[1]):
                            with open(os.path.join(data_path, 'skipped.txt'), 'a') as s_file:
                                s_file.write('{}\n'.format(deleted[1]))
                            continue
                        d_file.write('{}\n'.format(deleted[1]))


def clean_text(text):
    # text = text.lower()
    text = re.sub(r'[_\'"\-;%()|+&=*.,!?:#$@\[\]/~`<>{}^]', ' ', text)
    text = text.split()
    # stops = set(nltk.corpus.stopwords.words("english"))
    # text = [w for w in text if not w in stops]
    text = " ".join(text)
    text = nltk.WordPunctTokenizer().tokenize(text)
    # lemm = nltk.stem.WordNetLemmatizer()
    # text = list(map(lambda word: list(map(lemm.lemmatize, word)), text))

    return text


def feature_extractor(corpus):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    tfidf_vectorizer.fit(corpus)
    return tfidf_vectorizer


def get_data(mode='train'):
    ratio = 5
    df = pd.read_csv(os.path.join(data_path, 'openstack_links.csv'))
    df = df[df['Project'] == 'nova']
    fix_ids = df['FixHashId'].unique()
    split_idx = len(fix_ids) - (len(fix_ids) // ratio)
    if mode is 'train':
        return df[df['FixHashId'].isin(fix_ids[:split_idx])][['HashId', 'FixHashId', 'File']]
    elif mode is 'test':
        return df[df['FixHashId'].isin(fix_ids[split_idx:])][['HashId', 'FixHashId', 'File']]


def store_clean_data():
    with open(os.path.join(data_path, 'added.txt')) as file:
        content = file.readlines()

    added = dict()
    commit = ''
    file = ''
    for line in content:
        if line.strip().startswith('***&%$#@ commit'):
            commit = line.split()[2]
        elif line.strip().startswith('***&%$#@ file'):
            file = line.split()[2]
        else:
            if commit not in added:
                added[commit] = dict()
            if file not in added[commit]:
                added[commit][file] = []
            cleaned = clean_text(line)
            if cleaned:     # skip zero size docs
                added[commit][file] += cleaned

    with open(os.path.join(data_path, 'clean_added.json'), 'w') as file:
        json.dump(added, file)

    with open(os.path.join(data_path, 'deleted.txt')) as file:
        content = file.readlines()

    deleted = dict()
    commit = ''
    file = ''
    for line in content:
        if line.strip().startswith('***&%$#@ commit'):
            commit = line.split()[2]
        elif line.strip().startswith('***&%$#@ file'):
            file = line.split()[2]
        else:
            if commit not in deleted:
                deleted[commit] = dict()
            if file not in deleted[commit]:
                deleted[commit][file] = []
            cleaned = clean_text(line)
            if cleaned:     # skip zero size docs
                deleted[commit][file] += cleaned

    with open(os.path.join(data_path, 'clean_deleted.json'), 'w') as file:
        json.dump(deleted, file)

    # for i, row in df.iterrows():
    #     if i > split_idx:
    #         break


if __name__ == '__main__':
    # store_clean_data()

    # df = pd.read_csv(os.path.join(data_path, 'openstack_links.csv'))
    # df = df[df['Project'] == 'nova']
    # all_ids = list(pd.concat([df['HashId'], df['FixHashId']]).unique())
    # iterator = RepositoryMining(os.path.join(BASE_DIR, 'nova'),
    #                             only_commits=all_ids,
    #                             only_modifications_with_file_types=['.py']).traverse_commits()
    # for c in iterator:
    #     if c.hash is '3ad42eea208a85619efe0096be8388526b5ffe3b':
    #         pass

    pass
