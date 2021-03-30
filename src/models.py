import json
import math
import os
import re
import time

import pandas as pd
import numpy as np
from pydriller import GitRepository
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from dataset import get_data, clean_text
from gensim.summarization import bm25

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, 'data')
MAX_INT = 999999999


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{} min {:.2f} sec'.format(m, s)


def infinity():
    i = 0
    while True:
        yield i
        i += 1


def parse_fname(filename):
    try:
        filename = re.sub(r'\{.*\=>.*\}', re.search(r'\{.*\=>.*\}', filename).group(0).split(' => ')[-1][:-1], filename)
    except AttributeError:
        return filename
    return filename


class ActiveSZZ:
    def __init__(self, project_path):
        self.train_df = get_data(mode='train')
        with open(os.path.join(data_path, 'clean_added.json')) as file:
            self.added = json.load(file)
        with open(os.path.join(data_path, 'clean_deleted.json')) as file:
            self.deleted = json.load(file)
        self.git_repo = GitRepository(project_path)
        self.hashes_train = list(pd.concat([self.train_df['HashId'], self.train_df['FixHashId']]).unique())
        self.current_last_modified = list()
        self.clf = LogisticRegression(random_state=0)
        self.bm25 = None
        self.tfidf_model = None
        self.add_corpus = list()
        self.inv_added = list()
        self.evaluations = list()

    def extract_features(self):
        corpus = list()
        for hash in self.hashes_train:
            corpus += self.added.get(hash, {}).values()
            corpus += self.deleted.get(hash, {}).values()
        self.tfidf_model = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
        self.tfidf_model.fit(corpus)

    # for now: search among all, everything! no matter when or what file it is about.
    def index_docs(self):
        for hash in self.hashes_train:
            for file, doc in self.added.get(hash, {}).items():
                self.add_corpus.append(doc)
                self.inv_added.append(str(file + ' @ ' + hash))
        self.bm25 = bm25.BM25(self.add_corpus)
        del self.added

    @staticmethod
    def reciprocal_rank(y, y_true):
        best_rank = MAX_INT
        for h in y_true:
            try:
                rank = y.index(h)
            except ValueError:
                continue
            if rank < best_rank:
                best_rank = rank
                if best_rank == 0:
                    break
        reciprocal = 0 if best_rank == MAX_INT else 1 / (best_rank + 1)
        return reciprocal, best_rank

    @staticmethod
    def avg_reciprocal_rank(y, y_true):
        reciprocals = list()
        for i in range(len(y_true)):
            reciprocal, index = ActiveSZZ.reciprocal_rank(y, y_true)
            reciprocals.append(reciprocal)
            if reciprocal > 0:
                y = y[index+1:]
        avg_reciprocal = np.mean(reciprocals)
        return avg_reciprocal

    @staticmethod
    def precision_k(k, y, y_true):
        precision_k = len(set(y[:k]).intersection(set(y_true))) / k
        return precision_k

    @staticmethod
    def avg_precision_k(y, y_true):
        k = len(y_true)
        avg_precision_k = np.mean([ActiveSZZ.precision_k(i, y, y_true) for i in range(1, k+1)])
        return avg_precision_k

    def evaluate(self, y, y_true):
        reciprocal, _ = self.reciprocal_rank(y, y_true)
        avg_reciprocal = self.avg_reciprocal_rank(y, y_true)
        prc_k = self.precision_k(len(y_true), y, y_true)
        avg_prc_k = self.avg_precision_k(y, y_true)
        return reciprocal, avg_reciprocal, prc_k, avg_prc_k

    def initialize(self, hash, file):
        query = self.deleted[hash][file]
        scores = self.bm25.get_scores(query)
        best_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[0]
        return best_idx

    def fit(self, data):
        X = self.tfidf_model.transform([self.add_corpus[j] for j in data.keys()])
        y = np.fromiter(data.values(), dtype=float)
        X, y = shuffle(X, y, random_state=0)
        self.clf.fit(X, y)

    def review(self, labeled_indices, batch_size):
        relevance_prob = self.clf.predict_proba(self.tfidf_model.transform(self.add_corpus))[:, 1]
        k = min(batch_size + len(labeled_indices), len(relevance_prob))
        top_k_indices = np.argpartition(relevance_prob, -k)[-k:]
        sorted_top_k = top_k_indices[np.argsort(relevance_prob[top_k_indices])]
        top_b_indices = list(set(sorted_top_k) - set(labeled_indices))[-batch_size:]
        retrieved = [self.inv_added[j].split(' @ ')[1] for j in top_b_indices]
        retrieved_labels = [int(h in self.current_last_modified) for h in retrieved]
        return top_b_indices, retrieved_labels

    def get_ranked_output(self, labeled_indices):
        relevance_prob = self.clf.predict_proba(
            self.tfidf_model.transform(np.array(self.add_corpus, dtype='object')[labeled_indices]))[:, 1]
        sorted_ranks = np.argsort(-relevance_prob)
        retrieved = [self.inv_added[labeled_indices[i]].split(' @ ')[1] for i in sorted_ranks]
        return retrieved

    def train(self):
        reciprocals, avg_reciprocals, precision_ks, avg_precision_ks = list(), list(), list(), list()
        for i, row in self.train_df.iterrows():
            start = time.time()
            fix_hash = row['FixHashId']
            original_name = row['File']
            file = parse_fname(original_name)
            commit = self.git_repo.get_commit(fix_hash)
            try:
                self.current_last_modified = list(self.git_repo.get_commits_last_modified_lines(commit)[file])
            except KeyError:
                print('\n NO CORRESPONDING SZZ CANDIDATE EXISTS.\n')
                continue
            best_idx = self.initialize(fix_hash, file)

            data = dict()
            data[best_idx] = 1
            labeled_indices = [best_idx]
            batch_size = 1
            for cntr in infinity():
                indices = np.delete(np.arange(len(self.add_corpus)), labeled_indices)
                np.random.shuffle(indices)
                for idx in indices[:100]:
                    data[idx] = 0
                self.fit(data)

                top_b_indices, retrieved_labels = self.review(labeled_indices, batch_size)
                for h_i in range(len(retrieved_labels)):
                    data[top_b_indices[h_i]] = retrieved_labels[h_i]
                labeled_indices += top_b_indices
                data = {doc_idx: data[doc_idx] for doc_idx in labeled_indices}
                batch_size += math.ceil(batch_size / 10)

                # check if all szz commits are in labeled (collected) commits
                # or if the difference is equal to difference of entire dataset with szz commits
                # the 2nd condition is okay because we're sure the true commit is in the dataset.
                collected = [self.inv_added[j].split(' @ ')[1] for j in labeled_indices]
                print('{:>6}'.format(len(set(self.current_last_modified) - set(collected))), end='', flush=True)
                finished = set(self.current_last_modified).issubset(set(collected)) or \
                           (set(self.current_last_modified) - set([i.split(' @ ')[1] for i in self.inv_added]) ==
                            set(self.current_last_modified) - set(collected))

                if finished:
                    self.fit(data)
                    retrieved = self.get_ranked_output(labeled_indices)
                    true_hash = list(self.train_df[(self.train_df['FixHashId'] == fix_hash) &
                                                   (self.train_df['File'] == original_name)]['HashId'])
                    reciprocal, avg_reciprocal, prc_k, avg_prc_k = self.evaluate(retrieved, true_hash)

                    print('\ncommit {} file {} took {}.'.format(fix_hash[:7], file, time_since(start)))
                    print('reciprocal={:.2f}\t\tavg reciprocal={:.2f}\t\tp@k={:.2f}\t\tavg_p@k={:.2f}\n\n'
                          .format(reciprocal, avg_reciprocal, prc_k, avg_prc_k))
                    reciprocals.append(reciprocal)
                    avg_reciprocals.append(avg_reciprocal)
                    precision_ks.append(prc_k)
                    avg_precision_ks.append(avg_prc_k)
                    break
                if cntr % 20 == 19:
                    print()
            if (i - self.train_df.index[0]) % 100 == 99:
                print('\n*** 100 sample stats')
                print('MRR={:.2f}\t\tMARR={:.2f}\t\tmean P@k={:.2f}\t\tMAP@k={:.2f}\n'
                      .format(np.mean(reciprocals), np.mean(avg_reciprocals),
                              np.mean(precision_ks), np.mean(avg_precision_ks)))
        print('\n*** finished.')
        print('MRR={:.2f}\t\tMARR={:.2f}\t\tmean P@k={:.2f}\t\tMAP@k={:.2f}\n'
              .format(np.mean(reciprocals), np.mean(avg_reciprocals), np.mean(precision_ks), np.mean(avg_precision_ks)))

    def test(self):
        pass
        # self.train_df = get_data(mode='test')
        # retrieved = self.get_ranked_output(labeled_indices)
        # true_hash = list(self.train_df[(self.train_df['FixHashId'] == fix_hash) & (self.train_df['File'] == file)][
        #                      'HashId'])
        # reciprocal, prc_k, avg_prc_k = self.evaluate(retrieved, true_hash)
        #
        # print('\ncommit {} file {} took {}.'.format(fix_hash[:7], file, time_since(start)))
        # print('reciprocal={:.2f}\t\tp@k={:.2f}\t\tavg_p@k={:.2f}.\n'.format(reciprocal, prc_k, avg_prc_k))
        # reciprocals.append(reciprocal)
        # precision_ks.append(prc_k)
        # avg_precision_ks.append(avg_prc_k)

    def baseline(self):
        reciprocals, avg_reciprocals, precision_ks, avg_precision_ks = list(), list(), list(), list()
        for i, row in self.train_df.iterrows():
            start = time.time()
            fix_hash = row['FixHashId']
            original_name = row['File']
            file = parse_fname(original_name)
            commit = self.git_repo.get_commit(fix_hash)
            try:
                last_modified = list(self.git_repo.get_commits_last_modified_lines(commit)[file])
            except KeyError:
                print('\n NO CORRESPONDING SZZ CANDIDATE EXISTS.\n')
                continue
            query = self.deleted[fix_hash][file]
            corpus = list()
            inverse = list()
            for h in last_modified:
                c = self.git_repo.get_commit(h)
                for m in c.modifications:
                    if m.new_path == file:
                        corpus.append(list())
                        inverse.append(h)
                        for added in m.diff_parsed['added']:
                            if added[1] is '':
                                continue
                            cleaned = clean_text(added[1])
                            if cleaned:  # skip zero size docs
                                corpus[-1] += cleaned

            try:
                bm25_object = bm25.BM25(corpus)
            except ZeroDivisionError:
                print('\n*** THE COMMIT {} CORPUS IS EMPTY.\n'.format(fix_hash))
                continue
            scores = bm25_object.get_scores(query)
            sorted_indices = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
            sorted_hashes = [inverse[j] for j in sorted_indices]
            true_hash = list(self.train_df[(self.train_df['FixHashId'] == fix_hash) &
                                           (self.train_df['File'] == original_name)]['HashId'])
            reciprocal, avg_reciprocal, prc_k, avg_prc_k = self.evaluate(sorted_hashes, true_hash)

            print('\ncommit {} file {} took {}.'.format(fix_hash[:7], file, time_since(start)))
            print('reciprocal={:.2f}\t\tavg reciprocal={:.2f}\t\tp@k={:.2f}\t\tavg_p@k={:.2f}\n'
                  .format(reciprocal, avg_reciprocal, prc_k, avg_prc_k))

            reciprocals.append(reciprocal)
            avg_reciprocals.append(avg_reciprocal)
            precision_ks.append(prc_k)
            avg_precision_ks.append(avg_prc_k)

            if (i - self.train_df.index[0]) % 100 == 99:
                print('\n*** 100 sample stats')
                print('MRR={:.2f}\t\tMARR={:.2f}\t\tmean P@k={:.2f}\t\tMAP@k={:.2f}\n'
                      .format(np.mean(reciprocals), np.mean(avg_reciprocals), np.mean(precision_ks),
                              np.mean(avg_precision_ks)))

        print('\n*** finished.')
        print('MRR={:.2f}\t\tMARR={:.2f}\t\tmean P@k={:.2f}\t\tMAP@k={:.2f}\n'
              .format(np.mean(reciprocals), np.mean(avg_reciprocals),
                      np.mean(precision_ks), np.mean(avg_precision_ks)))


if __name__ == '__main__':
    szz = ActiveSZZ(os.path.join(BASE_DIR, 'nova'))
    szz.extract_features()
    szz.index_docs()
    szz.train()

