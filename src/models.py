import json
import math
import os
import time

import pandas as pd
import numpy as np
from pydriller import GitRepository
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from dataset import get_data
from gensim.summarization import bm25

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, 'data')


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


class ActiveSZZ:
    def __init__(self, project_path):
        self.train_df = get_data(mode='train')
        with open(os.path.join(data_path, 'clean_added.json')) as file:
            self.added = json.load(file)
        with open(os.path.join(data_path, 'clean_deleted.json')) as file:
            self.deleted = json.load(file)
        self.git_repo = GitRepository(project_path)
        self.hashes_train = list(pd.concat([self.train_df['HashId'], self.train_df['FixHashId']]).unique())
        self.current_last_modified = []
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
        best_rank = len(y_true)
        for h in y_true:
            try:
                rank = y.index(h)
            except ValueError:
                continue
            if rank < best_rank:
                best_rank = rank
        reciprocal = 0 if best_rank == len(y_true) else 1 / (best_rank + 1)
        return reciprocal

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
        reciprocal = self.reciprocal_rank(y, y_true)
        prc_k = self.precision_k(len(y_true), y, y_true)
        avg_prc_k = self.avg_precision_k(y, y_true)
        return reciprocal, prc_k, avg_prc_k

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
        reciprocals, precision_ks, avg_precision_ks = [], [], []
        for i, row in self.train_df.iterrows():
            start = time.time()
            fix_hash = row['FixHashId']
            file = row['File']
            commit = self.git_repo.get_commit(fix_hash)
            self.current_last_modified = list(self.git_repo.get_commits_last_modified_lines(commit)[file])
            best_idx = self.initialize(fix_hash, file)

            data = dict()
            data[best_idx] = 1
            labeled_indices = [best_idx]
            batch_size = 1
            for cntr in infinity():
                indices = np.delete(np.arange(len(self.add_corpus)), labeled_indices)
                np.random.shuffle(indices)
                for idx in indices[:20]:
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
                    retrieved = self.get_ranked_output(labeled_indices)
                    true_hash = list(self.train_df[(self.train_df['FixHashId'] == fix_hash) & (self.train_df['File'] == file)][
                        'HashId'])
                    reciprocal, prc_k, avg_prc_k = self.evaluate(retrieved, true_hash)

                    print('\ncommit {} file {} took {}.'.format(fix_hash[:7], file, time_since(start)))
                    print('reciprocal={:.2f}\t\tp@k={:.2f}\t\tavg_p@k={:.2f}.\n'.format(reciprocal, prc_k, avg_prc_k))
                    reciprocals.append(reciprocal)
                    precision_ks.append(prc_k)
                    avg_precision_ks.append(avg_prc_k)
                    break
                if cntr % 20 == 19:
                    print()
        print('\n*** finished.')
        print('MRR={:.2f}\t\tmean P@k={:.2f}\t\tMAP@k={:.2f}.\n'
              .format(np.mean(reciprocals), np.mean(precision_ks), np.mean(avg_precision_ks)))

    def test(self):
        pass

    def baseline(self):
        pass


if __name__ == '__main__':
    szz = ActiveSZZ(os.path.join(BASE_DIR, 'nova'))
    szz.extract_features()
    szz.index_docs()
    szz.train()

