# import numpy as np
# from sklearn.linear_model import LogisticRegression
#
#
# def get_top_k_predictions(model, X_test, k):
#     # get probabilities instead of predicted labels, since we want to collect top 3
#     probs = model.predict_proba(X_test)
#     # GET TOP K PREDICTIONS BY PROB - note these are just index
#     best_n = np.argsort(probs, axis=1)[:, -k:]
#     # GET CATEGORY OF PREDICTIONS
#     preds = [[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]
#     # REVERSE CATEGORIES - DESCENDING ORDER OF IMPORTANCE
#     preds = [item[::-1] for item in preds]
#
#     return preds
#
#
# scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
# model = scikit_log_reg.fit(X_train,Y_train)
#
# # GET TOP K PREDICTIONS
# preds = get_top_k_predictions(model, X_test, top_k)
#
# # GET PREDICTED VALUES AND GROUND TRUTH INTO A LIST OF LISTS
# eval_items = collect_preds(Y_test, preds)
#
# # GET EVALUATION NUMBERS ON TEST SET -- HOW DID WE DO?
# accuracy = compute_accuracy(eval_items)
# mrr_at_k = compute_mrr_at_k(eval_items)

import json
import math
import os
import pandas as pd
import numpy as np
from pydriller import GitRepository
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from dataset import get_data, feature_extractor
from gensim.summarization import bm25

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, 'data')

train_df = get_data(mode='train')
with open(os.path.join(data_path, 'clean_added.json')) as file:
    added = json.load(file)
with open(os.path.join(data_path, 'clean_deleted.json')) as file:
    deleted = json.load(file)

hashes_train = list(pd.concat([train_df['HashId'], train_df['FixHashId']]).unique())
corpus = list()
for hash in hashes_train:
    corpus += added.get(hash, {}).values()
    corpus += deleted.get(hash, {}).values()
tfidf_model = feature_extractor(corpus)

# for now: search among all, everything! no matter when or what file it is about.
add_corpus = list()
inv_added = list()
for hash in hashes_train:
    for file, doc in added.get(hash, {}).items():
        add_corpus.append(doc)
        inv_added.append(str(file + ' @ ' + hash))

del added
del corpus


def check_szz(hash, file, retrieved):
    git_repo = GitRepository(os.path.join(BASE_DIR, 'nova'))
    commit = git_repo.get_commit(hash)
    last_modified = list(git_repo.get_commits_last_modified_lines(commit)[file])
    labels = [int(h in last_modified) for h in retrieved]
    return labels


for i, row in train_df.iterrows():
    fix_hash = row['FixHashId']
    file = row['File']
    query = deleted[fix_hash][file]
    bm25_obj = bm25.BM25(add_corpus)
    scores = bm25_obj.get_scores(query)
    best_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[0]

    data = dict()
    data[best_idx] = 1
    labeled_indices = [best_idx]
    batch_size = 1
    while True:
        indices = np.delete(np.arange(len(add_corpus)), labeled_indices)
        np.random.shuffle(indices)
        for idx in indices[:20]:
            data[idx] = 0
        X = tfidf_model.transform([add_corpus[j] for j in data.keys()])
        y = np.fromiter(data.values(), dtype=float)
        X, y = shuffle(X, y, random_state=0)

        clf = LogisticRegression(random_state=0).fit(X, y)
        relevance_prob = clf.predict_proba(tfidf_model.transform(add_corpus))[:, 1]
        k = batch_size + len(labeled_indices)
        top_k_indices = np.argpartition(relevance_prob, -k)[-k:]
        sorted_top_k = top_k_indices[np.argsort(relevance_prob[top_k_indices])]
        top_b_indices = list(set(top_k_indices) - set(labeled_indices))[-batch_size:]
        retrieved = [inv_added[j].split(' @ ')[1] for j in top_b_indices]

        retrieved_labels = check_szz(fix_hash, file, retrieved)
        for h_i in range(batch_size):
            data[top_b_indices[h_i]] = retrieved_labels[h_i]
        labeled_indices += top_b_indices
        data = {doc_idx: data[doc_idx] for doc_idx in labeled_indices}
        batch_size += math.ceil(batch_size / 10)
        print()


# if __name__ == '__main__':
#     pass
