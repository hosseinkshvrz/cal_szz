import json
import math
import os
import time

import pandas as pd
from pydriller import GitRepository

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, 'data')


class GitHandler:
    def __init__(self, path):
        self.gr = GitRepository(path)

    def get_commit(self, commit_id):
        try:
            commit = self.gr.get_commit(commit_id)
        except ValueError:
            return []
        return [commit]

    def commit_extractor(self):
        columns = ['commit_id', 'buggy', 'fix']
        df = pd.read_csv(os.path.join(data_path, 'rawdata.csv'))
        df = df[columns]
        df.set_index('commit_id', inplace=True)
        fixings = list()
        for commit, fix in df.to_dict()['fix'].items():
            if fix:
                fixings += self.get_commit(commit)
        return fixings

    def get_candidates(self):
        candidates = dict()
        fixings = self.commit_extractor()
        for c in fixings:
            start = time.time()
            c_candids = self.gr.get_commits_last_modified_lines(c)
            c_candids = {k: list(v) for k, v in c_candids.items()}
            candidates[c.hash] = c_candids
            print('commit {} candidates collected in {}.'.format(c.hash[:7], time_since(start)))
        return candidates


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{} min {:.2f} sec'.format(m, s)


def store_candidates():
    git_hr = GitHandler(os.path.join(BASE_DIR, 'nova'))
    start = time.time()
    candidates = git_hr.get_candidates()    # dict of dict of list - 1st key: commit_id, 2nd key: file, list(commit_ids)
    print('\nall candidates collected in {}.'.format(time_since(start)))
    with open(os.path.join(data_path, 'candidates.json'), 'w') as file:
        json.dump(candidates, file)
    print('\ncandidates saved on file.')


if __name__ == '__main__':
    # store_candidates()
    git = GitRepository(os.path.join(BASE_DIR, 'nova'))
    commits = git.get_commits_last_modified_lines(git.get_commit('af911f12fe726ae17601e2381455742882ca71e8'))
    print(commits)
