import os

from gitdb.exc import BadName
from pydriller import GitRepository

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


class GitHandler:
    def __init__(self, path):
        self.gr = GitRepository(path)

    def commit_exists(self, commit_id):
        try:
            self.gr.get_commit(commit_id)
        except BadName:
            return False
        return True


if __name__ == '__main__':
    git = GitHandler(os.path.join(BASE_DIR, 'nova'))
    exists = git.commit_exists('ab07507')
    print(exists)
