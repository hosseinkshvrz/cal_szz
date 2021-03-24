import json
import math
import os
import time

import pandas as pd
from pydriller import GitRepository

# for c in commits:
#     for f in c.files:
#         result = bm25(f.deleted, commits.added)
#         training set = [result[0], random(20)]
#         train()
#         p = predict()
#         assess(p, szz)
