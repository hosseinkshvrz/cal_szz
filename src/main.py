import os

from models import ActiveSZZ, BASE_DIR

szz = ActiveSZZ(os.path.join(BASE_DIR, 'nova'))
szz.baseline()
