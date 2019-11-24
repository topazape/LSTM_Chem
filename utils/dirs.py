import os
import sys


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        print(f'Creating directories error: {err}')
        sys.exit()
