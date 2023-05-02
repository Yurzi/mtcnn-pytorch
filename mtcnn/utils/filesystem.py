import os
import shutil


def check_and_reset(path: str, is_file: bool = False):
    """
    check path dir is exist or not, if not, create or reset it
    """
    # if path is not exist
    if not os.path.exists(path):
        # create dir
        if is_file:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            else:
                # do nothing, just for syntax
                pass
        else:
            os.makedirs(path)
        return

    # if path is exist
    if os.path.isdir(path):
        # remove dir
        shutil.rmtree(path)
        # create dir
        os.makedirs(path)
    else:
        # remove file
        os.remove(path)
        # create file dir
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
