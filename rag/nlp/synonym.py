import json
import logging
import os

from api.utils.file_utils import get_project_base_directory

"""
[ 近义词 ]
"""


class Dealer:
    def __init__(self):
        self.dictionary = None
        path = os.path.join(get_project_base_directory(), "rag/res", "synonym.json")
        try:
            self.dictionary = json.load(open(path, 'r', encoding='utf-8'))
        except Exception as e:
            logging.warning("[WARNING] Load synonym.json FAIL!")
            self.dictionary = {}

    def lookup(self, tk):
        res = self.dictionary.get(tk.lower().strip(), [])
        if isinstance(res, str):
            res = [res]
        return res

# if __name__ == '__main__':
#     dl = Dealer()
#     print(dl.dictionary)
