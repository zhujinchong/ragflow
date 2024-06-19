#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from abc import ABC

import numpy as np
import requests

from rag.utils import num_tokens_from_string, truncate


class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def encode(self, texts: list, batch_size=32):
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str):
        raise NotImplementedError("Please implement encode method!")


class DefaultEmbedding(Base):

    def __init__(self):
        self.url = "http://localhost:11434/api/embeddings"
        self.model_name = "shaw/dmeta-embedding-zh-small"

    def encode(self, texts: list, batch_size=32):
        texts = [truncate(t, 1024) for t in texts]
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        res = []
        for i in range(0, len(texts)):
            data = {
                "model": self.model_name,
                "prompt": texts[i]
            }
            response = requests.post(self.url, json=data).json()
            res.append(response["embedding"])
        return np.array(res), token_count

    def encode_queries(self, text: str):
        token_count = num_tokens_from_string(text)
        data = {
            "model": self.model_name,
            "prompt": text
        }
        response = requests.post(self.url, json=data).json()
        return response["embedding"], token_count


# if __name__ == '__main__':
#     llm = DefaultEmbedding()
#     ret = llm.encode(["hello", "你好"])
#     print(ret)
#     print(llm.encode_queries("hello"))
#
