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

import openai
from openai import OpenAI
from rag.settings import Ollama


class Base(ABC):
    def __init__(self, key, model_name, base_url):
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                **gen_conf)
            ans = response.choices[0].message.content.strip()
            # if response.choices[0].finish_reason == "length":
            #     ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
            #         [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except openai.APIError as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        ans = ""
        total_tokens = 0
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                stream=True,
                **gen_conf)
            for resp in response:
                if not resp.choices or not resp.choices[0].delta.content: continue
                ans += resp.choices[0].delta.content
                total_tokens += 1
                # if resp.choices[0].finish_reason == "length":
                #     ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                #         [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                yield ans

        except openai.APIError as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens


class DefaultChat(Base):
    def __init__(self):
        super().__init__(key=Ollama.get("chat_key", ""),
                         model_name=Ollama.get("chat_model_name", ""),
                         base_url=Ollama.get("chat_base_url", ""))


# if __name__ == '__main__':
#     llm = DefaultChat()
#     history = [
#         {
#             "role": "user",
#             "content": "你好",
#         }
#     ]
#     for x in llm.chat_streamly(system="", history=history, gen_conf={}):
#         print(x)