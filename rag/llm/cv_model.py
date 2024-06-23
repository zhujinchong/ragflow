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
import base64
from abc import ABC
from io import BytesIO

from openai import OpenAI
import ollama


class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def describe(self, image, max_tokens=300):
        raise NotImplementedError("Please implement encode method!")

    def image2base64(self, image):
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        if isinstance(image, BytesIO):
            return base64.b64encode(image.getvalue()).decode("utf-8")
        buffered = BytesIO()
        try:
            image.save(buffered, format="JPEG")
        except Exception as e:
            image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def prompt(self, b64):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "请用中文详细描述一下图中的内容，比如时间，地点，人物，事情，人物心情等，如果有数据请提取出数据。" if self.lang.lower() == "chinese" else
                        "Please describe the content of this picture, like where, when, who, what happen. If it has number data, please extract them out.",
                    },
                ],
            }
        ]


class DefaultCV(Base):
    def __init__(self, key="EMPTY", model_name="qnguyen3/nanollava", lang="Chinese",
                 base_url="http://localhost:11434/v1"):
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name
        self.lang = lang

    def describe(self, image_data, max_tokens=300):
        image_base64 = base64.b64encode(image_data)
        image_base64_str = image_base64.decode('utf-8')
        res = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Please provide a brief description of the image content.',
                    'images': [f'{image_base64_str}']
                }
            ],
            options={
                "seed": 101,
                "temperature": 0
            }
        )
        return res['message']['content']


# if __name__ == '__main__':
#     img_path = "./454_4065_a3522f131d4c06d.jpg"
#     cv = DefaultCV()
#     with open(img_path, 'rb') as image_file:
#         image_data = image_file.read()
#     print(cv.describe(image_data))
