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
import io

import numpy as np
from PIL import Image

from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from deepdoc.vision import OCR
from rag.nlp import tokenize

ocr = OCR()


def chunk(filename, binary, tenant_id, lang, callback=None, **kwargs):
    try:
        cv_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, lang=lang)
        # from rag.llm.cv_model import DefaultCV
        # cv_mdl = DefaultCV()
    except Exception as e:
        callback(prog=-1, msg=str(e))
        return []
    img = Image.open(io.BytesIO(binary)).convert('RGB')
    doc = {
        "docnm_kwd": filename,
        "image": img
    }
    bxs = ocr(np.array(img))
    txt = "\n".join([t[0] for _, t in bxs if t[0]])
    eng = lang.lower() == "english"
    callback(0.4, "Finish OCR: (%s ...)" % txt[:12])
    # 如果图片内容>32，分词后返回即可
    if (eng and len(txt.split(" ")) > 32) or len(txt) > 32:
        tokenize(doc, txt, eng)
        callback(0.8, "OCR results is too long to use CV LLM.")
        return [doc]
    # 否则，用cv模型识别
    try:
        callback(0.4, "Use CV LLM to describe the picture.")
        ans = cv_mdl.describe(binary)
        callback(0.8, "CV LLM respoond: %s ..." % ans[:32])
        txt += "\n" + ans
    except Exception as e:
        callback(0.81, str(e))
    finally:
        tokenize(doc, txt, eng)
        callback(0.82, "Finally result: %s ..." % txt[:32])
        return [doc]

    return []

# if __name__ == "__main__":
#     def dummy(prog=None, msg=""):
#         pass
#
#     pdf = "技术培训.pdf"
#     docx = "技术培训.docx"
#     img = "454_4065_a3522f131d4c06d.jpg"
#     with open(img, 'rb') as f:
#         c = f.read()
#     chunk(img, c, "", 'eng', callback=dummy)
