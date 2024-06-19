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
import os
import re
from io import BytesIO

import pdfplumber
from PIL import Image
from ruamel.yaml import YAML

from api.db import FileType

PROJECT_BASE = os.getenv("RAG_PROJECT_BASE") or os.getenv("RAG_DEPLOY_BASE")


def get_project_base_directory(*args):
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
            )
        )

    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE


def load_yaml_conf(conf_path):
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(conf_path, encoding='utf-8') as f:
            yaml = YAML(typ='safe', pure=True)
            return yaml.load(f)
    except Exception as e:
        raise EnvironmentError(
            "loading yaml file config from {} failed:".format(conf_path), e
        )


def filename_type(filename):
    filename = filename.lower()
    if re.match(r".*\.pdf$", filename):
        return FileType.PDF.value

    if re.match(
            r".*\.(doc|docx|ppt|pptx|yml|xml|htm|json|csv|txt|ini|xls|xlsx|wps|rtf|hlp|pages|numbers|key|md|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|html)$",
            filename):
        return FileType.DOC.value

    if re.match(
            r".*\.(wav|flac|ape|alac|wavpack|wv|mp3|aac|ogg|vorbis|opus|mp3)$", filename):
        return FileType.AURAL.value

    if re.match(
            r".*\.(jpg|jpeg|png|tif|gif|pcx|tga|exif|fpx|svg|psd|cdr|pcd|dxf|ufo|eps|ai|raw|WMF|webp|avif|apng|icon|ico|mpg|mpeg|avi|rm|rmvb|mov|wmv|asf|dat|asx|wvx|mpe|mpa|mp4)$",
            filename):
        return FileType.VISUAL.value

    return FileType.OTHER.value


def thumbnail(filename, blob):
    filename = filename.lower()
    if re.match(r".*\.pdf$", filename):
        pdf = pdfplumber.open(BytesIO(blob))
        buffered = BytesIO()
        pdf.pages[0].to_image(resolution=32).annotated.save(buffered, format="png")
        return "data:image/png;base64," + \
            base64.b64encode(buffered.getvalue()).decode("utf-8")

    if re.match(r".*\.(jpg|jpeg|png|tif|gif|icon|ico|webp)$", filename):
        image = Image.open(BytesIO(blob))
        image.thumbnail((30, 30))
        buffered = BytesIO()
        image.save(buffered, format="png")
        return "data:image/png;base64," + \
            base64.b64encode(buffered.getvalue()).decode("utf-8")

    if re.match(r".*\.(ppt|pptx)$", filename):
        import aspose.slides as slides
        import aspose.pydrawing as drawing
        try:
            with slides.Presentation(BytesIO(blob)) as presentation:
                buffered = BytesIO()
                presentation.slides[0].get_thumbnail(0.03, 0.03).save(
                    buffered, drawing.imaging.ImageFormat.png)
                return "data:image/png;base64," + \
                    base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            pass


def traversal_files(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname
