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
import os
from enum import IntEnum, Enum

from api.utils.file_utils import get_project_base_directory
from api.utils.log_utils import LoggerFactory, getLogger

# Logger
LoggerFactory.set_directory(
    os.path.join(
        get_project_base_directory(),
        "logs",
        "api"))
# {CRITICAL: 50, FATAL:50, ERROR:40, WARNING:30, WARN:30, INFO:20, DEBUG:10, NOTSET:0}
LoggerFactory.LEVEL = 30

stat_logger = getLogger("stat")
access_logger = getLogger("access")
database_logger = getLogger("database")
chat_logger = getLogger("chat")

from rag.utils.es_conn import ELASTICSEARCH
from rag.nlp import search
from api.utils import get_base_config

API_VERSION = "v1"
RAG_FLOW_SERVICE_NAME = "ragflow"

HOST = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("host", "127.0.0.1")
HTTP_PORT = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("http_port")
SECRET_KEY = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("secret_key", "infiniflow")

# 数据库的默认值
LLM_FACTORY = "DefaultLLM"
CHAT_MDL = "default-chat"
EMBEDDING_MDL = "default-embedding"
RERANK_MDL = "default-rerank"
ASR_MDL = "default-asr"
IMAGE2TEXT_MDL = "default-vl"
LLM_BASE_URL = ""
API_KEY = ""
PARSERS = "naive:General,qa:Q&A,paper:Paper,laws:Laws,picture:Picture"

# MySQL
DATABASE = get_base_config("mysql")

# authentication
AUTHENTICATION_CONF = get_base_config("authentication", {})
CLIENT_AUTHENTICATION = AUTHENTICATION_CONF.get("client", {}).get("switch", False)
HTTP_APP_KEY = AUTHENTICATION_CONF.get("client", {}).get("http_app_key")

# OAuth
GITHUB_OAUTH = get_base_config("oauth", {}).get("github")
FEISHU_OAUTH = get_base_config("oauth", {}).get("feishu")

retrievaler = search.Dealer(ELASTICSEARCH)


class CustomEnum(Enum):
    @classmethod
    def valid(cls, value):
        try:
            cls(value)
            return True
        except BaseException:
            return False

    @classmethod
    def values(cls):
        return [member.value for member in cls.__members__.values()]

    @classmethod
    def names(cls):
        return [member.name for member in cls.__members__.values()]


class RetCode(IntEnum, CustomEnum):
    SUCCESS = 0
    NOT_EFFECTIVE = 10
    EXCEPTION_ERROR = 100
    ARGUMENT_ERROR = 101
    DATA_ERROR = 102
    OPERATING_ERROR = 103
    CONNECTION_ERROR = 105
    RUNNING = 106
    PERMISSION_ERROR = 108
    AUTHENTICATION_ERROR = 109
    SERVER_ERROR = 500
