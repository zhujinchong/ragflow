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

import logging
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from werkzeug.serving import run_simple
from api.apps import app
from api.db.services.document_service import DocumentService
from api.settings import (
    HOST, HTTP_PORT, access_logger, database_logger, stat_logger,
)
from api import utils

from api.db.db_models import init_database_tables as init_web_db
from api.db.init_data import init_web_data


def update_progress():
    while True:
        time.sleep(10)
        try:
            DocumentService.update_progress()
        except Exception as e:
            stat_logger.error("update_progress exception:" + str(e))


if __name__ == '__main__':
    print("""
    ____                 ______ __               
   / __ \ ____ _ ____ _ / ____// /____  _      __
  / /_/ // __ `// __ `// /_   / // __ \| | /| / /
 / _, _// /_/ // /_/ // __/  / // /_/ /| |/ |/ / 
/_/ |_| \__,_/ \__, //_/    /_/ \____/ |__/|__/  
              /____/                             

    """, flush=True)
    stat_logger.info(
        f'project base: {utils.file_utils.get_project_base_directory()}'
    )

    # init db
    init_web_db()
    init_web_data()
    # init runtime config

    peewee_logger = logging.getLogger('peewee')
    peewee_logger.propagate = False
    # rag_arch.common.log.ROpenHandler
    peewee_logger.addHandler(database_logger.handlers[0])
    peewee_logger.setLevel(database_logger.level)

    thr = ThreadPoolExecutor(max_workers=1)
    thr.submit(update_progress)

    # start http server
    try:
        stat_logger.info("RAG Flow http server start...")
        werkzeug_logger = logging.getLogger("werkzeug")
        for h in access_logger.handlers:
            werkzeug_logger.addHandler(h)
        DEBUG = True
        run_simple(hostname=HOST, port=HTTP_PORT, application=app, threaded=True, use_reloader=DEBUG, use_debugger=DEBUG)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGKILL)