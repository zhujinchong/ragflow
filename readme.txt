启动大模型、嵌入模型#####################################################################################
docker pull ollama/ollama:latest
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama /bin/bash
ollama pull qwen:0.5b-chat
ollama pull shaw/dmeta-embedding-zh-small
# 只需要启动LLM，嵌入模型不需要启动
ollama run qwen:0.5b-chat

启动数据库、前端#####################################################################################
cd ./docker
# 注意：前端需要改成本地ip
docker-compose -f docker-compose-base.yml -p ragflow up -d

ES数据#################################################################################################
d["doc_id"] =
d["kb_id"] = [""]
d["_id"] = md5.hexdigest(content+doc_id)
d["img_id"] = "{}-{}".format(row["kb_id"], d["_id"]) # MINIO.put(row["kb_id"], d["_id"], output_buffer.getvalue())
d["create_time"] = "2024-06-17 21:42:13"
d["create_timestamp_flt"] = 1.718631733036242E9

d["docnm_kwd"] = "技术培训.pdf"	# filename
d["title_tks"] = "技术培训" 	# rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
d["title_sm_tks"] = "技术 培训" # rag_tokenizer.fine_grained_tokenize(doc["title_tks"])

d["q_%d_vec"] = []
d["content_with_weight"] = content
d["content_ltks"] = rag_tokenizer.tokenize(re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", content))
d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])

d["page_num_int"] = [2]
d["position_int"] = [[2,82,506,106,425]]
d["top_int"] = [106]

d["important_kwd"] = chunk中自定义关键字
ES Query#################################################################################################
