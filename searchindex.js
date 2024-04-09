Search.setIndex({"docnames": ["fine-tuning", "introduction", "llama-cpp", "retrieval", "slurm_gpu_usage", "survey", "takeaways", "transformers", "use_case", "vision", "welcome"], "filenames": ["fine-tuning.ipynb", "introduction.ipynb", "llama-cpp.ipynb", "retrieval.ipynb", "slurm_gpu_usage.ipynb", "survey.ipynb", "takeaways.ipynb", "transformers.ipynb", "use_case.ipynb", "vision.ipynb", "welcome.ipynb"], "titles": ["Fine-Tuning", "<font color='purple'><strong>Introduction</strong></font>", "Using Llama_cpp_python to run LLMs", "Retrieval Augmented Generation", "<font color='purple'><strong>Using GPUs at Northwestern</strong></font>", "LLMs for Survey", "Summary and Takeaways", "Using Transformers to run LLMs", "<font color='purple'><strong>Example Use Case</strong></font>", "Process Images with bakllava", "Welcome"], "terms": {"thi": [0, 1, 2, 3, 9, 10], "websit": 10, "kellogg": [0, 1, 2, 3, 4, 7, 9, 10], "research": [1, 4, 10], "support": [3, 10], "open": [0, 1, 2, 3, 4, 9, 10], "sourc": [1, 3, 7, 9, 10], "llm": [1, 3, 9, 10], "cookbook": 10, "The": [0, 1, 2, 3, 4, 7, 9, 10], "book": 10, "aim": 10, "give": [0, 2, 10], "you": [0, 2, 3, 4, 7, 9, 10], "skill": [0, 10], "need": [0, 1, 2, 3, 7, 10], "work": [0, 2, 10], "effect": [0, 10], "larg": [0, 2, 3, 4, 7, 10], "languag": [0, 3, 10], "model": [3, 10], "1": [0, 2, 3, 4, 7, 9], "load": [2, 3, 4, 7, 9], "paramet": [0, 2, 4, 7], "2": [0, 2, 3, 4, 7, 9], "convert": [0, 2, 3, 7], "prompt": [0, 1, 2, 7, 9], "queri": [0, 2, 3, 7], "token": [0, 2, 3, 7, 9], "3": [0, 2, 3, 7, 9], "call": [2, 3, 7], "process": [0, 2, 3, 4, 7], "gener": [0, 1, 2, 4, 7, 9], "respons": [0, 2, 3, 4, 7, 9], "4": [0, 2, 3, 7, 9], "decod": [0, 2, 3, 7, 9], "text": [0, 2, 3, 7], "test_llama2": [3, 7], "py": [0, 2, 3, 7, 9], "from": [0, 1, 2, 3, 4, 7, 9], "import": [0, 2, 3, 4, 7, 9], "autotoken": [0, 3, 7], "automodelforcausallm": [0, 3, 7], "bitsandbytesconfig": [0, 7], "time": [0, 1, 2, 3, 4, 7, 9], "panda": [0, 7], "pd": 7, "pathlib": [3, 7], "path": [0, 2, 3, 4, 7], "start_tim": 3, "llm_dir": [0, 7, 9], "data": [0, 1, 2, 3, 7, 9], "llm_models_opensourc": [0, 3, 7, 9], "llama2_meta_huggingfac": [3, 7], "llm_model": [3, 7, 9], "meta": [2, 3, 7], "llama": [2, 3, 7], "7b": [0, 2, 3, 7], "chat": [3, 7], "hf": [3, 7, 9], "13b": [3, 7], "70b": 7, "quantization_config": [0, 7], "load_in_8bit": 7, "true": [0, 2, 3, 7, 9], "from_pretrain": [0, 3, 7, 9], "cache_dir": [0, 7, 9], "device_map": [0, 3, 7], "auto": [3, 7], "print": [0, 2, 3, 4, 7, 9], "f": [0, 3, 7], "second": [2, 7], "For": [0, 2, 3, 4, 7], "llama2": [0, 3, 9], "enclos": 7, "your": [0, 1, 2, 3, 4, 7, 9], "inst": [0, 3, 7], "tell": 7, "fun": 7, "fact": 7, "about": [1, 2, 3, 7], "busi": [0, 7], "school": 7, "devic": [0, 2, 3, 4, 7], "cuda": [0, 2, 7], "model_input": [0, 3, 7], "return_tensor": [0, 3, 7, 9], "pt": [0, 3, 7, 9], "customize_set": 7, "max_new_token": [0, 3, 7, 9], "400": [7, 9], "do_sampl": [0, 3, 7, 9], "temperatur": [2, 7], "0": [0, 2, 3, 4, 7, 9], "8": [0, 2, 7, 9], "custom": 7, "set": [0, 2, 3, 4, 7, 9], "kei": [0, 1, 2, 7], "valu": [2, 7], "item": 7, "output": [0, 1, 2, 3, 4, 7, 9], "skip_special_token": [7, 9], "end_tim": [], "execution_tim": [], "execut": [1, 4, 7], "finished_tim": 7, "strftime": 7, "y": 7, "m": [3, 7], "d": [7, 9], "h": 7, "s": [0, 2, 3, 4, 7, 9], "localtim": 7, "finish": [3, 7], "log": [3, 7], "column": [0, 7], "row": [0, 7], "append": [4, 7], "df": [0, 7], "datafram": [0, 7], "llm_name": 7, "split": [0, 2, 3, 7], "log_fil": 7, "log_": 7, "csv": [3, 7], "to_csv": 7, "index": [3, 7], "fals": [0, 2, 7, 9], "mode": 7, "header": 7, "exist": [0, 7], "rag": [0, 1, 2, 3], "power": [2, 3, 4], "paradigm": 3, "natur": 3, "combin": [0, 3], "strength": 3, "approach": [0, 3], "involv": [0, 3], "relev": [0, 3], "dataset": 3, "enhanc": [0, 3], "accur": [3, 9], "It": [0, 3], "can": [0, 1, 2, 3, 4, 7, 9], "anoth": 4, "method": [0, 3, 4], "fine": [1, 2, 3, 9], "tune": [1, 2, 3, 9], "phrase": 3, "come": [1, 2, 3, 4], "recent": 3, "paper": 3, "lewi": 3, "et": 3, "al": 3, "facebook": 3, "ai": [1, 3, 7], "idea": 3, "pre": [0, 2, 3], "lm": 3, "separ": [3, 4], "system": [3, 4, 7], "find": [0, 2, 3], "document": [0, 3], "condit": 3, "start": [0, 1, 2, 3, 4], "reli": 3, "5": [2, 3, 7, 9], "step": [2, 3], "differ": [0, 2, 3], "file": [0, 2, 3, 4, 9], "url": 3, "pdf": 3, "divers": [2, 3], "locat": [3, 4], "s3": 3, "storag": [0, 2, 3], "public": 3, "site": [0, 3, 7], "etc": [3, 7, 9], "transform": [0, 3, 8], "also": [0, 2, 3, 9], "determin": [2, 4], "part": [0, 4], "To": [0, 3, 4, 7], "prepar": 3, "larger": 3, "often": [], "necessari": [2, 3], "chunk": [3, 4], "emb": 3, "next": 7, "embed": [2, 3, 9], "must": [0, 4], "creat": [0, 2, 3, 4, 7], "captur": 3, "semant": 3, "mean": [2, 3, 4], "later": [3, 4], "enabl": 3, "mdoel": [], "effici": [0, 2, 3, 4], "other": [2, 3, 4], "piec": 3, "ar": [0, 1, 2, 3, 4, 9], "similar": [2, 3], "store": [2, 3], "vector": [3, 4], "search": 3, "retriv": [], "One": [1, 3, 4], "organ": [], "produc": [0, 3], "more": [0, 2, 3, 4], "context": [0, 2, 3], "awar": 3, "dure": [0, 2, 3, 9], "runtim": 3, "blend": 3, "rich": 3, "content": [3, 9], "taken": [1, 3], "http": [0, 2, 3, 7, 9], "doc": 3, "llamaindex": 3, "en": [2, 3], "stabl": 3, "_static": 3, "getting_start": 3, "basic_rag": 3, "png": 3, "question": [3, 4], "answer": [0, 3], "vast": [0, 3], "knowledg": [0, 1, 2, 3], "base": [0, 2, 3, 4], "multipl": 3, "creation": 3, "In": [0, 2, 4], "applic": [1, 2], "creativ": [1, 7], "pull": [0, 3], "detail": [2, 3, 9], "wide": [1, 2, 3], "rang": [2, 3], "convers": [2, 3], "agent": 3, "chatbot": 1, "benefit": [0, 2], "incorpor": [0, 2, 3], "extern": 3, "code": [0, 1, 2, 3, 9], "assist": [3, 9], "snippet": 3, "program": [0, 3, 4], "todai": 0, "prevent": 3, "hallucin": [1, 3], "bring": 3, "check": [2, 3, 4, 7], "whether": [2, 3, 4], "gpt": [2, 3], "provid": [0, 1, 2, 4, 9], "wai": [1, 2, 3, 4], "feed": [2, 3], "up": [0, 1, 2, 3, 4, 7], "date": [0, 1, 3], "wa": [0, 3], "between": [3, 9], "januari": 3, "2023": 3, "juli": 3, "mistral": [0, 3], "releas": 3, "septemb": 3, "let": [3, 4], "ask": [0, 3, 9], "run": [0, 1, 3, 4], "throught": [], "what": [1, 3], "i": [1, 3], "familiar": 2, "possibl": [2, 4, 9], "proprietari": [], "develop": 2, "specif": [0, 2, 3], "individu": 4, "known": [], "There": [1, 4], "mani": [1, 2, 4], "avail": [2, 4, 9], "each": 2, "own": [2, 3, 4, 7], "weak": [], "choos": [1, 2], "right": [0, 2, 3, 7], "some": [0, 1, 2, 3, 4, 7, 9], "popular": 7, "includ": [2, 7], "bert": [], "roberta": [], "xlnet": [], "These": [0, 2], "have": [0, 1, 3, 4, 9], "been": 9, "task": [0, 2, 3, 4], "sentiment": [], "analysi": [], "classif": 0, "If": 4, "ani": [0, 3], "its": [0, 1, 2, 3, 7], "capabl": 2, "perform": [0, 1, 3, 4], "mai": [1, 3, 9], "abl": [], "found": [2, 3], "here": [0, 2, 3, 4, 7, 9], "new": [0, 7, 9], "announc": 0, "resolv": [], "we": [0, 1, 2, 3, 4, 9], "note": [0, 2, 3], "below": [0, 2, 3, 4], "take": [0, 2, 3, 4, 9], "webpag": 3, "follow": [0, 2, 3, 7, 9], "outlin": 3, "abov": [3, 9], "librari": [0, 2, 3], "os": 3, "sy": 3, "llama_index": 3, "core": [3, 9], "vectorstoreindex": 3, "simpledirectoryread": 3, "storagecontext": 3, "load_index_from_storag": 3, "chatprompttempl": [], "response_synthes": [], "treesummar": [], "openai": [], "openaiembed": [], "node_pars": 3, "sentencesplitt": 3, "huggingfac": [0, 2, 3, 7, 9], "huggingfacellm": 3, "chatmessag": [], "messagerol": [], "faiss": 3, "vector_stor": 3, "faissvectorstor": 3, "torch": [0, 4, 9], "request": [0, 3, 4, 9], "bs4": 3, "beautifulsoup": 3, "basicconfig": 3, "stream": [3, 4, 9], "stdout": [3, 4], "level": 3, "info": [3, 7], "getlogg": 3, "addhandl": 3, "streamhandl": 3, "download": [1, 2], "web": 4, "url_link": 3, "get": [0, 2, 3, 4, 9], "soup": 3, "html": 3, "parser": 3, "webpage_cont": 3, "get_text": 3, "strip": 3, "data_fold": [], "test_data": 0, "mkdir": 3, "txt_file": 3, "txt": 3, "w": [0, 3], "write": [0, 3], "api": 2, "api_fil": [], "env": [0, 4, 7, 9], "r": 0, "environ": 4, "openai_api_kei": [], "read": 7, "embedding_model": [], "small": 2, "embed_model": 3, "embedding_chunk_s": 3, "512": [2, 3], "load_data": 3, "1536": [], "dimens": 3, "faiss_index": 3, "indexflatl2": 3, "storage_context": 3, "from_default": 3, "from_docu": 3, "chunk_siz": 3, "llama2_13b_chat": 3, "snapshot": 3, "29655417e51232f4f2b9b5d3e1418e5a9b04e80": 3, "selected_model": [], "context_window": 3, "4096": [2, 3], "256": 2, "tokenizer_nam": 3, "model_nam": 3, "query_engin": 3, "as_query_engin": 3, "select": [2, 3, 4], "httpx": [], "post": [], "com": 9, "v1": 9, "200": [], "ok": [], "7": 2, "3b": [], "size": [2, 9], "outperform": [], "all": [0, 2, 4], "benchmark": 1, "codellama": [], "group": [], "attent": 2, "gqa": [], "faster": [2, 4], "infer": 2, "slide": [], "window": 2, "swa": [], "handl": [0, 4], "longer": 4, "sequenc": 2, "smaller": [0, 4], "cost": [0, 1, 7], "under": [1, 2], "apach": [], "licens": 1, "without": 0, "restrict": [], "hug": [0, 1, 2, 7], "face": [0, 1, 2, 7], "co": [0, 2, 3, 7, 9], "directori": [2, 3, 7, 9], "gpu": [0, 2, 7, 9], "job": 4, "quest": [1, 4, 7], "chang": 0, "sbatch": [2, 4, 7, 9], "A": [0, 2, 4, 7, 9], "p": [7, 9], "line": [4, 7], "run_batch_llama2": [], "sh": [7, 9], "bin": [2, 4, 7, 9], "bash": [2, 4, 7, 9], "your_quest_allocation_account": 9, "gengpu": [2, 4, 7, 9], "gre": [2, 4, 7, 9], "a100": [2, 4, 7, 9], "n": [0, 7, 9], "t": [0, 2, 4, 7, 9], "30": [0, 2, 4, 7, 9], "00": [0, 2, 3, 4, 7, 9], "mem": [2, 4, 7, 9], "40g": [2, 4, 7, 9], "modul": [0, 2, 4, 7, 9], "purg": [2, 4, 7, 9], "mamba": [7, 9], "23": [7, 9], "hpc": [7, 9], "softwar": [0, 2, 4, 7, 9], "profil": [7, 9], "conda": [7, 9], "activ": [2, 4, 7, 9], "out": [0, 2, 4, 7], "test_mistr": 7, "tset_gemma": [], "python": [2, 9], "wrapper": 2, "c": 2, "implement": [1, 3], "architectur": [0, 2, 4], "access": [0, 2], "built": [], "cpp": 2, "framework": 2, "like": [0, 2, 4, 7, 9], "translat": [], "inform": [0, 2], "see": [0, 2, 4], "repo": [0, 2, 3], "ggerganov": [], "abetlen": [], "overview": [], "www": [], "datacamp": [], "tutori": [], "_": 0, "compat": [], "weight": [0, 2, 7], "proviv": [], "sampl": [0, 2, 9], "gemma": [], "instruct": [0, 2, 3, 7], "llama_cpp": [2, 9], "input": [2, 3, 4, 9], "model_path": 2, "gguf": 9, "context_s": 2, "float": 2, "basic": 0, "summari": [2, 3], "gui": 2, "debord": 2, "societ": 2, "du": 2, "spectacl": 2, "written": 2, "syntax": [0, 2], "start_of_turn": 2, "user": [2, 3, 9], "end_of_turn": 2, "THE": [], "n_ctx": [2, 9], "max": 2, "length": [2, 3], "adjust": [0, 2, 3, 4], "requir": [0, 2, 3, 4], "n_thread": [2, 9], "number": [0, 2, 4, 9], "cpu": [2, 9], "thread": 2, "n_gpu_lay": [2, 9], "want": [2, 3], "onli": [0, 2, 4], "send": [2, 4], "concis": 0, "max_token": 2, "1000": [0, 2, 3, 4], "response_text": 2, "choic": [1, 2, 4, 9], "save": [0, 2, 3, 4, 9], "klc": [1, 2, 4], "bound": 2, "our": [0, 1, 2, 3, 4, 9], "modulefil": [2, 4, 9], "38": [2, 9], "python3": [0, 2, 9], "llama_cpp_test": [], "consist": [0, 2], "billion": [], "order": [1, 2, 4], "speed": [2, 4], "node": 2, "alloc": [2, 4], "northwestern": 7, "edu": [], "depart": [], "servic": 4, "comput": [2, 7], "type": [2, 4, 7, 9], "ll": [0, 2, 4], "submit": [0, 4], "through": [0, 1, 2, 3, 4], "schedul": 4, "allow": [0, 2, 4], "test": 4, "tensor": [2, 4], "admin": [], "gpu_test_fil": [], "pytorch_gpu_test": [], "is_avail": 4, "device_count": 4, "get_device_nam": 4, "els": [0, 3, 4], "accordingli": 4, "being": [0, 2, 4], "us": [0, 9], "two": [4, 7], "random": [2, 4], "tensor1": 4, "randn": 4, "tensor2": 4, "add": [0, 3, 4], "oper": 4, "result": [0, 1, 2, 4], "so": [4, 7], "long": [], "launch": 4, "pytorch": [2, 4], "break": [0, 2], "down": [0, 2], "srun": [], "an": [2, 4], "interact": [], "partit": [2, 4], "direct": 4, "genom": 4, "cluster": [1, 4], "account": [2, 4], "xxxxx": [], "refer": [], "given": [2, 3, 4], "specifi": [1, 2, 4], "sinc": 2, "cannot": [], "ntask": [2, 4], "per": [2, 4], "how": [0, 1, 2, 4, 9], "paralleliz": 4, "otherwis": 4, "slow": 4, "stand": [], "resourc": [0, 2], "By": 2, "exclud": [], "indic": [0, 4], "minut": 4, "much": [1, 2, 4], "memori": [0, 2, 4, 7], "after": [0, 1, 4], "final": [0, 4], "receiv": [], "id": [], "batch": 3, "9428806": [], "onc": [0, 3], "complet": [0, 1, 4], "job_id": [], "current": 7, "llm_core": 9, "llavacppmodel": 9, "q4_k_m": 9, "name": [0, 2, 3, 7, 9], "llama_cpp_kwarg": 9, "logits_al": 9, "8000": 9, "verbos": [0, 9], "don": [2, 9], "clip_model_path": 9, "clip": 9, "load_model": 9, "histori": 9, "role": [3, 9], "image_url": 9, "advanc": [2, 9], "stack": 9, "asset": 9, "img": 9, "mappemond": 9, "jpg": 9, "describ": [0, 9], "messag": [3, 9], "pleas": [0, 2, 4, 9], "home": 9, "slurm": [2, 9], "script": [0, 2, 9], "image_test": 9, "info_": [], "test_gemma": 7, "project": [3, 4, 7], "space": 7, "option": [2, 4, 7], "veri": [0, 2, 7], "reduc": [0, 2, 7], "repres": [2, 7, 9], "lower": [2, 7], "precis": [2, 7], "form": 8, "demonstr": [0, 2, 4, 8], "reproduc": [1, 8], "principl": 8, "huggingfaceembed": 3, "embedding_nam": 3, "baai": [], "bge": [], "intfloat": 3, "multilingu": 3, "e5": 3, "max_length": 3, "generate_kwarg": [], "top_p": 2, "9": 2, "top_k": 2, "1024": 3, "100": [0, 3, 7, 9], "66": [], "66it": [], "variou": [], "while": [0, 1, 2, 4, 7], "remain": [], "good": [0, 1, 3], "english": [], "easi": [], "ha": [2, 4, 7], "gemma_test": 2, "likewis": [], "mistral_test": [], "llama2_test": 2, "respect": [], "subfold": [], "workshop": [0, 1, 4], "leverag": [2, 4], "do": [1, 2, 3, 4], "temporari": 4, "pxxxxxx": [], "afterward": 4, "purpos": [], "primarili": 2, "calcul": [0, 2, 4], "hear": [], "talk": [], "section": [], "term": 0, "equip": 4, "both": [2, 4, 9], "processor": [4, 9], "graphic": 4, "card": 4, "central": 4, "unit": 4, "mathemat": 4, "logic": 4, "nutshel": 4, "extrem": 4, "most": [0, 2, 4], "infinitesim": 4, "short": [0, 4], "amount": [3, 4], "one": [0, 1, 4], "thing": 4, "sequenti": [0, 4], "solv": 4, "singl": [2, 4], "problem": 4, "simultan": 4, "essenti": 2, "distribut": 4, "over": 4, "special": [0, 2, 4, 9], "hardwar": 4, "compon": 4, "comparison": 4, "24": [2, 4], "entitl": [], "contain": [0, 4], "6": [2, 4], "912": 4, "whiel": [], "less": [2, 4], "than": [0, 2, 4], "sheer": 4, "volum": 4, "them": [0, 1, 4], "make": [0, 2, 3, 4, 9], "ideal": 4, "better": [0, 1, 2, 4, 9], "why": 4, "aren": 4, "quickli": [], "actual": [3, 4], "instanc": [], "sort": [], "list": 0, "divid": [], "among": [], "portion": [], "commun": 4, "coordin": 4, "rel": [], "across": 4, "might": [0, 2, 4], "compelt": [], "alon": 4, "potenti": [0, 2, 4], "ineffici": 4, "rais": 4, "know": [1, 4, 7], "when": [0, 2, 4], "where": [0, 4, 7], "unifi": [2, 4], "platform": 4, "help": [0, 2, 4, 9], "intensitv": [], "switch": [], "certain": 4, "folder": [0, 3], "github": [0, 3, 4], "goal": 1, "look": 1, "best": [0, 1], "practic": 1, "regard": 1, "linux": [1, 4], "adapt": [0, 7], "improv": [2, 3], "integr": [], "lifecycl": [], "diagram": [1, 2], "imag": [2, 4], "deeplearn": [1, 7], "common": 1, "defin": [], "case": [0, 2, 4], "hub": [0, 1], "leaderboard": 1, "e32337": [2, 4, 7], "On": 4, "intens": 4, "optim": [0, 2, 4], "which": [0, 2, 3, 4, 7, 9], "deleg": 4, "gpu_pytorch_test": [], "simpli": 4, "explain": 4, "pil": 9, "autoprocessor": 9, "llavaforconditionalgener": 9, "llava": 9, "nyou": 9, "mom": 9, "ag": [0, 9], "25": [2, 9], "market": [0, 9], "survei": 9, "diaper": 9, "packag": [0, 9], "appeal": 9, "nassist": 9, "image_fil": 9, "i5": 9, "walmartimag": 9, "seo": 9, "babi": 9, "newborn": 9, "10": [0, 2, 9], "lb": 9, "120": 9, "count": [1, 9], "pamper": 9, "swaddler": 9, "onemonth": 9, "suppli": 9, "vary_95ea2d1c": 9, "769a": 9, "4ec2": 9, "8da8": 9, "7e6ea4829e55": 9, "3c890f8095b5b30ae263c996151f9575": 9, "jpeg": 9, "torch_dtyp": [0, 9], "float16": [0, 2, 9], "low_cpu_mem_usag": [0, 9], "raw_imag": 9, "raw": 9, "checkpoint": [0, 7, 9], "shard": [0, 7, 9], "01": [2, 9], "61it": 9, "75": 9, "81it": 9, "02it": 9, "ad": [3, 9], "vocabulari": 9, "sure": [0, 3, 7, 9], "associ": [0, 9], "word": [2, 9], "train": [0, 1, 2, 9], "er": 9, "me": 9, "featur": 9, "heartwarm": 9, "parent": 9, "hold": 9, "evok": 9, "feel": 9, "love": 9, "care": [2, 9], "joi": 9, "parenthood": 9, "shown": [4, 9], "rest": 9, "comfort": 9, "pad": [0, 2, 3, 9], "sens": [3, 9], "secur": 9, "warmth": 9, "infant": 9, "sooth": 9, "colicki": 9, "ensur": [2, 9], "sleep": 9, "connect": 9, "tender": 9, "moment": 9, "experi": [0, 9], "brand": 9, "understand": [0, 3, 4, 9], "bond": 9, "live": 9, "runn": 9, "test_bakllava": 9, "tranform": [2, 3], "tool": 0, "appli": [0, 2, 3], "deriv": 2, "thei": [2, 4], "howev": [0, 2], "slightli": [0, 2], "easier": 2, "upshot": 2, "tweak": 2, "techniqu": 2, "normal": 2, "stabil": 2, "befor": [0, 2], "consid": 2, "akin": 2, "guitar": 2, "song": 2, "string": [2, 3], "tension": 2, "sound": 2, "util": [0, 2], "swiglu": 2, "function": [0, 2], "pick": 2, "grip": 2, "strum": 2, "chord": 2, "plai": [2, 4], "simpler": [0, 2], "posit": 2, "rotari": 2, "track": 2, "compar": [0, 2], "absolut": 2, "imagin": [0, 2], "fret": 2, "neck": 2, "were": [2, 4], "instead": [0, 2], "just": [2, 3], "metal": 2, "bar": 2, "quicker": 2, "fretboard": 2, "addit": [2, 3], "llma_cpp_python": 2, "unlik": [0, 2], "intricaci": 2, "technic": 2, "hood": 2, "focu": [0, 2], "directli": 2, "format": [0, 2, 3], "either": 2, "plain": 2, "place": [0, 2], "appropri": [2, 3], "retriev": [0, 1, 2], "build": 2, "block": 2, "complex": [0, 2], "multidimension": 2, "arrai": 2, "chosen": 2, "float64": 2, "even": [0, 2, 4], "integ": 2, "impact": [2, 7], "accuraci": [0, 1, 2], "usag": [0, 2], "think": 2, "digit": 2, "higher": [0, 2], "e": 2, "g": 2, "offer": [2, 4], "greater": 2, "introduc": 2, "slight": 2, "trade": 2, "off": [0, 2], "simplifi": 2, "compress": 2, "further": 2, "limit": 2, "quantizaton": 2, "feasibl": 2, "theblok": 2, "though": [0, 2], "suggest": 2, "factual": [0, 2], "comprehens": 2, "experiment": 2, "slower": 2, "overhead": [2, 4], "outweigh": 2, "tini": 2, "loss": 2, "neglig": 2, "high": 2, "full": 2, "could": [2, 3, 4], "prefer": [0, 2, 4], "crucial": [0, 2], "deploi": 2, "max_tokens_select": 2, "temperature_select": 2, "top_p_select": 2, "top_k_select": 2, "int": 2, "kind": [2, 4], "pickup": 2, "esp": 2, "ltd": 2, "alexi": 2, "rip": 2, "prompt_sytnax": 2, "echo": 2, "At": 2, "default": 2, "pass": 2, "maximum": 2, "determinist": 2, "end": 2, "lead": [0, 2], "control": 2, "predict": [2, 3], "probabl": 2, "whose": 2, "cumul": 2, "exce": 2, "threshold": 2, "zero": 2, "increas": 2, "chanc": 2, "boolean": 2, "origin": 2, "begin": 2, "ones": 2, "readthedoc": 2, "io": 2, "latest": [2, 4], "ggml_init_cubla": 2, "disabl": 2, "llama_model_load": 2, "19": 2, "pair": 2, "254": 2, "version": [1, 2], "v3": 2, "dump": [0, 2], "metadata": 2, "kv": 2, "overrid": 2, "str": [0, 2], "context_length": 2, "u32": 2, "8192": 2, "block_count": 2, "28": 2, "embedding_length": 2, "3072": 2, "feed_forward_length": 2, "24576": 2, "head_count": 2, "16": [0, 2, 7], "head_count_kv": 2, "key_length": 2, "value_length": 2, "layer_norm_rms_epsilon": 2, "f32": 2, "000001": 2, "11": [2, 7], "ggml": 2, "12": [0, 2], "bos_token_id": 2, "13": 2, "eos_token_id": [0, 2], "14": [0, 2], "padding_token_id": 2, "15": 2, "unknown_token_id": 2, "arr": 2, "256128": 2, "eo": 2, "bo": 2, "unk": 2, "17": 2, "score": 2, "000000": 2, "0000": 2, "18": [2, 4], "token_typ": 2, "i32": 2, "llm_load_vocab": 2, "mismatch": 2, "definit": 2, "544": 2, "vs": 1, "388": 2, "llm_load_print_meta": 2, "arch": 2, "vocab": 2, "spm": 2, "n_vocab": 2, "n_merg": 2, "n_ctx_train": 2, "n_embd": 2, "n_head": 2, "n_head_kv": 2, "n_layer": 2, "n_rot": 2, "192": 2, "n_embd_head_k": 2, "n_embd_head_v": 2, "n_gqa": 2, "n_embd_k_gqa": 2, "n_embd_v_gqa": 2, "f_norm_ep": 2, "0e": 2, "f_norm_rms_ep": 2, "06": 2, "f_clamp_kqv": 2, "f_max_alibi_bia": 2, "n_ff": 2, "n_expert": 2, "n_expert_us": 2, "rope": 2, "scale": 2, "linear": 2, "freq_base_train": 2, "10000": 2, "freq_scale_train": 2, "n_yarn_orig_ctx": 2, "rope_finetun": 2, "unknown": 2, "ftype": 2, "guess": 2, "param": 2, "54": 2, "b": 2, "31": 2, "81": [2, 7], "gib": 2, "32": [0, 2], "bpw": 2, "lf": 2, "227": 2, "0x0a": 2, "llm_load_tensor": 2, "ctx": 2, "mib": 2, "offload": 2, "repeat": 2, "layer": 2, "non": 2, "29": 2, "buffer": 2, "32570": 2, "llama_new_context_with_model": 2, "freq_bas": 2, "freq_scal": 2, "warn": [0, 2], "fail": 2, "224": 2, "mb": 2, "pin": 2, "detect": [0, 2], "llama_kv_cache_init": 2, "self": [2, 7], "k": 2, "f16": 2, "112": 2, "v": 2, "506": 2, "cuda_host": 2, "graph": 2, "measur": 2, "avx": 2, "avx_vnni": 2, "avx2": 2, "avx512": 2, "avx512_vbmi": 2, "avx512_vnni": 2, "fma": 2, "neon": 2, "arm_fma": 2, "f16c": 2, "fp16_va": 2, "wasm_simd": 2, "bla": 2, "sse3": 2, "ssse3": 2, "vsx": 2, "matmul_int8": 2, "llama_print_tim": 2, "1047": 2, "55": 2, "ms": 2, "111": 2, "74": 2, "34": 2, "304": 2, "eval": 2, "37": 2, "69": 2, "82": [0, 2], "136433": 2, "05": [0, 2, 7], "33": 2, "4134": 2, "total": [0, 2], "138373": 2, "07": 2, "48": [2, 7], "ripper": 2, "ceram": 2, "humbuck": 2, "coil": 2, "tap": 2, "clean": 2, "overdr": 2, "tone": [0, 2], "everyth": [0, 2], "misral_test": 2, "git": 2, "tuturi": 2, "exampl": [0, 2, 4], "adn": 2, "engin": [0, 2], "guid": [0, 2], "depend": [3, 4], "serial": 4, "reason": [0, 4], "nvidia": 4, "dozen": 4, "show": 4, "via": 4, "demand": 4, "exclus": 4, "ident": 4, "doe": [0, 2], "teach": 7, "materi": 7, "quantiz": [0, 7], "uncom": 7, "comment": 7, "50": 7, "08": 7, "98": 7, "65569519996643": 7, "manag": 7, "univers": 7, "did": 7, "campu": 7, "ic": 7, "cream": 7, "shop": 7, "ye": 7, "spot": 7, "student": 7, "indulg": 7, "sweet": 7, "tooth": 7, "enjoi": 7, "varieti": 7, "flavor": 7, "uniqu": 7, "chocol": 7, "chip": 7, "cooki": 7, "dough": 7, "cinnamon": 7, "swirl": 7, "food": 7, "societi": 7, "cours": [4, 7], "focus": [0, 7], "social": 7, "cultur": 7, "re": 7, "stop": 7, "treat": 7, "learn": [0, 7], "751539945602417": 7, "2024": [1, 7], "04": [1, 7], "03": [0, 1, 7], "repositori": 4, "jupyt": 4, "notebook": [0, 4], "constraint": 4, "pcie": 4, "j": 4, "micromamba": 4, "load_hook": 4, "sxm": 4, "80gb": 4, "40gb": 4, "mistralai": [0, 3, 7], "v0": [0, 3, 7], "googl": [4, 7], "beyond": 0, "tailor": 0, "desir": 0, "sever": 0, "advantag": [0, 1], "superior": 0, "deliv": 0, "qualiti": 0, "becaus": 0, "wider": [], "fit": [], "latenc": 0, "excel": 0, "style": 0, "reliabl": 0, "achiev": [0, 1], "tackl": 0, "edg": 0, "empow": 0, "master": 0, "difficult": 0, "articul": 0, "simpl": 0, "delv": 0, "mechan": 0, "alwai": [0, 1], "solut": 0, "breakdown": 0, "extens": 0, "chain": 0, "augment": [0, 1], "situat": 0, "databas": 0, "benefici": 0, "particular": 0, "behavior": 0, "express": 0, "advic": 0, "craft": [0, 1], "attempt": 0, "especi": [0, 4], "apt": 0, "first": 0, "should": [0, 1], "properli": 0, "box": 0, "ag_new": 0, "classifi": 0, "categori": 0, "world": 0, "sport": 0, "scienc": 0, "technolog": 0, "query_initi": 0, "fine_tun": 0, "initi": 0, "bitsandbyt": 0, "bnb": 0, "mistral_mistralai": 0, "model_id": 0, "bnb_config": 0, "load_in_4bit": 0, "bnb_4bit_use_double_qu": 0, "bnb_4bit_quant_typ": 0, "nf4": 0, "bnb_4bit_compute_dtyp": 0, "bfloat16": 0, "add_eos_token": 0, "def": [0, 3, 7, 9], "get_complet": 0, "prompt_templ": 0, "four": 0, "class": [0, 3], "sci": 0, "tech": 0, "encod": [0, 3], "add_special_token": 0, "generated_id": [0, 3], "pad_token_id": 0, "batch_decod": [0, 3], "return": 0, "wall": 0, "st": 0, "bear": 0, "claw": 0, "back": 0, "Into": 0, "black": 0, "reuter": 0, "seller": 0, "street": 0, "dwindl": 0, "band": 0, "ultra": 0, "cynic": 0, "green": 0, "again": 0, "lib": 0, "diffus": 0, "63": [0, 3], "userwarn": 0, "_pytre": 0, "_register_pytree_nod": 0, "deprec": 0, "register_pytree_nod": 0, "correct": 0, "padding_sid": 0, "left": 0, "discuss": 0, "who": 0, "typic": 0, "financ": 0, "invest": 0, "fall": 0, "within": 0, "mention": 0, "strong": 0, "articl": 0, "greent": 0, "monei": 0, "doesn": 0, "necessarili": 0, "profit": 0, "capit": 0, "gain": 0, "financi": 0, "addition": 0, "relat": 0, "those": 0, "domain": 0, "notic": 0, "prime": 0, "As": 0, "earlier": 0, "json": 0, "prepare_data": 0, "load_dataset": 0, "create_text_row": 0, "text_row": 0, "label": 0, "jsonl": 0, "process_dataframe_to_jsonl": 0, "output_file_path": 0, "output_jsonl_fil": 0, "iterrow": 0, "json_object": 0, "train_test_split": 0, "test_siz": 0, "train_data": 0, "train_json_fil": 0, "proj": 0, "awc6034": 0, "test_json_fil": 0, "train_df": 0, "to_panda": 0, "test_df": 0, "head": 0, "christoph": 0, "lee": 0, "becom": 0, "video": [0, 4], "game": 0, "wizard": 0, "law": 0, "respond": 0, "internet": 0, "revolut": 0, "curren": 0, "wood": 0, "vijai": 0, "fame": 0, "nice": 0, "alcoa": 0, "aluminium": 0, "india": 0, "laud": 0, "arafat": 0, "lifetim": 0, "devot": 0, "pale": 0, "peft": 0, "toolbox": 0, "fill": 0, "standard": 0, "would": 0, "everi": [0, 1], "try": 0, "toolkit": 0, "updat": [0, 1], "retrain": 0, "lora": 0, "low": 0, "rank": 0, "identifi": 0, "area": 0, "act": 0, "spotlight": 0, "shine": 0, "network": 0, "target": 0, "sft": 0, "supervis": 0, "intern": 0, "abrupt": 0, "gentler": 0, "nuanc": 0, "refin": 0, "drastic": 0, "alter": 0, "overal": 0, "sfttrainer": 0, "Then": 0, "merg": 0, "togeth": 0, "futur": 0, "made": 0, "trainingargu": 0, "datacollatorforlanguagemodel": 0, "loraconfig": 0, "get_peft_model": 0, "peftmodel": 0, "load_from_disk": 0, "find_all_linear_nam": 0, "cl": 0, "nn": 0, "linear4bit": 0, "lora_module_nam": 0, "named_modul": 0, "isinst": 0, "len": 0, "lm_head": 0, "bit": 0, "remov": 0, "gradient_checkpointing_en": 0, "prepare_model_for_kbit_train": 0, "configur": 0, "lora_config": 0, "lora_alpha": 0, "target_modul": 0, "lora_dropout": 0, "bia": 0, "none": 0, "task_typ": 0, "causal_lm": 0, "trainabl": 0, "get_nb_trainable_paramet": 0, "percentag": 0, "4f": 0, "pad_token": 0, "eos_token": 0, "clear": 0, "cach": 0, "empty_cach": 0, "setup": 0, "trainer": 0, "train_dataset": 0, "eval_dataset": 0, "dataset_text_field": 0, "peft_config": 0, "arg": 0, "per_device_train_batch_s": 0, "gradient_accumulation_step": 0, "warmup_step": 0, "max_step": 0, "learning_r": 0, "2e": 0, "logging_step": 0, "output_dir": 0, "paged_adamw_8bit": 0, "save_strategi": 0, "epoch": 0, "data_col": 0, "mlm": 0, "new_model": 0, "push": 0, "save_pretrain": 0, "base_model": 0, "return_dict": 0, "merged_model": 0, "merge_and_unload": 0, "safe_seri": 0, "revisit": 0, "query_finetun": 0, "llm_path": 0, "appear": 0, "beginn": 0, "qlora": 0, "colab": [0, 4], "finetun": 0, "explan": 0, "still": 0, "shorter": 0, "alreadi": 0, "newer": 0, "cutoff": [0, 1], "few": 0, "shot": 0, "thought": 0, "effort": 0, "collect": 0, "extra": 0, "goe": 1, "least": 1, "success": 1, "well": [1, 3], "plan": 1, "my": 1, "evalu": 1, "enough": 1, "avaialbl": 1, "local": 1, "privaci": 1, "flexibl": 1, "arena": 1, "sometim": [1, 4], "No": 1, "anyth": 1, "event": 1, "occur": 1, "overcom": 1, "obstacl": 1, "tehcniqu": 1, "fewer": 1, "instal": 3, "bertmodel": 3, "uncas": 3, "input_text": 3, "input_id": 3, "attention_mask": 3, "truncat": 3, "interpret": 3, "consult": 3, "page": 3, "link": 3, "prep_text": 3, "workdir": 3, "linesep": 3, "join": 3, "splitlin": 3, "data_path": 3, "llamaindex_data": 3, "process_llamaindex": 3, "embed_d": 3, "persist_dir": 3, "llamaindex_storage_faiss": 3, "persist": 3, "from_persist_dir": 3, "replac": 3, "your_file_directori": 3, "file_dir": 3, "e5_infloat": 3, "baa7be480a7de1539afce709c8f13f833a510e0a": 3, "llama2_7b_chat": 3, "92011f62d7604e261f748ec0cfe6329f31193e33": 3, "21": [], "pip": 3, "favourit": 3, "condiment": 3, "quit": 3, "partial": 3, "squeez": 3, "fresh": 3, "lemon": 3, "juic": 3, "zesti": 3, "flavour": 3, "whatev": 3, "cook": 3, "kitchen": 3, "mayonnais": 3, "recip": 3, "apply_chat_templ": 3, "generated_respons": [], "autotokenizerforcausallm": [], "causallm": [], "free": 4, "browser": 4, "cloud": 4, "amazon": 4, "microsoft": 4, "azur": 4, "price": 4, "sprung": 4, "paperspac": 4, "bui": 4, "budget": 4, "expertis": 4, "64": 4, "2tb": 4, "share": 4, "ram": 4, "theori": 4, "h100": 4, "astound": 4, "432": 4, "matrix": 4, "design": 4, "speedup": 4, "magnitud": 4, "stderr": 4, "interfac": 4, "command": 4, "termin": [3, 4], "run_llama2": 7, "run_bakllava": 9, "51it": 3, "hyperparamet": 3, "placehold": 3}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"introduct": 1, "welcom": 10, "fine": 0, "tune": 0, "how": 3, "us": [1, 2, 3, 4, 7, 8], "llama": [], "cpp": [], "run": [2, 7, 9], "llm": [2, 4, 5, 7], "retriev": 3, "augment": 3, "gener": 3, "slurm": [4, 7], "job": [], "manag": [], "gpu": 4, "usag": [], "survei": 5, "summari": 6, "takeawai": 6, "transform": [2, 7, 9], "vision": [], "imag": 9, "text": [], "basic": [2, 7], "workflow": [2, 7], "work": 3, "sampl": [3, 4], "case": [1, 3, 8], "exampl": [3, 8], "inform": 3, "non": 3, "exist": 3, "train": 3, "llama2": [2, 7], "model": [0, 1, 2, 7, 9], "python": [4, 7], "script": [3, 4, 7], "mistral": [2, 7], "gemma": [2, 7], "font": [0, 1, 2, 3, 4, 7, 8, 9], "color": [0, 1, 2, 3, 4, 7, 8, 9], "purpl": [0, 1, 2, 3, 4, 7, 8, 9], "llama_cpp_python": [2, 9], "1": [], "code": 4, "2": [], "process": [8, 9], "bakllava": 9, "10k": 8, "understand": [], "parallel": 4, "comput": 4, "more": [], "object": 1, "cpu": 4, "limit": [], "cuda": 4, "access": 4, "node": 4, "refer": [0, 2, 4], "sourc": [0, 2, 4], "doe": [], "differ": [], "from": [], "quantiz": 2, "what": 2, "gguf": 2, "tradeoff": [0, 2], "between": [0, 2], "northwestern": 4, "resourc": [1, 4], "vs": 2, "tool": 2, "techniqu": 0, "load": 0, "test": 0, "origin": 0, "prepar": 0, "dataset": 0, "step": 0, "project": 1, "lifecycl": 1, "defin": 1, "select": 1, "adapt": 1, "improv": 1, "integr": 1, "extern": 1, "multipl": 4, "core": 4, "pytorch_gpu_test": 4, "py": 4, "sh": 4, "break": 4, "down": 4, "thi": 4}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 56}})