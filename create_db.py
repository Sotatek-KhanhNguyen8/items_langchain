from langchain_community.vectorstores import FAISS

from utils import text_splitter, read_txt, read_all_txt_files, remove_accents
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
data = read_all_txt_files('data/txt_file')
# for file_name, content in data.items():
#     print(f"File: {file_name}\nContent:\n{content}\n")
# print(remove_accents('đ á ớ ô â ừ ị ự'))
data_all = ''
for file_name, content in data.items():
    # print(len(content))
    data_all = data_all+content
    print(remove_accents(file_name))
    if len(content) > 7000:
        chunks = text_splitter.split_text(content)
        db = FAISS.from_texts(chunks, embeddings)
        file_name = remove_accents(file_name)
        pathdb = f"FAISS_db/{file_name}"
        db.save_local(pathdb)
print(data_all)
chunks_all = text_splitter.split_text(data_all)
db_all = FAISS.from_texts(chunks_all, embeddings)
pathdb = f"FAISS_db/db_all"
db_all.save_local(pathdb)






















# data_banla = read_txt()
# data_binhnuoc = read_txt()
# data_beptu = read_txt()
# data_camera = read_txt()
# data_congtac = read_txt()
# data_massage = read_txt()
# data_lovisong = read_txt()
# data_maygiat = read_txt()
# data_maylockk = read_txt()
# data_maylocnuoc = read_txt()
# data_mayxay = read_txt()
# data_noichien = read_txt()
# data_noicom = read_txt()
# data_noiapsuat = read_txt()
# data_robot = read_txt()
# data_wifi = read_txt()
# data_dieuhoa = read_txt()
# data_denmattroi = read_txt()