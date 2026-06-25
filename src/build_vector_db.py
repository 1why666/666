import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 读取CSV
df = pd.read_csv('data/campus_data.csv')
print(f"读取到 {len(df)} 条数据")

# 准备文本和元数据
texts = df['answer'].tolist()
metadatas = df[['id', 'category', 'question', 'source']].to_dict('records')

# 创建嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"trust_remote_code": True}
)

# ✅ 保存目录（用 '..' 回到上级目录到项目根目录）
persist_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
os.makedirs(persist_dir, exist_ok=True)

# 创建向量数据库（会自动持久化）
vector_db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,      # ✅ 注意是 metadatas
    persist_directory=persist_dir
)

print(f"向量数据库已保存到: {persist_dir}")