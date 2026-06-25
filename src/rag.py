# 第一行就配置镜像，只导入一次 os
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT

load_dotenv()

# ✅ 修改：使用正确的路径
persist_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')

# 初始化嵌入和向量库
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"trust_remote_code": True}
)
vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# 注意：将下面的 APIPASSWORD 替换为您从控制台复制的完整密码
APIPASSWORD = "qxeawQTjRWNyiAmOUOjd:rNPPXlMxWClIzRtZtHkz"   # 粘贴完整密码

def rag_answer(question):
    # 1. 向量检索
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    
    # 2. 拼接提示词
    prompt_text = RAG_PROMPT.format(context=context, question=question)
    
    # 3. 调用 HTTP 接口（讯飞 Spark X）
    url = "https://spark-api-open.xf-yun.com/x2/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {APIPASSWORD}"
    }
    payload = {
        "model": "spark-x",          # 改为 spark-x（或 generalx2）
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"HTTP 请求失败（{response.status_code}）：{response.text}"
    except Exception as e:
        return f"请求异常：{e}"

# 测试
if __name__ == "__main__":
    print(rag_answer("怎么请病假？"))