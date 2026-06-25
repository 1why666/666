import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import re
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa

load_dotenv()

# ------------------- 页面配置 -------------------
st.set_page_config(
    page_title="校园百事通", 
    page_icon="🏫",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------- 缓存资源 -------------------
@st.cache_resource
def load_embeddings():
    """加载嵌入模型，带错误处理"""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh",
            model_kwargs={"trust_remote_code": True}
        )
    except ImportError as e:
        st.error("❌ 缺少 sentence-transformers 依赖，请安装：pip install sentence-transformers")
        st.stop()
    except Exception as e:
        st.error(f"❌ 加载模型失败：{e}")
        st.stop()

@st.cache_resource
def load_vector_db():
    """加载向量数据库"""
    try:
        embeddings = load_embeddings()
        persist_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
        
        if not os.path.exists(persist_dir):
            st.error(f"❌ 向量数据库不存在：{persist_dir}")
            st.info("请确保 data/chroma_db 目录已上传到仓库")
            st.stop()
            
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    except Exception as e:
        st.error(f"❌ 加载向量数据库失败：{e}")
        st.stop()

# 加载资源
embeddings = load_embeddings()
vector_db = load_vector_db()

# ------------------- Spark X API 配置 -------------------
SPARK_APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
SPARK_HTTP_URL = os.getenv("SPARK_HTTP_URL", "https://spark-api-open.xf-yun.com/x2/chat/completions")
SPARK_MODEL = os.getenv("SPARK_MODEL", "spark-x")

if not SPARK_APIPASSWORD:
    st.error("❌ 请在 .env 文件中设置 SPARK_APIPASSWORD，或在 Streamlit Cloud Secrets 中配置")
    st.stop()

# ------------------- 调用 Spark X HTTP API -------------------
def call_spark_api(user_message):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SPARK_APIPASSWORD}"
    }
    payload = {
        "model": SPARK_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个校园生活助手，请用中文回答，回答要清晰、简洁、有帮助。用友好的语气，适当使用emoji。"},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.3,
        "max_tokens": 2048
    }
    try:
        resp = requests.post(SPARK_HTTP_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            result = resp.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "⚠️ API 返回格式异常"
        else:
            return f"❌ API 错误：{resp.status_code}"
    except requests.exceptions.Timeout:
        return "⚠️ 请求超时，请稍后重试"
    except Exception as e:
        return f"⚠️ 请求异常：{e}"

# ------------------- RAG 问答 -------------------
def rag_retrieve_answer(question):
    try:
        docs = vector_db.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        prompt_text = RAG_PROMPT.format(context=context, question=question)
        return call_spark_api(prompt_text)
    except Exception as e:
        return f"⚠️ 检索失败：{e}"

# ------------------- 智能体路由 -------------------
def agent_answer(question):
    if re.search(r'第.*周|校历|本周|几周', question):
        return get_current_week()
    if re.search(r'绩点|GPA|平均分|分数', question):
        nums = re.findall(r'\d+', question)
        if nums:
            return calculate_gpa(','.join(nums))
        else:
            return "📊 请提供您的各科分数，例如：85,90,78"
    return rag_retrieve_answer(question)

# ------------------- 快捷问题 -------------------
quick_questions = [
    {"icon": "🏥", "label": "怎么请病假"},
    {"icon": "📅", "label": "现在是第几周"},
    {"icon": "📊", "label": "怎么算绩点"},
    {"icon": "📚", "label": "图书馆开放时间"},
]

# ------------------- UI 主界面 -------------------
# 顶部标题
st.markdown("""
<div class="main-title">
    <h1>🏫 校园生活百事通</h1>
    <p>💡 我可以回答校园问题 · 查询校历周数 · 计算绩点</p>
</div>
""", unsafe_allow_html=True)

# 快捷问题卡片
st.markdown('<div class="quick-cards">', unsafe_allow_html=True)
cols = st.columns(len(quick_questions))
for idx, q in enumerate(quick_questions):
    with cols[idx]:
        if st.button(
            f"{q['icon']}\n{q['label']}", 
            key=f"quick_{idx}",
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.prompt_input = q['label']
st.markdown('</div>', unsafe_allow_html=True)

# 侧边栏信息
with st.sidebar:
    st.markdown("### 🎓 校园助手")
    st.markdown("---")
    
    st.markdown(f"""
    <div class="info-box">
        <div class="label">📅 当前时间</div>
        <div class="value">{datetime.now().strftime('%Y年%m月%d日 %H:%M')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    week_info = get_current_week()
    st.markdown(f"""
    <div class="info-box">
        <div class="label">📆 校历信息</div>
        <div class="value">{week_info}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.85rem; color:#888; text-align:center;">
        🤖 基于 Spark X 大模型<br>
        数据来源于校园知识库
    </div>
    """, unsafe_allow_html=True)

# ------------------- 聊天界面 -------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 你好！我是校园生活百事通助手，有什么可以帮助你的吗？\n\n💡 试试问我：怎么请病假、现在是第几周、怎么算绩点..."}
    ]
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message-user">🧑‍🎓 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

# 输入框
with st.container():
    col1, col2 = st.columns([5, 1])
    with col1:
        prompt = st.text_input(
            "请输入你的校园问题...",
            value=st.session_state.prompt_input,
            key="chat_input",
            placeholder="💬 输入你的问题，比如：怎么请病假",
            label_visibility="collapsed"
        )
        if st.session_state.prompt_input:
            st.session_state.prompt_input = ""
    with col2:
        st.write("")
        st.write("")
        send_btn = st.button("📤 发送", use_container_width=True, type="primary")

if send_btn and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🤔 思考中..."):
        answer = agent_answer(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# 底部清空按钮
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 你好！我是校园生活百事通助手，有什么可以帮助你的吗？\n\n💡 试试问我：怎么请病假、现在是第几周、怎么算绩点..."}
        ]
        st.rerun()

# 底部版权
st.markdown("""
<div style="text-align:center; color:#bbb; font-size:0.75rem; padding:2rem 0 0.5rem 0; border-top:1px solid #f0f0f0; margin-top:2rem;">
    © 2026 校园生活百事通 · 用 ❤️ 打造
</div>
""", unsafe_allow_html=True)
