import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import re
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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

# ------------------- 自定义 CSS -------------------
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .main-title h1 {
        color: white !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.3rem !important;
    }
    .main-title p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem !important;
        margin-bottom: 0 !important;
    }
    
    /* 快捷功能卡片 */
    .quick-cards {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
        justify-content: center;
    }
    .quick-card {
        background: white;
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e8ecf1;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        flex: 1;
        min-width: 120px;
        text-align: center;
    }
    .quick-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    .quick-card .icon {
        font-size: 1.8rem;
        display: block;
        margin-bottom: 0.3rem;
    }
    .quick-card .label {
        font-size: 0.85rem;
        color: #333;
        font-weight: 500;
    }
    
    /* 聊天消息样式 */
    .chat-message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 85%;
        margin-left: auto;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }
    .chat-message-assistant {
        background: white;
        color: #1a1a2e !important;
        padding: 14px 20px;
        border-radius: 18px 18px 18px 4px;
        max-width: 85%;
        margin-right: auto;
        margin-bottom: 0.8rem;
        border: 1px solid #e8ecf1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        line-height: 1.7;
    }
    .chat-message-assistant strong {
        color: #667eea;
    }
    
    /* 输入框美化 */
    .stTextInput > div > div > input {
        border-radius: 25px !important;
        border: 2px solid #e8ecf1 !important;
        padding: 12px 20px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }
    
    /* 侧边栏信息 */
    .info-box {
        background: #f8f9fc;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin-top: 1rem;
    }
    .info-box .label {
        color: #666;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .info-box .value {
        color: #1a1a2e;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    
    /* 空状态 */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #999;
    }
    .empty-state .icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .empty-state .title {
        font-size: 1.2rem;
        color: #333;
        font-weight: 500;
    }
    .empty-state .subtitle {
        font-size: 0.95rem;
        color: #999;
        margin-top: 0.3rem;
    }
    
    /* 响应式 */
    @media (max-width: 600px) {
        .main-title h1 { font-size: 2rem !important; }
        .quick-card { min-width: 80px; padding: 0.5rem 1rem; }
        .quick-card .icon { font-size: 1.4rem; }
        .quick-card .label { font-size: 0.75rem; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------- 缓存资源 -------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True}
    )

@st.cache_resource
def load_vector_db():
    embeddings = load_embeddings()
    # ✅ 修改：使用 data/chroma_db 路径
    persist_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

embeddings = load_embeddings()
vector_db = load_vector_db()

# ------------------- Spark X API 配置 -------------------
SPARK_APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
SPARK_HTTP_URL = os.getenv("SPARK_HTTP_URL", "https://spark-api-open.xf-yun.com/x2/chat/completions")
SPARK_MODEL = os.getenv("SPARK_MODEL", "spark-x")

if not SPARK_APIPASSWORD:
    st.error("❌ 请在 .env 文件中设置 SPARK_APIPASSWORD")
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
                return f"⚠️ API 返回格式异常"
        else:
            return f"❌ API 错误：{resp.status_code}"
    except requests.exceptions.Timeout:
        return "⚠️ 请求超时，请稍后重试"
    except Exception as e:
        return f"⚠️ 请求异常：{e}"

# ------------------- RAG 问答 -------------------
def rag_retrieve_answer(question):
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    prompt_text = RAG_PROMPT.format(context=context, question=question)
    return call_spark_api(prompt_text)

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
            # 自动填充问题
            st.session_state.prompt_input = q['label']
st.markdown('</div>', unsafe_allow_html=True)

# 侧边栏信息
with st.sidebar:
    st.markdown("### 🎓 校园助手")
    st.markdown("---")
    
    # 当前时间
    st.markdown(f"""
    <div class="info-box">
        <div class="label">📅 当前时间</div>
        <div class="value">{datetime.now().strftime('%Y年%m月%d日 %H:%M')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 校历信息
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
# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 你好！我是校园生活百事通助手，有什么可以帮助你的吗？\n\n💡 试试问我：怎么请病假、现在是第几周、怎么算绩点..."}
    ]
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""

# 显示聊天历史
for idx, msg in enumerate(st.session_state.messages):
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
        # 清空快捷输入
        if st.session_state.prompt_input:
            st.session_state.prompt_input = ""
    with col2:
        st.write("")
        st.write("")
        send_btn = st.button("📤 发送", use_container_width=True, type="primary")

# 处理发送
if send_btn and prompt:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 获取回复
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