import streamlit as st
import json
import sqlite3
import datetime
import pandas as pd
import requests
import io
import os
from fpdf import FPDF
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 用于读取 .docx
try:
    import docx
except ImportError:
    os.system("pip install python-docx")
    import docx

# ========== 配置页面 ==========
st.set_page_config(page_title="六年级智能复习系统", layout="wide")

# ========== 初始化数据库 ==========
def init_db():
    conn = sqlite3.connect('review_system.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge_points
                 (id INTEGER PRIMARY KEY, subject TEXT, unit TEXT, name TEXT, description TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS exercise_records
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER DEFAULT 1,
                  record_type TEXT,
                  title TEXT,
                  knowledge_points TEXT,
                  questions_snapshot TEXT,
                  user_answers TEXT,
                  score REAL,
                  total_score REAL,
                  start_time TIMESTAMP,
                  submit_time TIMESTAMP,
                  is_paper BOOLEAN DEFAULT 0,
                  parent_record_id INTEGER,
                  paper_images TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS study_records
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  question_id TEXT,
                  knowledge_point_id INTEGER,
                  is_correct BOOLEAN,
                  user_answer TEXT,
                  correct_answer TEXT,
                  exercise_record_id INTEGER,
                  timestamp TIMESTAMP)''')
    conn.commit()
    conn.close()

# ========== DeepSeek API ==========
def call_deepseek(prompt: str, api_key: str) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4000
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"DeepSeek API 调用失败: {e}")
        return None

def parse_word_to_knowledge_points(docx_bytes, api_key):
    """使用AI解析Word文档，提取知识点列表"""
    # 读取docx文本
    doc = docx.Document(io.BytesIO(docx_bytes))
    full_text = "\n".join([para.text for para in doc.paragraphs])
    # 限制长度（避免超出token）
    if len(full_text) > 8000:
        full_text = full_text[:8000]
    
    prompt = f"""你是一位教学大纲分析专家。请根据以下小学六年级复习大纲文本，提取出所有的科目、单元、知识点。
要求：
- 输出严格的JSON数组，每个元素格式：{{"subject": "科目", "unit": "单元名称", "name": "知识点名称", "description": "简短描述"}}
- 科目包括：语文、数学、英语。每个科目下按单元组织，每个单元下包含若干知识点。
- 单元名称请使用大纲中的原始标题（如“六年级上册 第一单元 自然之美”或“六年级下册 第三单元 圆柱与圆锥”）。
- 知识点名称应具体，如“课文考点《草原》”、“分数乘整数”、“词汇与句型”等。
- description字段填写该知识点的核心内容提要（一句话）。
请只输出JSON数组，不要有其他文字。

大纲文本：
{full_text}
"""
    result = call_deepseek(prompt, api_key)
    if result:
        try:
            start = result.find('[')
            end = result.rfind(']') + 1
            if start != -1 and end != 0:
                json_str = result[start:end]
                kp_list = json.loads(json_str)
                return kp_list
        except Exception as e:
            st.error(f"解析AI返回的知识点失败: {e}")
            return None
    return None

def insert_knowledge_points(kp_list):
    """将知识点列表插入数据库（去重）"""
    conn = sqlite3.connect('review_system.db')
    c = conn.cursor()
    count = 0
    for kp in kp_list:
        subject = kp.get('subject', '').strip()
        unit = kp.get('unit', '').strip()
        name = kp.get('name', '').strip()
        desc = kp.get('description', '').strip()
        if subject and unit and name:
            # 避免重复插入
            c.execute("SELECT id FROM knowledge_points WHERE subject=? AND unit=? AND name=?", (subject, unit, name))
            if not c.fetchone():
                c.execute("INSERT INTO knowledge_points (subject, unit, name, description) VALUES (?,?,?,?)",
                          (subject, unit, name, desc))
                count += 1
    conn.commit()
    conn.close()
    return count

# ========== AI出题 ==========
def generate_questions(knowledge_points: List[str], num_questions: int, question_type: str, api_key: str) -> List[Dict]:
    prompt = f"""你是一位小学六年级出题专家。请根据以下知识点生成{num_questions}道{question_type}题。
知识点：{', '.join(knowledge_points)}
要求：
- 题型可以是单选、多选、判断、填空，难度适合六年级学生。
- 返回严格的JSON数组，每个元素格式：
  {{"type": "单选"/"多选"/"判断"/"填空",
    "question": "题干",
    "options": ["选项A", "选项B", ...],  // 判断和填空可省略
    "answer": "正确答案",  // 多选用逗号分隔
    "explanation": "详细讲解",
    "knowledge_point": "所属知识点"
  }}
请确保只输出JSON，不要有其他文字。
"""
    result = call_deepseek(prompt, api_key)
    if result:
        try:
            start = result.find('[')
            end = result.rfind(']') + 1
            if start != -1 and end != 0:
                json_str = result[start:end]
                questions = json.loads(json_str)
                return questions[:num_questions]
        except:
            st.error("解析AI返回的题目失败，请重试。")
    return []

def generate_explanation(question: Dict, api_key: str) -> str:
    prompt = f"请详细讲解以下题目：{json.dumps(question, ensure_ascii=False)}"
    result = call_deepseek(prompt, api_key)
    return result if result else "暂无讲解"

def grade_question(question: Dict, user_answer: str) -> bool:
    correct = question['answer'].strip()
    user = user_answer.strip()
    if question['type'] == '多选':
        correct_set = set(correct.replace('，', ',').split(','))
        user_set = set(user.replace('，', ',').split(','))
        return correct_set == user_set
    elif question['type'] == '判断':
        return user.lower() == correct.lower()
    else:
        return user == correct

# ========== PDF生成（支持中文，使用DejaVu）==========
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_font("DejaVu", "", 12)
    def header(self):
        self.set_font("DejaVu", "", 12)
        self.cell(0, 10, '六年级智能复习系统 - 练习题', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.cell(0, 10, f'第 {self.page_no()} 页', 0, 0, 'C')

def create_pdf(questions: List[Dict], title: str) -> bytes:
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, title, 0, 1)
    pdf.ln(5)
    for i, q in enumerate(questions, 1):
        pdf.multi_cell(0, 10, f"{i}. {q['question']}")
        if 'options' in q and q['options']:
            for opt in q['options']:
                pdf.multi_cell(0, 10, f"   {opt}")
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin1')

def create_report_pdf(df_mastery, weak_kps, wrong_df) -> bytes:
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("DejaVu", "", 14)
    pdf.cell(0, 10, '学情分析报告', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, f'生成时间：{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
    pdf.ln(5)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, '各知识点掌握度：', 0, 1)
    pdf.set_font("DejaVu", "", 10)
    col_width = pdf.w / 4
    pdf.cell(col_width, 10, '科目', 1)
    pdf.cell(col_width, 10, '单元', 1)
    pdf.cell(col_width, 10, '知识点', 1)
    pdf.cell(col_width, 10, '掌握度(%)', 1)
    pdf.ln()
    for _, row in df_mastery.iterrows():
        pdf.cell(col_width, 10, str(row['subject']), 1)
        pdf.cell(col_width, 10, str(row['unit']), 1)
        pdf.cell(col_width, 10, str(row['name']), 1)
        pdf.cell(col_width, 10, f"{row['mastery']:.1f}", 1)
        pdf.ln()
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, '薄弱知识点（掌握度<60%）：', 0, 1)
    if not weak_kps.empty:
        pdf.set_font("DejaVu", "", 10)
        for _, row in weak_kps.iterrows():
            pdf.cell(0, 10, f"{row['subject']}-{row['unit']}-{row['name']}: {row['mastery']:.1f}%", 0, 1)
    else:
        pdf.cell(0, 10, '无薄弱知识点，恭喜！', 0, 1)
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, '最近错题记录：', 0, 1)
    if not wrong_df.empty:
        pdf.set_font("DejaVu", "", 9)
        for i, row in wrong_df.head(20).iterrows():
            pdf.multi_cell(0, 8, f"题目：{row['question_id']}  知识点：{row['kp_name']}  你的答案：{row['user_answer']}  正确答案：{row['correct_answer']}")
            pdf.ln(2)
    else:
        pdf.cell(0, 10, '暂无错题，继续保持！', 0, 1)
    return pdf.output(dest='S').encode('latin1')

# ========== 存储练习记录 ==========
def save_exercise_record(record_type, title, kp_ids, questions, user_answers, score, total_score):
    conn = sqlite3.connect('review_system.db')
    c = conn.cursor()
    now = datetime.datetime.now().isoformat()
    c.execute('''INSERT INTO exercise_records 
                 (user_id, record_type, title, knowledge_points, questions_snapshot, user_answers, score, total_score, start_time, submit_time, is_paper)
                 VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)''',
              (record_type, title, json.dumps(kp_ids), json.dumps(questions), json.dumps(user_answers), score, total_score, now, now))
    record_id = c.lastrowid
    for i, q in enumerate(questions):
        is_correct = grade_question(q, user_answers[i]) if i < len(user_answers) else False
        kp_name = q.get('knowledge_point', '')
        c.execute("SELECT id FROM knowledge_points WHERE name=?", (kp_name,))
        kp = c.fetchone()
        kp_id = kp[0] if kp else None
        c.execute('''INSERT INTO study_records
                     (user_id, question_id, knowledge_point_id, is_correct, user_answer, correct_answer, exercise_record_id, timestamp)
                     VALUES (1, ?, ?, ?, ?, ?, ?, ?)''',
                  (f"q_{i}", kp_id, is_correct, user_answers[i], q['answer'], record_id, now))
    conn.commit()
    conn.close()
    return record_id

# ========== 学情分析数据 ==========
def get_knowledge_mastery():
    conn = sqlite3.connect('review_system.db')
    df = pd.read_sql_query('''
        SELECT kp.id, kp.subject, kp.unit, kp.name,
               AVG(CASE WHEN sr.is_correct THEN 100 ELSE 0 END) as mastery
        FROM knowledge_points kp
        LEFT JOIN study_records sr ON kp.id = sr.knowledge_point_id
        GROUP BY kp.id
    ''', conn)
    conn.close()
    return df

def get_wrong_questions():
    conn = sqlite3.connect('review_system.db')
    df = pd.read_sql_query('''
        SELECT sr.*, kp.subject, kp.unit, kp.name as kp_name
        FROM study_records sr
        JOIN knowledge_points kp ON sr.knowledge_point_id = kp.id
        WHERE sr.is_correct = 0
        ORDER BY sr.timestamp DESC
    ''', conn)
    conn.close()
    return df

# ========== 主函数 ==========
def main():
    st.title("📚 六年级智能复习系统")
    init_db()

    # 侧边栏 API Key
    with st.sidebar:
        st.header("⚙️ 配置")
        api_key = st.text_input("DeepSeek API Key", type="password")
        if not api_key:
            st.warning("请先输入DeepSeek API Key")
            st.stop()
        st.success("API Key 已设置")

    # 获取当前知识点
    conn = sqlite3.connect('review_system.db')
    df_kp = pd.read_sql_query("SELECT id, subject, unit, name FROM knowledge_points", conn)
    conn.close()
    subjects = df_kp['subject'].unique() if not df_kp.empty else []

    # 导航
    menu = ["📚 知识库", "✍️ 智能出题", "📝 在线练习", "📊 学情分析", "📓 错题本", "🎯 针对性组卷", "📄 综合模拟", "📜 历史记录", "📤 纸质批改"]
    choice = st.sidebar.radio("导航", menu)

    if choice == "📚 知识库":
        st.subheader("知识结构管理")
        st.dataframe(df_kp)
        
        # 新增：从Word文档导入（六年级复习大纲）
        with st.expander("📄 从六年级复习大纲.docx 导入知识点（推荐）"):
            uploaded_docx = st.file_uploader("上传 Word 文档 (.docx)", type=["docx"])
            if uploaded_docx and api_key:
                if st.button("智能解析并导入"):
                    with st.spinner("正在调用AI解析大纲，请稍候（约10-20秒）..."):
                        kp_list = parse_word_to_knowledge_points(uploaded_docx.read(), api_key)
                    if kp_list:
                        st.success(f"解析成功，共获取 {len(kp_list)} 个知识点条目")
                        st.json(kp_list[:10])  # 预览前10条
                        if st.button("确认导入数据库"):
                            count = insert_knowledge_points(kp_list)
                            st.success(f"成功导入 {count} 个新知识点")
                            st.rerun()
                    else:
                        st.error("解析失败，请检查API Key或文档格式")
            elif not api_key:
                st.info("请先在侧边栏输入DeepSeek API Key")

        with st.expander("手动添加知识点"):
            subject = st.text_input("科目")
            unit = st.text_input("单元")
            name = st.text_input("知识点名称")
            desc = st.text_area("描述")
            if st.button("添加"):
                if subject and unit and name:
                    conn = sqlite3.connect('review_system.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO knowledge_points (subject, unit, name, description) VALUES (?,?,?,?)", (subject, unit, name, desc))
                    conn.commit()
                    conn.close()
                    st.success("添加成功")
                    st.rerun()

        with st.expander("导入 CSV/JSON"):
            uploaded_file = st.file_uploader("上传文件", type=["csv","json"])
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    new_df = pd.read_csv(uploaded_file)
                else:
                    new_df = pd.read_json(uploaded_file)
                if all(col in new_df.columns for col in ['subject','unit','name']):
                    conn = sqlite3.connect('review_system.db')
                    c = conn.cursor()
                    for _, row in new_df.iterrows():
                        c.execute("INSERT OR IGNORE INTO knowledge_points (subject, unit, name, description) VALUES (?,?,?,?)",
                                  (row['subject'], row['unit'], row['name'], row.get('description','')))
                    conn.commit()
                    conn.close()
                    st.success("导入成功")
                    st.rerun()
                else:
                    st.error("文件缺少必要的列：subject, unit, name")

    elif choice == "✍️ 智能出题":
        if df_kp.empty:
            st.warning("暂无知识点，请先在「知识库」中导入（推荐上传六年级复习大纲.docx）")
        else:
            st.subheader("智能出题")
            col1, col2 = st.columns(2)
            with col1:
                subject = st.selectbox("科目", subjects) if len(subjects)>0 else st.selectbox("科目", [])
                if subject:
                    units = df_kp[df_kp['subject']==subject]['unit'].unique()
                    unit = st.selectbox("单元", units)
                    kp_options = df_kp[(df_kp['subject']==subject) & (df_kp['unit']==unit)]['name'].tolist()
                    selected_kps = st.multiselect("知识点", kp_options)
            with col2:
                question_type = st.selectbox("题目类型", ["练习题", "例题", "试卷"])
                num = st.number_input("题目数量", min_value=1, max_value=20, value=5)
                include_explanation = st.checkbox("包含讲解", value=True)
            if st.button("生成题目"):
                if not selected_kps:
                    st.warning("请选择知识点")
                else:
                    with st.spinner("AI生成中..."):
                        questions = generate_questions(selected_kps, num, question_type, api_key)
                    if questions:
                        st.session_state['current_questions'] = questions
                        st.session_state['current_kps'] = selected_kps
                        st.session_state['current_type'] = question_type
                        st.success(f"生成{len(questions)}道题目")
                        for i, q in enumerate(questions):
                            st.write(f"**{i+1}. {q['question']}**")
                            if 'options' in q:
                                for opt in q['options']:
                                    st.write(f"   {opt}")
                            if include_explanation and 'explanation' in q:
                                st.info(f"讲解：{q['explanation']}")
                        pdf_bytes = create_pdf(questions, f"{subject}_{unit}_练习")
                        st.download_button("📥 下载PDF（无答案）", data=pdf_bytes, file_name="exercises.pdf", mime="application/pdf")
                        # 保存空白记录用于纸质批改
                        conn = sqlite3.connect('review_system.db')
                        c = conn.cursor()
                        now = datetime.datetime.now().isoformat()
                        c.execute('''INSERT INTO exercise_records 
                                     (user_id, record_type, title, knowledge_points, questions_snapshot, user_answers, score, total_score, start_time, submit_time, is_paper)
                                     VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)''',
                                  (question_type, f"{subject}_{unit}_练习", json.dumps(selected_kps), json.dumps(questions), json.dumps([]), None, num, now, now))
                        record_id = c.lastrowid
                        conn.commit()
                        conn.close()
                        st.info(f"已生成练习记录ID: {record_id}，可进行纸质批改")
                    else:
                        st.error("生成失败")

    elif choice == "📝 在线练习":
        if 'current_questions' not in st.session_state or not st.session_state['current_questions']:
            st.warning("请先在「智能出题」中生成题目")
        else:
            questions = st.session_state['current_questions']
            user_answers = []
            st.subheader("在线练习")
            for i, q in enumerate(questions):
                st.markdown(f"**{i+1}. {q['question']}**")
                if q['type'] == '单选':
                    answer = st.radio(f"答案 {i+1}", q['options'], key=f"q_{i}")
                elif q['type'] == '多选':
                    answer = st.multiselect(f"答案 {i+1}（多选）", q['options'], key=f"q_{i}")
                    answer = ','.join(answer)
                elif q['type'] == '判断':
                    answer = st.radio(f"答案 {i+1}", ["正确", "错误"], key=f"q_{i}")
                else:
                    answer = st.text_input(f"答案 {i+1}", key=f"q_{i}")
                user_answers.append(answer)
                if st.button(f"查看讲解 {i+1}", key=f"exp_{i}"):
                    if 'explanation' in q:
                        st.info(q['explanation'])
                    else:
                        with st.spinner("AI生成讲解中..."):
                            exp = generate_explanation(q, api_key)
                            st.info(exp)
            if st.button("提交练习"):
                total = len(questions)
                correct = 0
                for i, q in enumerate(questions):
                    if grade_question(q, user_answers[i]):
                        correct += 1
                score = correct
                st.success(f"得分: {score}/{total}")
                save_exercise_record(st.session_state.get('current_type', '练习'), "在线练习", 
                                     st.session_state.get('current_kps', []), questions, user_answers, score, total)
                st.balloons()

    elif choice == "📊 学情分析":
        st.subheader("学情分析报告")
        df = get_knowledge_mastery()
        if df.empty:
            st.info("暂无学习数据，请先完成一些练习")
        else:
            st.dataframe(df)
            st.bar_chart(df.set_index('name')['mastery'])
            weak = df[df['mastery'] < 60]
            if not weak.empty:
                st.warning("薄弱知识点：")
                st.dataframe(weak[['subject', 'unit', 'name', 'mastery']])
            wrong_df = get_wrong_questions()
            report_pdf = create_report_pdf(df, weak, wrong_df)
            st.download_button("📄 导出学情报告PDF", data=report_pdf, file_name="learning_report.pdf", mime="application/pdf")

    elif choice == "📓 错题本":
        st.subheader("错题本")
        wrong_df = get_wrong_questions()
        if wrong_df.empty:
            st.info("暂无错题")
        else:
            st.dataframe(wrong_df)

    elif choice == "🎯 针对性组卷":
        df = get_knowledge_mastery()
        weak_kps = df[df['mastery'] < 60]['name'].tolist()
        if not weak_kps:
            st.info("没有薄弱知识点")
        else:
            st.subheader("针对性查缺补漏")
            st.write(f"薄弱知识点：{', '.join(weak_kps)}")
            num = st.number_input("题目数量", min_value=1, max_value=20, value=5)
            if st.button("生成针对性练习"):
                with st.spinner("AI生成中..."):
                    questions = generate_questions(weak_kps, num, "针对性练习", api_key)
                if questions:
                    st.session_state['current_questions'] = questions
                    st.session_state['current_kps'] = weak_kps
                    st.session_state['current_type'] = "针对性练习"
                    st.success("生成成功，请前往「在线练习」作答")
                else:
                    st.error("生成失败")

    elif choice == "📄 综合模拟":
        if df_kp.empty:
            st.warning("请先导入知识点")
        else:
            st.subheader("综合模拟试卷")
            subject = st.selectbox("科目", subjects) if len(subjects)>0 else st.selectbox("科目", [])
            num = st.number_input("题量", min_value=5, max_value=30, value=10)
            if st.button("生成模拟试卷"):
                all_kps = df_kp[df_kp['subject']==subject]['name'].tolist()
                if not all_kps:
                    st.warning("该科目无知识点")
                else:
                    with st.spinner("AI生成中..."):
                        questions = generate_questions(all_kps, num, "模拟试卷", api_key)
                    if questions:
                        st.session_state['current_questions'] = questions
                        st.session_state['current_kps'] = all_kps
                        st.session_state['current_type'] = "模拟试卷"
                        st.success("生成成功，请前往「在线练习」作答")
                        pdf_bytes = create_pdf(questions, f"{subject}_模拟试卷")
                        st.download_button("下载试卷PDF", pdf_bytes, "simulation.pdf")
                    else:
                        st.error("生成失败")

    elif choice == "📜 历史记录":
        st.subheader("练习历史记录")
        conn = sqlite3.connect('review_system.db')
        df_records = pd.read_sql_query("SELECT id, record_type, title, score, total_score, submit_time, is_paper FROM exercise_records ORDER BY submit_time DESC", conn)
        conn.close()
        st.dataframe(df_records)
        record_id = st.number_input("查看详情（输入ID）", min_value=1, step=1)
        if st.button("加载详情"):
            conn = sqlite3.connect('review_system.db')
            c = conn.cursor()
            c.execute("SELECT questions_snapshot, user_answers FROM exercise_records WHERE id=?", (record_id,))
            row = c.fetchone()
            conn.close()
            if row:
                questions = json.loads(row[0])
                user_answers = json.loads(row[1]) if row[1] else []
                for i, q in enumerate(questions):
                    st.write(f"**{i+1}. {q['question']}**")
                    if i < len(user_answers):
                        st.write(f"你的答案：{user_answers[i]}")
                        st.write(f"正确答案：{q['answer']}")
                    if st.button(f"讲解{i+1}", key=f"hist_exp_{i}"):
                        st.info(q.get('explanation', '暂无讲解'))
            else:
                st.error("记录不存在")

    elif choice == "📤 纸质批改":
        st.subheader("纸质作业批改")
        conn = sqlite3.connect('review_system.db')
        df_pending = pd.read_sql_query("SELECT id, title, submit_time FROM exercise_records WHERE is_paper=0 AND user_answers = '[]' ORDER BY submit_time DESC", conn)
        conn.close()
        if df_pending.empty:
            st.info("没有待批改的练习记录，请先智能出题并下载PDF")
        else:
            record_id = st.selectbox("选择练习记录", df_pending['id'].tolist(), format_func=lambda x: f"{x} - {df_pending[df_pending['id']==x]['title'].iloc[0]}")
            if st.button("开始批改"):
                conn = sqlite3.connect('review_system.db')
                c = conn.cursor()
                c.execute("SELECT questions_snapshot FROM exercise_records WHERE id=?", (record_id,))
                row = c.fetchone()
                conn.close()
                if row:
                    questions = json.loads(row[0])
                    st.write(f"共{len(questions)}道题目，请按纸质答案录入")
                    user_answers = []
                    for i, q in enumerate(questions):
                        st.markdown(f"**{i+1}. {q['question']}**")
                        if 'options' in q:
                            for opt in q['options']:
                                st.write(f"   {opt}")
                        ans = st.text_input(f"你的答案（手写内容）", key=f"paper_ans_{i}")
                        user_answers.append(ans)
                    if st.button("提交批改"):
                        total = len(questions)
                        correct = 0
                        for i, q in enumerate(questions):
                            if grade_question(q, user_answers[i]):
                                correct += 1
                        score = correct
                        # 保存批改记录
                        conn = sqlite3.connect('review_system.db')
                        c = conn.cursor()
                        now = datetime.datetime.now().isoformat()
                        c.execute('''INSERT INTO exercise_records 
                                     (user_id, record_type, title, knowledge_points, questions_snapshot, user_answers, score, total_score, start_time, submit_time, is_paper, parent_record_id)
                                     VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)''',
                                  ("纸质批改", f"批改-{record_id}", json.dumps([]), json.dumps(questions), json.dumps(user_answers), score, total, now, now, record_id))
                        new_id = c.lastrowid
                        for i, q in enumerate(questions):
                            is_correct = grade_question(q, user_answers[i])
                            kp_name = q.get('knowledge_point', '')
                            c.execute("SELECT id FROM knowledge_points WHERE name=?", (kp_name,))
                            kp_row = c.fetchone()
                            kp_id = kp_row[0] if kp_row else None
                            c.execute('''INSERT INTO study_records
                                         (user_id, question_id, knowledge_point_id, is_correct, user_answer, correct_answer, exercise_record_id, timestamp)
                                         VALUES (1, ?, ?, ?, ?, ?, ?, ?)''',
                                      (f"paper_{new_id}_{i}", kp_id, is_correct, user_answers[i], q['answer'], new_id, now))
                        conn.commit()
                        conn.close()
                        st.success(f"批改完成！得分 {score}/{total}")
                        st.balloons()
                else:
                    st.error("读取题目失败")

if __name__ == "__main__":
    main()
