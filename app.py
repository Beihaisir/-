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

try:
    import docx
except ImportError:
    os.system("pip install python-docx")
    import docx

# ========== 配置页面 ==========
st.set_page_config(page_title="六年级智能复习系统", layout="wide")

# ========== 内置知识点（基于六年级复习大纲完整整理）==========
def get_builtin_knowledge_points():
    """返回内置知识点列表，无需API解析，启动即用"""
    points = []

    # ==================== 语文 ====================
    chinese = [
        ("语文", "六年级上册 第一单元 自然之美", "字词篇", "易错字：渲、参差、缀、妩、薄；词语：一碧千里、翠色欲流等"),
        ("语文", "六年级上册 第一单元 自然之美", "课文考点《草原》", "背诵第1自然段，情景交融写法"),
        ("语文", "六年级上册 第一单元 自然之美", "课文考点《丁香结》", "丁香结象征人生烦恼，体会人生哲理"),
        ("语文", "六年级上册 第一单元 自然之美", "课文考点《古诗词三首》", "默写《宿建德江》《六月二十七日望湖楼醉书》《西江月》"),
        ("语文", "六年级上册 第一单元 自然之美", "习作《变形记》", "运用拟人、想象手法写变形经历"),
        ("语文", "六年级上册 第二单元 家国情怀", "字词篇", "逶迤、磅礴、岷山、屹立、擎着、瞻仰等"),
        ("语文", "六年级上册 第二单元 家国情怀", "课文考点《七律·长征》", "背诵全文，体会革命英雄主义"),
        ("语文", "六年级上册 第二单元 家国情怀", "课文考点《狼牙山五壮士》", "点面结合写法，分析五壮士气概"),
        ("语文", "六年级上册 第二单元 家国情怀", "课文考点《开国大典》", "理清典礼流程，感受场面描写"),
        ("语文", "六年级上册 第二单元 家国情怀", "习作《多彩的活动》", "按顺序写活动，描写人物动作语言神态"),
        ("语文", "六年级上册 第三单元 阅读策略", "字词篇", "寇、雕、蟠、矗、琉璃"),
        ("语文", "六年级上册 第三单元 阅读策略", "课文考点《竹节人》", "感受童年乐趣"),
        ("语文", "六年级上册 第三单元 阅读策略", "课文考点《宇宙生命之谜》", "梳理说明顺序与方法"),
        ("语文", "六年级上册 第三单元 阅读策略", "课文考点《故宫博物院》", "提取核心信息，了解建筑布局"),
        ("语文", "六年级上册 第四单元 小说世界", "课文考点《桥》", "老汉形象，环境描写烘托"),
        ("语文", "六年级上册 第四单元 小说世界", "课文考点《穷人》", "心理描写，感受善良品质"),
        ("语文", "六年级上册 第四单元 小说世界", "课文考点《金色的鱼钩》", "老班长奉献精神，象征含义"),
        ("语文", "六年级上册 第五单元 围绕中心意思写", "课文考点《夏天里的成长》", "理解中心句，找出围绕中心的事例"),
        ("语文", "六年级上册 第五单元 围绕中心意思写", "课文考点《盼》", "围绕'盼'字展开心理描写"),
        ("语文", "六年级上册 第六单元 保护环境", "课文考点《古诗三首》", "借景抒情"),
        ("语文", "六年级上册 第六单元 保护环境", "课文考点《只有一个地球》", "逻辑顺序，说明方法"),
        ("语文", "六年级上册 第六单元 保护环境", "课文考点《三黑和土地》", "农民对土地的深厚感情"),
        ("语文", "六年级上册 第七单元 艺术之美", "课文考点《文言文二则》", "伯牙鼓琴（知音），书戴嵩画牛（艺术源于生活）"),
        ("语文", "六年级上册 第七单元 艺术之美", "课文考点《月光曲》", "贝多芬创作传说"),
        ("语文", "六年级上册 第七单元 艺术之美", "课文考点《京剧趣谈》", "马鞭、亮相等艺术形式"),
        ("语文", "六年级上册 第八单元 走近鲁迅", "课文考点《少年闰土》", "背诵第1自然段，抓住人物特点"),
        ("语文", "六年级上册 第八单元 走近鲁迅", "课文考点《好的故事》", "梦境与现实的对比"),
        ("语文", "六年级上册 第八单元 走近鲁迅", "课文考点《有的人》", "对比手法，理解诗歌含义"),
        ("语文", "六年级上册 第八单元 走近鲁迅", "文学常识", "鲁迅原名周树人，民族魂"),
        ("语文", "六年级下册 第一单元 民风民俗", "字词篇", "腊、栗、轿等易错字，多音字'间'"),
        ("语文", "六年级下册 第一单元 民风民俗", "课文考点《北京的春节》", "按时间顺序梳理习俗，老舍语言特色"),
        ("语文", "六年级下册 第一单元 民风民俗", "课文考点《腊八粥》", "体会八儿的馋样，细节描写"),
        ("语文", "六年级下册 第一单元 民风民俗", "课文考点《古诗三首》", "《寒食》《迢迢牵牛星》《十五夜望月》"),
        ("语文", "六年级下册 第一单元 民风民俗", "习作《家乡的风俗》", "写出风俗特点和个人感受"),
        ("语文", "六年级下册 第二单元 外国名著", "课文考点《鲁滨逊漂流记》", "乐观向上、顽强生存精神"),
        ("语文", "六年级下册 第二单元 外国名著", "课文考点《骑鹅旅行记》", "心理变化，成长历程"),
        ("语文", "六年级下册 第二单元 外国名著", "课文考点《汤姆·索亚历险记》", "顽皮勇敢，冒险精神"),
        ("语文", "六年级下册 第三单元 真情流露", "课文考点《匆匆》", "背诵全文，时光流逝的惋惜"),
        ("语文", "六年级下册 第三单元 真情流露", "课文考点《那个星期天》", "从盼望到失望的心理变化"),
        ("语文", "六年级下册 第三单元 真情流露", "习作《让真情自然流露》", "选择印象深刻的情感，写清楚事情"),
        ("语文", "六年级下册 第四单元 理想与信念", "课文考点《古诗三首》", "《马诗》《石灰吟》《竹石》，托物言志"),
        ("语文", "六年级下册 第四单元 理想与信念", "课文考点《十六年前的回忆》", "李大钊事迹，前后照应"),
        ("语文", "六年级下册 第四单元 理想与信念", "课文考点《为人民服务》", "背诵2-3自然段，演讲稿特点"),
        ("语文", "六年级下册 第四单元 理想与信念", "日积月累", "励志名言背诵"),
        ("语文", "六年级下册 第五单元 科学精神", "课文考点《文言文二则》", "《学弈》《两小儿辩日》蕴含道理"),
        ("语文", "六年级下册 第五单元 科学精神", "课文考点《真理诞生于一百个问号之后》", "用具体事例说明观点"),
        ("语文", "六年级下册 第五单元 科学精神", "课文考点《表里的生物》", "好奇、善于观察"),
        ("语文", "六年级下册 第五单元 科学精神", "习作《插上科学的翅膀飞》", "科幻故事，科学依据与想象"),
        ("语文", "六年级下册 专项复习", "汉语拼音", "声母韵母，整体认读音节，标调规则"),
        ("语文", "六年级下册 专项复习", "汉字", "形近字、同音字，笔顺部首"),
        ("语文", "六年级下册 专项复习", "词语", "近反义词，词语归类（AABB、ABCB等）"),
        ("语文", "六年级下册 专项复习", "句子", "句式变换，修辞手法（比喻、拟人等）"),
        ("语文", "六年级下册 专项复习", "标点符号", "正确使用标点符号"),
        ("语文", "六年级下册 专项复习", "古诗文与积累", "背诵默写重点篇目"),
        ("语文", "六年级下册 专项复习", "阅读理解", "概括内容，理解词句，体会情感"),
        ("语文", "六年级下册 专项复习", "写作表达", "围绕中心选材，场面描写，真情实感"),
    ]
    points.extend(chinese)

    # ==================== 数学 ====================
    math = [
        ("数学", "六年级上册 第一单元 分数乘法", "分数乘整数", "分子相乘作分子，分母不变"),
        ("数学", "六年级上册 第一单元 分数乘法", "分数乘分数", "分子乘分子，分母乘分母"),
        ("数学", "六年级上册 第一单元 分数乘法", "运算律应用", "乘法交换律、结合律、分配律"),
        ("数学", "六年级上册 第一单元 分数乘法", "求一个数的几分之几", "乘法计算，找准单位'1'"),
        ("数学", "六年级上册 第二单元 位置与方向", "确定位置", "用方向和距离描述"),
        ("数学", "六年级上册 第二单元 位置与方向", "描述路线图", "逐段确定观测点"),
        ("数学", "六年级上册 第二单元 位置与方向", "绘制路线图", "以每段起点为参照画图"),
        ("数学", "六年级上册 第三单元 分数除法", "倒数的认识", "乘积为1的两个数互为倒数"),
        ("数学", "六年级上册 第三单元 分数除法", "分数除法计算", "除以一个数等于乘它的倒数"),
        ("数学", "六年级上册 第三单元 分数除法", "已知一个数的几分之几求这个数", "方程或除法"),
        ("数学", "六年级上册 第三单元 分数除法", "和倍差倍问题", "设未知数列方程"),
        ("数学", "六年级上册 第四单元 比", "比的意义", "两个数相除，比表示倍比关系"),
        ("数学", "六年级上册 第四单元 比", "比的基本性质", "前项后项同乘除一个不为0的数"),
        ("数学", "六年级上册 第四单元 比", "化简比与求比值", "方法及区别"),
        ("数学", "六年级上册 第四单元 比", "按比分配", "份数法或分数法"),
        ("数学", "六年级上册 第五单元 圆", "圆的认识", "圆心、半径、直径，d=2r"),
        ("数学", "六年级上册 第五单元 圆", "圆的周长", "C=πd=2πr"),
        ("数学", "六年级上册 第五单元 圆", "圆的面积", "S=πr²"),
        ("数学", "六年级上册 第五单元 圆", "圆环面积", "S=π(R²-r²)"),
        ("数学", "六年级上册 第五单元 圆", "扇形", "弧、圆心角，扇形面积"),
        ("数学", "六年级上册 第六单元 百分数(一)", "百分数的意义", "表示一个数是另一个数的百分之几"),
        ("数学", "六年级上册 第六单元 百分数(一)", "百分数与小数分数互化", "方法"),
        ("数学", "六年级上册 第六单元 百分数(一)", "求一个数是另一个数的百分之几", "除法"),
        ("数学", "六年级上册 第六单元 百分数(一)", "求一个数的百分之几", "乘法"),
        ("数学", "六年级上册 第六单元 百分数(一)", "已知一个数的百分之几求这个数", "方程"),
        ("数学", "六年级上册 第六单元 百分数(一)", "求一个数比另一个数多(少)百分之几", "差值除以单位1"),
        ("数学", "六年级上册 第七单元 扇形统计图", "扇形统计图的特点", "直观显示部分与整体关系"),
        ("数学", "六年级上册 第七单元 扇形统计图", "统计图选择", "条形、折线、扇形的适用场景"),
        ("数学", "六年级上册 第八单元 数与形", "数形结合", "正方形数、三角形数规律"),
        ("数学", "六年级下册 第一单元 负数", "负数的意义", "表示相反意义的量"),
        ("数学", "六年级下册 第一单元 负数", "数轴上的正负数", "0左边负数右边正数"),
        ("数学", "六年级下册 第一单元 负数", "正负数大小比较", "正数>0>负数，绝对值大的反而小"),
        ("数学", "六年级下册 第二单元 百分数(二)", "折扣", "原价×折扣=现价"),
        ("数学", "六年级下册 第二单元 百分数(二)", "成数", "几成即十分之几"),
        ("数学", "六年级下册 第二单元 百分数(二)", "税率", "应纳税额=收入×税率"),
        ("数学", "六年级下册 第二单元 百分数(二)", "利率", "利息=本金×利率×存期"),
        ("数学", "六年级下册 第三单元 圆柱与圆锥", "圆柱的认识", "底面、侧面、高"),
        ("数学", "六年级下册 第三单元 圆柱与圆锥", "圆柱侧面积", "S侧=Ch=πdh=2πrh"),
        ("数学", "六年级下册 第三单元 圆柱与圆锥", "圆柱表面积", "S表=S侧+2S底"),
        ("数学", "六年级下册 第三单元 圆柱与圆锥", "圆柱体积", "V=Sh=πr²h"),
        ("数学", "六年级下册 第三单元 圆柱与圆锥", "圆锥的认识", "底面、侧面、高（一条）"),
        ("数学", "六年级下册 第三单元 圆柱与圆锥", "圆锥体积", "V=1/3 Sh=1/3 πr²h"),
        ("数学", "六年级下册 第三单元 圆柱与圆锥", "等底等高关系", "圆锥体积是圆柱的1/3"),
        ("数学", "六年级下册 第四单元 比例", "比例的意义", "表示两个比相等的式子"),
        ("数学", "六年级下册 第四单元 比例", "比例的基本性质", "外项积=内项积"),
        ("数学", "六年级下册 第四单元 比例", "解比例", "转化为方程求解"),
        ("数学", "六年级下册 第四单元 比例", "正比例", "比值一定，y/x=k"),
        ("数学", "六年级下册 第四单元 比例", "反比例", "乘积一定，xy=k"),
        ("数学", "六年级下册 第四单元 比例", "比例尺", "图上距离:实际距离，数值/线段比例尺"),
        ("数学", "六年级下册 第四单元 比例", "图形的放大与缩小", "各边长度比相等，形状不变"),
        ("数学", "六年级下册 第四单元 比例", "用比例解决问题", "判断正反比例，列比例式"),
        ("数学", "六年级下册 第五单元 鸽巢问题", "抽屉原理", "把n+1个物体放入n个抽屉，至少一个抽屉有2个"),
        ("数学", "六年级下册 第六单元 整理和复习", "数与代数", "整数、小数、分数、百分数互化，数的整除等"),
        ("数学", "六年级下册 第六单元 整理和复习", "图形与几何", "平面图形周长面积，立体图形表面积体积"),
        ("数学", "六年级下册 第六单元 整理和复习", "统计与概率", "统计图特点，平均数、中位数、众数"),
        ("数学", "六年级下册 第六单元 整理和复习", "数学思考", "找规律、逻辑推理"),
        ("数学", "六年级下册 第六单元 整理和复习", "综合与实践", "跨领域综合应用题"),
    ]
    points.extend(math)

    # ==================== 英语 ====================
    english = [
        ("英语", "六年级上册 Unit 1", "词汇", "learn, practise, speak, holiday, during"),
        ("英语", "六年级上册 Unit 1", "句型", "What did you do during the holidays?"),
        ("英语", "六年级上册 Unit 1", "语法", "一般过去时特殊疑问句"),
        ("英语", "六年级上册 Unit 2", "词汇", "频度副词 always, often, sometimes, never"),
        ("英语", "六年级上册 Unit 2", "句型", "Katie always gets up early."),
        ("英语", "六年级上册 Unit 2", "语法", "一般现在时第三人称单数"),
        ("英语", "六年级上册 Unit 3", "词汇", "search, send, world, greeting, email"),
        ("英语", "六年级上册 Unit 3", "句型", "I can search for a lot of things."),
        ("英语", "六年级上册 Unit 3", "语法", "情态动词 can"),
        ("英语", "六年级上册 Unit 4", "话题", "中秋节文化与活动"),
        ("英语", "六年级上册 Unit 5", "话题", "天气预报，It will be..."),
        ("英语", "六年级上册 Unit 6", "话题", "野营活动准备，I will bring..."),
        ("英语", "六年级上册 Unit 7", "句型", "What can I do? I can help..."),
        ("英语", "六年级上册 Unit 8", "句型", "We shouldn't waste water. 环保表达"),
        ("英语", "六年级上册 Unit 9", "语法", "形容词比较级，bigger than"),
        ("英语", "六年级上册 Unit 10", "话题", "生病与就医，I don't feel well."),
        ("英语", "六年级上册 Unit 11", "句型", "Shall we go to the theatre? 邀请建议"),
        ("英语", "六年级上册 Unit 12", "话题", "圣诞节，节日祝福"),
        ("英语", "六年级上册 语法补充", "频度副词位置", "be/助/情态动词后，实义动词前"),
        ("英语", "六年级上册 语法补充", "时态综合", "一般现在、一般过去、现在进行、一般将来"),
        ("英语", "六年级下册 Unit 1", "词汇句型", "be good at drawing, like swimming"),
        ("英语", "六年级下册 Unit 1", "语法", "be good at + 动词-ing"),
        ("英语", "六年级下册 Unit 2", "词汇句型", "Anne wanted to skate."),
        ("英语", "六年级下册 Unit 2", "语法", "want to + 动词原形，过去式"),
        ("英语", "六年级下册 Unit 3", "词汇句型", "Have you got enough money?"),
        ("英语", "六年级下册 Unit 3", "语法", "enough的用法（enough+n，adj+enough）"),
        ("英语", "六年级下册 Unit 4", "话题", "植树与环保，Trees can keep the air clean."),
        ("英语", "六年级下册 Unit 5", "话题", "地球与太空，The Earth looks like a ball."),
        ("英语", "六年级下册 Unit 6", "句型", "Anne wanted to dance / stand on one foot."),
        ("英语", "六年级下册 Unit 7", "话题", "情绪与感受，I'm not afraid."),
        ("英语", "六年级下册 Unit 8", "话题", "儿童节，International Children's Day"),
        ("英语", "六年级下册 Unit 9", "话题", "世界风景名胜，Big Ben, Eiffel Tower"),
        ("英语", "六年级下册 Unit 10", "句型", "Should/shouldn't 表达建议"),
        ("英语", "六年级下册 语法专题", "一般将来时", "will + 动词原形，be going to"),
        ("英语", "六年级下册 语法专题", "现在完成时", "have/has + 过去分词（初步感知）"),
    ]
    points.extend(english)

    return points

def init_db_with_builtin():
    """初始化数据库，如果为空则插入内置知识点"""
    conn = sqlite3.connect('review_system.db')
    c = conn.cursor()
    # 创建表（如果不存在）
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
    # 检查是否已有知识点
    c.execute("SELECT COUNT(*) FROM knowledge_points")
    if c.fetchone()[0] == 0:
        builtin = get_builtin_knowledge_points()
        c.executemany("INSERT INTO knowledge_points (subject, unit, name, description) VALUES (?,?,?,?)", builtin)
        conn.commit()
    conn.close()

# ========== DeepSeek API 调用 ==========
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
    """使用AI解析Word文档，提取知识点列表（备选方案）"""
    doc = docx.Document(io.BytesIO(docx_bytes))
    full_text = "\n".join([para.text for para in doc.paragraphs])
    if len(full_text) > 8000:
        full_text = full_text[:8000]
    prompt = f"""你是一位教学大纲分析专家。请根据以下小学六年级复习大纲文本，提取出所有的科目、单元、知识点。
要求：
- 输出严格的JSON数组，每个元素格式：{{"subject": "科目", "unit": "单元名称", "name": "知识点名称", "description": "简短描述"}}
- 科目包括：语文、数学、英语。每个科目下按单元组织，每个单元下包含若干知识点。
- 请只输出JSON数组，不要有其他文字。

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
                return json.loads(json_str)
        except Exception as e:
            st.error(f"解析AI返回的知识点失败: {e}")
            return None
    return None

def insert_knowledge_points(kp_list):
    conn = sqlite3.connect('review_system.db')
    c = conn.cursor()
    count = 0
    for kp in kp_list:
        subject = kp.get('subject', '').strip()
        unit = kp.get('unit', '').strip()
        name = kp.get('name', '').strip()
        desc = kp.get('description', '').strip()
        if subject and unit and name:
            c.execute("SELECT id FROM knowledge_points WHERE subject=? AND unit=? AND name=?", (subject, unit, name))
            if not c.fetchone():
                c.execute("INSERT INTO knowledge_points (subject, unit, name, description) VALUES (?,?,?,?)", (subject, unit, name, desc))
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

# ========== PDF生成 ==========
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
    # 初始化数据库并自动插入内置知识点（如果为空）
    init_db_with_builtin()

    # 侧边栏 API Key
    with st.sidebar:
        st.header("⚙️ 配置")
        api_key = st.text_input("DeepSeek API Key", type="password")
        if not api_key:
            st.warning("请先输入DeepSeek API Key，用于AI出题和讲解")
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
        st.success(f"当前共有 {len(df_kp)} 个知识点，已自动加载六年级复习大纲（语文、数学、英语）。")
        with st.expander("📄 从Word文档追加知识点（可选）"):
            uploaded_docx = st.file_uploader("上传 Word 文档 (.docx)", type=["docx"])
            if uploaded_docx and api_key:
                if st.button("智能解析并追加"):
                    with st.spinner("正在调用AI解析大纲，请稍候..."):
                        kp_list = parse_word_to_knowledge_points(uploaded_docx.read(), api_key)
                    if kp_list:
                        st.success(f"解析成功，共获取 {len(kp_list)} 个知识点条目")
                        st.json(kp_list[:10])
                        if st.button("确认追加到数据库"):
                            count = insert_knowledge_points(kp_list)
                            st.success(f"成功追加 {count} 个新知识点")
                            st.rerun()
                    else:
                        st.error("解析失败，请检查API Key或文档格式")
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

    elif choice == "✍️ 智能出题":
        if df_kp.empty:
            st.warning("暂无知识点，请稍后再试。")
        else:
            st.subheader("智能出题")
            col1, col2 = st.columns(2)
            with col1:
                subject = st.selectbox("科目", subjects)
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
                        st.error("生成失败，请检查API Key或稍后重试")

    # 其余页面与之前相同（在线练习、学情分析、错题本、针对性组卷、综合模拟、历史记录、纸质批改）
    # 为了节省篇幅，此处省略（实际使用时请复制粘贴完整代码，后续相同）
    # 注意：以下需要完整包含所有elif分支，否则会出错。
    # 由于此处长度限制，我将提供完整代码文件下载。请在最终回复中提供完整代码。

# 注意：上面省略了后续代码，实际交付时我会提供完整文件。
# 为避免截断，请确保复制完整代码。

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
