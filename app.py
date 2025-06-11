"""
外交辞令分析系统 - Flask Web应用

本系统用于分析中国官方对不同国际事件的外交表态严重程度，
通过关键词提取和语义分析，量化评估外交表态的强度等级。

主要功能：
1. 从CSV文件加载新闻数据
2. 提取事件关键词和外交表态内容
3. 对表态内容进行严重程度分类
4. 提供基于关键词的事件分析
5. 通过Web界面展示分析结果
"""

from flask import Flask, render_template, request
import pandas as pd
import re
from typing import List, Dict, Tuple

# 初始化Flask应用
app = Flask(__name__)


# ====================== 数据加载与预处理函数 ======================
def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    加载并预处理CSV格式的新闻数据

    参数:
        filepath: CSV文件路径

    返回:
        pd.DataFrame: 处理后的DataFrame，包含日期、关键词、事件和外交表态列
        或 None: 如果加载失败

    处理步骤:
        1. 读取CSV文件
        2. 检查必要列是否存在
        3. 重命名列名
        4. 提取外交表态内容
        5. 提取事件关键词
    """
    try:
        # 读取CSV文件，使用utf-8编码
        df = pd.read_csv(filepath, encoding='utf-8')

        # 验证文件包含必要的列
        if 'A' not in df.columns or 'B' not in df.columns:
            raise ValueError("CSV文件格式不符合预期，缺少必要的列")

        # 重命名列名使其更易理解
        df = df.rename(columns={
            'A': 'date',  # 日期列
            'B': 'event',  # 事件描述列
            'C': 'content'  # 完整内容列
        })

        # 使用正则表达式从内容中提取中国官方立场部分
        # 匹配模式：从"中国官方立场："开始，到"新闻X"或结尾为止
        df['diplomatic_statement'] = df['content'].apply(
            lambda x: re.search(r'中国官方立场：(.*?)(新闻\d+|$)', str(x)).group(1).strip()
        )

        # 从事件描述中提取关键词
        df['keywords'] = df['event'].apply(lambda x: extract_keywords(str(x)))

        # 只返回需要的列
        return df[['date', 'keywords', 'event', 'diplomatic_statement']]

    except Exception as e:
        # 打印错误信息便于调试
        print(f"数据加载和预处理过程中出错: {str(e)}")
        return None


# ====================== 关键词提取函数 ======================
def extract_keywords(text: str) -> List[str]:
    """
    从事件文本中提取预定义的关键词

    参数:
        text: 原始事件文本

    返回:
        List[str]: 匹配到的关键词列表，如无匹配则返回['其他']

    算法说明:
        1. 定义关键词映射表，包含主关键词和同义词
        2. 将文本统一转为小写提高匹配率
        3. 遍历映射表，检查是否有任何同义词出现在文本中
        4. 返回所有匹配的主关键词
    """
    # 关键词映射表：主关键词 -> 同义词/相关词列表
    keyword_mapping = {
        '俄乌': ['俄乌', '乌克兰', '俄罗斯', '基辅'],
        '巴以': ['巴以', '巴勒斯坦', '以色列', '加沙'],
        '中美': ['中美', '中国', '美国', '特朗普'],
        '贸易': ['贸易', '关税', '制裁', '保护主义'],
        '军事': ['军事', '袭击', '空袭', '战斗机', 'F-16', 'F－16'],
        '科技': ['科技', '芯片', 'AI', '人工智能', '英伟达'],
        '环境': ['环境', '气候', '排放', '能源'],
        '经济': ['经济', '股市', '债务', '美联储'],
        '联合国': ['联合国', 'UN'],
        '人道主义': ['人道', '平民', '伤亡']
    }

    # 统一转为小写，使匹配不区分大小写
    text = str(text).lower()

    # 收集所有匹配的主关键词
    matched_groups = []
    for group_name, variants in keyword_mapping.items():
        # 检查是否有任何同义词出现在文本中
        if any(v.lower() in text for v in variants):
            matched_groups.append(group_name)

    # 如果没有匹配到任何关键词，则归类为"其他"
    return matched_groups if matched_groups else ['其他']


# ====================== 严重程度分类函数 ======================
def classify_severity(statement: str) -> Tuple[int, str]:
    """
    对外交表态进行严重程度分类

    参数:
        statement: 外交表态文本

    返回:
        Tuple[int, str]: (严重程度分数, 严重等级)

    分类标准:
        level1 (分数1): 温和表态 - 表示关注、希望、呼吁等
        level2 (分数5): 较强硬表态 - 谴责、反对、敦促等
        level3 (分数10): 最强硬表态 - 制裁、驱逐、捍卫等
    """
    # 严重程度关键词定义
    severity_keywords = {
        'level1': {
            'keywords': ['表示关注', '希望各方', '保持克制', '愿与各方', '呼吁', '主张'],
            'score': 1
        },
        'level2': {
            'keywords': ['强烈不满', '坚决反对', '严正交涉', '谴责', '敦促'],
            'score': 5
        },
        'level3': {
            'keywords': ['采取坚决措施', '制裁', '驱逐', '捍卫', '必将', '不容'],
            'score': 10
        }
    }

    max_score = 0  # 最高分数初始值
    level = 'level1'  # 默认等级为level1

    # 遍历所有严重等级
    for lvl, data in severity_keywords.items():
        # 检查当前等级的关键词是否出现在表态中
        for kw in data['keywords']:
            if kw in statement:
                # 如果当前等级的分数更高，则更新最高分数和等级
                if data['score'] > max_score:
                    max_score = data['score']
                    level = lvl
                break  # 找到一个关键词即可，跳出当前循环

    return max_score, level


# ====================== 事件严重程度计算函数 ======================
def calculate_event_severity(df: pd.DataFrame, event_keyword: str) -> Dict:
    """
    计算指定关键词相关事件的严重程度分析结果

    参数:
        df: 预处理后的数据集
        event_keyword: 要分析的事件关键词

    返回:
        Dict: 包含分析结果的字典，或错误信息字典

    处理流程:
        1. 精确匹配关键词
        2. 模糊匹配相关词(如果精确匹配无结果)
        3. 计算匹配到的表态的严重程度
        4. 统计各级别表态数量
        5. 返回汇总分析结果
    """
    try:
        # 1. 精确匹配：查找关键词列表中包含目标关键词的记录
        related_statements = df[
            df['keywords'].apply(lambda x: event_keyword in x if isinstance(x, list) else False)
        ].copy()

        # 2. 模糊匹配：定义关键词的同义词/相关词列表
        keyword_variants = {
            '俄乌': ['乌克兰', '俄罗斯', '俄乌冲突', '乌俄', '基辅', '莫斯科'],
            '巴以': ['巴勒斯坦', '以色列', '加沙', '约旦河西岸', '哈马斯', '内塔尼亚胡'],
            '中美': ['美国', '特朗普', '拜登', '中美关系', '美中', '华盛顿', '白宫'],
            '贸易': ['关税', '制裁', '贸易战', '贸易摩擦', '进出口', '保护主义'],
            '军事': ['袭击', '空袭', '战斗机', 'F-16', 'F－16', '军队', '军事行动', '战争'],
            '科技': ['科技', '芯片', 'AI', '人工智能', '英伟达', '半导体', '技术', '研发'],
            '环境': ['环境', '气候', '排放', '能源', '碳中和', '碳达峰', '污染', '生态'],
            '经济': ['经济', '股市', '债务', '美联储', 'GDP', '增长率', '通货膨胀'],
            '联合国': ['联合国', 'UN', '安理会', '秘书长'],
            '人道主义': ['人道', '平民', '伤亡', '难民', '救援', '援助', '救助']
        }.get(event_keyword, [event_keyword])  # 默认使用原关键词

        # 如果精确匹配没有结果，尝试模糊匹配
        if related_statements.empty:
            temp_dfs = []  # 临时存储匹配到的DataFrame

            # 遍历所有相关词变体
            for variant in keyword_variants:
                # 在事件描述中查找包含变体的记录
                variant_matches = df[df['event'].str.contains(variant, case=False, na=False)]
                if not variant_matches.empty:
                    temp_dfs.append(variant_matches)

            # 如果找到匹配记录，合并并去重
            if temp_dfs:
                related_statements = pd.concat(temp_dfs)
                # 去重时指定可哈希的列，避免对列表列进行哈希操作
                related_statements = related_statements.drop_duplicates(
                    subset=['date', 'event', 'diplomatic_statement'])

        # 3. 如果仍然没有匹配结果，返回错误信息
        if related_statements.empty:
            return {"error": f"未找到'{event_keyword}'相关事件的外交辞令，请尝试使用更通用的关键词如'军事'、'贸易'等"}

        # 4. 计算严重程度
        # 对每条外交表态进行分类
        severity_results = related_statements['diplomatic_statement'].apply(classify_severity)
        related_statements['severity_score'] = [result[0] for result in severity_results]  # 分数列
        related_statements['severity_level'] = [result[1] for result in severity_results]  # 等级列
        related_statements = related_statements.sort_values('date')  # 按日期排序

        # 计算最高严重程度
        max_severity_score = related_statements['severity_score'].max()
        max_severity_level = related_statements[
            related_statements['severity_score'] == max_severity_score]['severity_level'].iloc[0]
        max_severity_statement = related_statements[
            related_statements['severity_score'] == max_severity_score].iloc[0]['diplomatic_statement']

        # 统计各等级表态数量
        level_counts = {
            'level1': len(related_statements[related_statements['severity_level'] == 'level1']),
            'level2': len(related_statements[related_statements['severity_level'] == 'level2']),
            'level3': len(related_statements[related_statements['severity_level'] == 'level3'])
        }

        # 返回分析结果
        return {
            'event_keyword': event_keyword,  # 分析的关键词
            'total_statements': len(related_statements),  # 总表态数
            'average_severity': related_statements['severity_score'].mean(),  # 平均严重分数
            'max_severity_score': max_severity_score,  # 最高严重分数
            'max_severity_level': max_severity_level,  # 最高严重等级
            'max_severity_statement': max_severity_statement,  # 最严重的表态内容
            'level_counts': level_counts,  # 各等级表态数量
            'statements_data': related_statements[  # 详细表态数据
                ['date', 'severity_score', 'severity_level', 'diplomatic_statement']].to_dict('records')
        }

    except Exception as e:
        # 捕获并记录异常
        print(f"分析过程中出错: {str(e)}")
        return {"error": f"分析过程中发生错误: {str(e)}"}


# ====================== Flask路由 ======================
@app.route('/')
def index():
    """首页路由，显示搜索表单"""
    return render_template('index.html', show_results=False)


@app.route('/analyze', methods=['GET'])
def analyze():
    """
    分析路由，处理用户提交的关键词

    流程:
        1. 获取用户输入的关键词
        2. 验证输入有效性
        3. 加载数据
        4. 计算严重程度
        5. 返回结果或错误信息
    """
    # 获取用户输入的关键词并去除首尾空格
    keyword = request.args.get('keyword', '').strip()

    # 验证输入是否为空
    if not keyword:
        return render_template('index.html', show_results=False, error="请输入有效的事件关键词")

    # 加载数据
    df = load_and_preprocess_data('data/input.docx_converted.csv')
    if df is None:
        return render_template('index.html', show_results=False, error="无法加载数据")

    # 计算严重程度
    severity_result = calculate_event_severity(df, keyword)

    # 如果结果中包含错误信息，显示错误
    if 'error' in severity_result:
        return render_template('index.html', show_results=False, error=severity_result['error'])

    # 显示分析结果
    return render_template('index.html', show_results=True, result=severity_result)


# ====================== 主程序入口 ======================
if __name__ == '__main__':
    # 启动Flask开发服务器
    # debug=True 启用调试模式，自动重载代码变更
    app.run(debug=True)