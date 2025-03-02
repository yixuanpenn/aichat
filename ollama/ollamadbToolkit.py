from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseSequentialChain
from langchain_ollama import OllamaLLM
import re

class OllamaDBToolkit:
    def __init__(self):
        # 初始化 Ollama
        self.ollama = OllamaLLM(            
            model="deepseek-r1:70b",
            base_url="http://localhost:11434",
        )
        
        # 创建数据库连接
        self.db = SQLDatabase.from_uri(
            "mysql+pymysql://test:test1234@localhost/test"
        )
        
        # 获取数据库所有表的COMMENT
        self.table_comments = self.get_table_comments()
        print(f"数据库表描述：\n{self.table_comments}")
        
        # 创建一个更智能的SQL生成提示
        self.sql_prompt = """基于以下数据库表结构及其描述：
        {table_info}
        
        问题：{question}
        
        请按以下步骤分析和处理：
        1. 仔细阅读每个表的字段描述，理解字段的实际含义
        2. 分析问题中的关键词（如：销量、价格、种类等）
        3. 将问题关键词与表字段的描述进行匹配
        4. 选择描述最匹配的表和字段
        5. 生成使用这些表和字段的SQL查询
        
        要求：
        1. 只使用表结构中实际存在的表和字段
        2. 确保选择的字段描述与问题需求相符
        3. 直接返回单个SQL语句
        4. 不要包含任何解释或思考过程
        5. 如果找不到描述匹配的表或字段，返回"无法找到相关字段"
        6. 使用标准的SQL语法
        
        SQL查询：
        """
        
        # 创建更详细的结果解释提示
        self.result_prompt = """基于以下信息回答问题：
        原始问题：{question}
        使用的表和字段：{tables_and_fields}
        字段描述：{field_descriptions}
        查询结果：{results}
        
        要求：
        1. 只基于实际的查询结果来回答
        2. 确保回答与字段的实际含义相符
        3. 直接给出具体的答案，不要解释思考过程
        4. 用简洁的中文回答
        5. 如果结果包含具体数字，请在回答中包含这些数字
        """
        
        # 创建数据库链
        self.db_chain = SQLDatabaseSequentialChain.from_llm(
            llm=self.ollama,
            db=self.db,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def get_table_comments(self):
        """获取所有表的COMMENT信息"""
        query = """
        SELECT 
            CONCAT(TABLE_NAME, ': ', IFNULL(TABLE_COMMENT, '')) as table_info
        FROM 
            information_schema.TABLES 
        WHERE 
            TABLE_SCHEMA = 'test'
        """
        result = self.db.run(query)
        # 将结果转换为易读的格式
        if isinstance(result, list):
            return "\n".join([str(row['table_info']) for row in result])
        return str(result)
    
    def get_table_info(self, table_name: str):
        """获取指定表的详细信息"""
        query = """
        SELECT 
            CONCAT(
                COLUMN_NAME, ' (', 
                COLUMN_TYPE, ')', 
                CASE WHEN COLUMN_COMMENT != '' 
                    THEN CONCAT(' - ', COLUMN_COMMENT)
                    ELSE ''
                END
            ) as column_info
        FROM 
            information_schema.COLUMNS 
        WHERE 
            TABLE_SCHEMA = 'test' 
            AND TABLE_NAME = :table_name
        ORDER BY 
            ORDINAL_POSITION
        """
        result = self.db.run(query, parameters={"table_name": table_name})
        # 将结果转换为易读的格式
        if isinstance(result, list):
            return "\n".join([str(row['column_info']) for row in result])
        return str(result)

    def select_table(self, question: str):
        """根据问题和表描述选择合适的表"""
        prompt = f"""基于以下数据库表的描述：
        {self.table_comments}
        
        问题：{question}
        
        请分析问题并选择最合适的数据表。
        
        要求：
        1. 仔细分析每个表的描述
        2. 将问题与表的用途进行匹配
        3. 只返回一个最合适的表名，不要包含任何其他文字
        4. 如果没有合适的表，返回"无合适表"
        5. 不要包含任何思考过程或解释
        6. 不要使用XML标签（如<think>）
        
        表名："""
        
        selected_table = self.ollama.invoke(prompt)
        
        # 移除思考过程和XML标签
        selected_table = re.sub(r'<think>.*?</think>|<think>.*', '', selected_table, flags=re.DOTALL)
        selected_table = re.sub(r'<.*?>', '', selected_table)
        
        # 清理返回的表名
        selected_table = selected_table.strip().split('\n')[0].strip()
        # 移除可能的引号、标点和其他无关文字
        selected_table = re.sub(r'["\';,]|表名：|表：', '', selected_table)
        selected_table = selected_table.strip()
        
        # 验证表名是否有效
        if not selected_table or selected_table.lower() == 'none' or '<' in selected_table:
            return "无合适表"
            
        print(f"清理后的表名: {selected_table}")
        return selected_table

    def generate_sql(self, question: str, table_name: str, table_info: str):
        """根据表结构生成SQL查询"""
        prompt = f"""基于以下表结构：
        表名：{table_name}
        表结构：{table_info}
        
        问题：{question}
        
        请生成SQL查询。要求：
        1. 只使用表中实际存在的字段
        2. 确保字段用途与问题需求相符
        3. 直接返回单个SQL语句
        4. 不要包含任何解释或思考过程
        
        SQL查询："""
        
        return self.ollama.invoke(prompt)

    def extract_sql(self, sql_text: str) -> str:
        """提取并清理SQL语句"""
        # 确保输入是字符串类型
        sql_text = str(sql_text)
        
        # 移除可能的XML标签和思考过程
        sql_text = re.sub(r'<think>.*?</think>|<think>.*', '', sql_text, flags=re.DOTALL)
        
        # 首先移除 SQL 代码块标记
        sql_text = re.sub(r'```sql|```', '', sql_text)
        
        # 使用正则表达式匹配SQL语句
        sql_pattern = r"(?i)(SELECT|INSERT|UPDATE|DELETE).+?(?=;|$)"
        match = re.search(sql_pattern, sql_text)
        if match:
            sql = match.group(0).strip()
            # 移除可能的引号和反引号
            sql = sql.strip('`"\' ')
            return sql
        return None

    def get_field_descriptions(self, sql: str) -> dict:
        """从表结构中获取使用的字段的描述"""
        # 从SQL中提取表名和字段名
        table_pattern = r'(?i)from\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        field_pattern = r'(?i)select\s+(.*?)\s+from'
        
        tables = re.findall(table_pattern, sql)
        fields_match = re.search(field_pattern, sql)
        
        descriptions = {}
        if fields_match and tables:
            fields = [f.strip() for f in fields_match.group(1).split(',')]
            # 从表结构信息中查找字段描述
            for table in tables:
                for field in fields:
                    # 在self.table_info中查找对应的描述
                    field_clean = field.split('.')[-1] if '.' in field else field
                    # 这里需要根据实际的table_info格式来提取描述
                    # 示例格式，需要根据实际情况调整
                    description = f"表 {table} 中 {field_clean} 字段的描述"
                    descriptions[f"{table}.{field_clean}"] = description
                    
        return descriptions

    def query_database(self, question: str):
        try:
            # 1. 根据表描述选择合适的表
            selected_table = self.select_table(question)
            if selected_table == "无合适表":
                return {"error": "找不到合适的数据表"}
            
            print(f"选择的表: {selected_table}")
            
            # 2. 获取选中表的详细信息
            table_info = self.get_table_info(selected_table)
            print(f"表结构: {table_info}")
            
            # 3. 生成SQL查询
            raw_sql = self.generate_sql(question, selected_table, table_info)
            print(f"生成的SQL: {raw_sql}")
            
            # 4. 清理SQL
            clean_sql = self.extract_sql(raw_sql)
            if not clean_sql:
                return {"error": "无法生成有效的SQL查询"}
            
            print(f"处理后的SQL: {clean_sql}")
            
            # 5. 执行查询
            results = self.db.run(clean_sql)
            print(f"查询结果: {results}")
            
            # 6. 生成回答
            answer_prompt = f"""基于以下查询结果回答问题：
            问题：{question}
            查询结果：{results}
            
            要求：
            1. 直接给出答案
            2. 用简洁的中文回答
            3. 包含具体的数字（如果有）
            """
            
            final_answer = self.ollama.invoke(answer_prompt)
            
            return {
                "selected_table": selected_table,
                "table_info": table_info,
                "sql": clean_sql,
                "results": results,
                "answer": final_answer
            }
            
        except Exception as e:
            print(f"处理错误: {str(e)}")
            return {"error": str(e)}

# 测试代码
def test_toolkit():
    toolkit = OllamaDBToolkit()
    questions = [
        # "哪种鲜花的销量最高？",
        # "统计每种花的平均价格",
        "计算所有花的总销量"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        result = toolkit.query_database(question)
        print(f"结果: {result}")
        print("="*50)

if __name__ == "__main__":
    test_toolkit()
