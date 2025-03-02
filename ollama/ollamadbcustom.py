from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseSequentialChain
from langchain_ollama import OllamaLLM
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from typing import Optional, Dict, Any

class CustomSQLDatabaseChain:
    def __init__(self):
        self.ollama = OllamaLLM(
            model="deepseek-r1:70b",
            base_url="http://localhost:11434",
        )
        
        self.db = SQLDatabase.from_uri(
            "mysql+pymysql://test:test1234@localhost/test"
        )
        
        # 创建数据库链
        self.db_chain = SQLDatabaseSequentialChain.from_llm(
            llm=self.ollama,
            db=self.db,
            verbose=True,
            return_intermediate_steps=True  # 保留此选项以获取中间步骤
        )

    def process_sql(self, sql: str) -> str:
        """处理生成的 SQL 查询"""
        processed_sql = sql.strip()
        
        # 示例：添加执行计划分析
        processed_sql = f"EXPLAIN {processed_sql}"
        
        print(f"处理后的 SQL: {processed_sql}")
        return processed_sql

    def query(self, question: str) -> Optional[str]:
        """执行查询并返回结果"""
        try:
            # 使用 invoke 替代 run，并正确处理返回结果
            result: Dict[str, Any] = self.db_chain.invoke({"query": question})
            
            # 打印生成的 SQL（从中间步骤中获取）
            if "intermediate_steps" in result:
                sql = result["intermediate_steps"][0]
                print(f"生成的 SQL: {sql}")
                # 可以在这里处理 SQL
                processed_sql = self.process_sql(sql)
            
            # 返回最终结果
            return result["result"]
            
        except Exception as e:
            print(f"查询执行错误: {str(e)}")
            return None

def test_custom_chain():
    """测试自定义查询链"""
    chain = CustomSQLDatabaseChain()
    
    questions = [
        "哪种鲜花的销量最高？",
        "统计每种花的平均价格",
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        result = chain.query(question)
        if result:
            print(f"回答: {result}")
        print("="*50)

if __name__ == "__main__":
    test_custom_chain()