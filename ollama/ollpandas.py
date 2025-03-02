# ! pip install tabulate

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_ollama import OllamaLLM
import pandas as pd
import pymysql
from sqlalchemy import create_engine

# 创建数据库连接
connection_string = "mysql+pymysql://test:test1234@localhost/test"
engine = create_engine(connection_string)

# 从 MySQL 读取数据表
try:
    # 假设要查询的表名是 flowers，你可以根据实际情况修改
    query = "SELECT * FROM flowers"
    df = pd.read_sql(query, engine)
    
    # 创建 Ollama 实例
    ollama = OllamaLLM(
        model="deepseek-r1:70b",
        base_url="http://localhost:11434",
    )

    # 创建 Pandas DataFrame Agent
    agent = create_pandas_dataframe_agent(ollama, df, verbose=True, allow_dangerous_code=True)

    # 测试查询
    questions = [
        "一共有多少种鲜花？",
        "哪种鲜花的存货数量最少？",
        "哪种鲜花的销售量最高？"
    ]

    for question in questions:
        print(f"\n问题: {question}")
        response = agent.invoke(question)
        print(f"回答: {response}")
        print("="*50)

except Exception as e:
    print(f"发生错误: {str(e)}")
finally:
    engine.dispose()  # 关闭数据库连接