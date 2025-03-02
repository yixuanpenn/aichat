from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseSequentialChain
from langchain_ollama import OllamaLLM
import pymysql

ollama = OllamaLLM(
    model="deepseek-r1:70b",
    # model= "llama3.1:8b",
    # model= "llama3.3",
    # model= "qwen2.5:14b",
    # model = "mistral",
    base_url="http://localhost:11434",
)

# 连接到FlowerShop数据库
db = SQLDatabase.from_uri(f"mysql+pymysql://test:test1234@localhost/test")

db_chain = SQLDatabaseSequentialChain.from_llm(ollama, db, verbose=True, use_query_checker=True)

# 使用Agent执行SQL查询
questions = [
    # "一共有多少员工？",
    "哪种鲜花的存货数量最少？",
    "哪种鲜花的销售量最高？",
    "从法国进口的鲜花有多少种？",
    "哪种鲜花的销售量最高？"
]

for question in questions:
    response = db_chain.invoke(question)
    print(response)
