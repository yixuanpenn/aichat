from langchain_community.graphs import NebulaGraph
from langchain_ollama import OllamaLLM
from langchain_community.chains.graph_qa.nebulagraph import NebulaGraphQAChain
from langchain.prompts import PromptTemplate
from typing import Dict, Any
import yaml
import os

class NebulaGraphRAG:
    def __init__(self, config_path: str = "rgp_config.yaml"):
        """初始化 NebulaGraphRAG 系统"""
        try:
            print("\n=== 初始化 NebulaGraphRAG ===")
            self.config = self._load_config(config_path)
            self.graph = self._create_graph()
            self.llm = self._create_llm()
            self.qa_chain = self._create_qa_chain()
            print("✓ NebulaGraphRAG 初始化成功")
        except Exception as e:
            raise RuntimeError(f"初始化失败: {str(e)}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {str(e)}")

    def _create_llm(self) -> OllamaLLM:
        """创建 LLM 实例"""
        try:
            print("\n=== 创建 LLM ===")
            llm_config = self.config.get("llm", {})
            base_params = {
                "model": llm_config.get("model_name"),
                "base_url": llm_config.get("base_url")
            }
            if llm_config.get("parameters"):
                base_params.update(llm_config["parameters"])
            
            llm = OllamaLLM(**base_params)
            print(f"✓ LLM ({llm_config.get('model_name')}) 创建成功")
            return llm
        except Exception as e:
            raise RuntimeError(f"创建 LLM 失败: {str(e)}")

    def _create_graph(self) -> NebulaGraph:
        """创建 NebulaGraph 连接"""
        try:
            print("\n=== 创建 NebulaGraph 连接 ===")
            graph = NebulaGraph(**self.config.get("graph", {}).get("nebula", {}))
            print("✓ NebulaGraph 连接成功")
            return graph
        except Exception as e:
            raise RuntimeError(f"创建图数据库连接失败: {str(e)}")

    def _create_qa_chain(self) -> NebulaGraphQAChain:
        """创建问答链"""
        try:
            print("\n=== 创建 QA Chain ===")
            chain = NebulaGraphQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                verbose=True,
                return_intermediate_steps=True,
                return_direct=True,
                allow_dangerous_requests=True
            )
            print("✓ QA Chain 创建成功")
            return chain
        except Exception as e:
            raise RuntimeError(f"创建 QA Chain 失败: {str(e)}")

    def query(self, question: str) -> str:
        """执行图谱问答"""
        try:
            result = self.qa_chain.invoke(question)
            return result
            
        except Exception as e:
            error_msg = f"查询失败: {str(e)}"
            print(f"\n错误: {error_msg}")
            return error_msg

def test_nebula_rag():
    """测试 NebulaGraphRAG 系统"""
    try:
        rag = NebulaGraphRAG()
        questions = [
            "谁出演了《教父 2》(The Godfather II)?",
            # "Who played in The Godfather II?"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            print("回答:", rag.query(question))
            print("="*50)
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    test_nebula_rag()