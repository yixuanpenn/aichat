from langchain_community.graphs import NebulaGraph
from langchain_ollama import OllamaLLM
from typing import Optional, Dict, Any
import yaml
import os
import re

class NebulaGraphRAG:
    def __init__(self, config_path: str = "rgp_config.yaml"):
        """初始化 NebulaGraphRAG 系统"""
        try:
            self.config = self._load_config(config_path)
            self.graph = self._create_graph()
            self.llm = self._create_llm()
            self.schema_info = self._get_schema()
        except Exception as e:
            raise RuntimeError(f"初始化失败: {str(e)}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(current_dir, config_path)
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {str(e)}")

    def _create_llm(self) -> OllamaLLM:
        """创建 LLM 实例"""
        try:
            llm_config = self.config.get("llm", {})
            base_params = {
                "model": llm_config.get("model_name"),
                "base_url": llm_config.get("base_url")
            }
            if llm_config.get("parameters"):
                base_params.update(llm_config["parameters"])
            return OllamaLLM(**base_params)
        except Exception as e:
            raise RuntimeError(f"创建 LLM 失败: {str(e)}")

    def _create_graph(self) -> NebulaGraph:
        """创建 NebulaGraph 连接"""
        try:
            nebula_config = self.config.get("graph", {}).get("nebula", {})
            return NebulaGraph(**nebula_config)
        except Exception as e:
            raise RuntimeError(f"创建图数据库连接失败: {str(e)}")

    def _get_schema(self):
        """获取图数据库 Schema 信息"""
        try:
            return self.graph.refresh_schema()
        except Exception as e:
            try:
                tags = self.graph.query("SHOW TAGS;")
                edges = self.graph.query("SHOW EDGES;")
                return f"标签(Tags):\n{tags}\n\n关系(Edges):\n{edges}"
            except Exception as backup_error:
                raise ValueError(f"获取 Schema 失败: {str(e)}, 备用方法也失败: {str(backup_error)}")

    def _get_prompt_template(self):
        """获取提示词模板"""
        return """你是一个 NebulaGraph 专家。请严格按照提供的 Schema 信息生成查询语句。

【Schema 信息】
{schema}

【严格要求】
1. 生成查询前，必须先检查 Schema 中的具体定义：
   - 确认要使用的标签(Tag)在 Schema 中存在
   - 确认要使用的关系(Edge)在 Schema 中存在
   - 确认每个属性都属于对应的标签或关系
2. 如果所需的标签、关系或属性不在 Schema 中，不要生成查询
3. 标签和属性名必须用反引号 `
4. 查询必须以分号结束
5. 只输出查询语句

【检查步骤】
1. 检查标签是否存在
2. 检查标签的属性是否存在
3. 检查关系是否存在
4. 检查关系的属性是否存在
5. 仅使用确认存在的元素生成查询

【示例】
Schema:
Tags: person(name, age), movie(name, year)
Edges: acted_in

问题: 谁出演了《教父》？
MATCH (p:`person`)-[:acted_in]->(m:`movie`) WHERE m.`movie`.name == 'The Godfather' RETURN p.`person`.name;

【用户问题】
{query}"""

    def _clean_llm_response(self, response: str) -> str:
        """清理LLM回答，移除<think>标签及其内容"""
        try:
            # 移除<think>标签及其内容
            cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            # 移除多余的空行
            cleaned = '\n'.join(line for line in cleaned.split('\n') if line.strip())
            return cleaned
        except Exception as e:
            print(f"清理回答失败: {str(e)}")
            return response

    def _get_explanation_prompt(self, query: str, result: str, question: str) -> str:
        """生成解释查询结果的提示词"""
        return f"""请根据以下信息，用中文回答用户的问题：

用户问题：{question}
执行的查询：{query}
查询结果：{result}

请直接给出答案，不需要解释查询过程。如果查询结果为空，请说明未找到相关信息。"""

    def query(self, question: str) -> str:
        """获取 LLM 生成的查询语句并执行，然后解释结果"""
        try:
            # 生成查询语句
            system_prompt = self._get_prompt_template().format(
                schema=self.schema_info,
                query=question
            )
            llm_response = self.llm.invoke(system_prompt).strip()
            
            # 清理LLM回答
            cypher_query = self._clean_llm_response(llm_response)
            print(f"LLM生成的查询语句:\n{cypher_query}")
            
            # 执行查询
            if cypher_query:
                try:
                    query_result = self.graph.query(cypher_query)
                    print(f"\n查询结果:\n{query_result}")
                    
                    # 生成结果解释
                    explanation_prompt = self._get_explanation_prompt(
                        cypher_query, 
                        str(query_result), 
                        question
                    )
                    explanation = self.llm.invoke(explanation_prompt).strip()
                    print(f"\nLLM解释:\n{explanation}")
                    
                    return explanation
                    
                except Exception as e:
                    print(f"执行查询失败: {str(e)}")
                    return f"执行查询失败: {str(e)}"
            else:
                return "无法生成有效的查询语句"
            
        except Exception as e:
            return f"查询过程失败: {str(e)}"

    def query_multiple(self, questions: list[str]) -> list[str]:
        """处理多个问题并返回答案列表"""
        try:
            results = []
            for question in questions:
                print(f"\n处理问题: {question}")
                result = self.query(question)
                results.append(result)
            return results
        except Exception as e:
            raise RuntimeError(f"批量处理问题失败: {str(e)}")

if __name__ == "__main__":
    try:
        rag = NebulaGraphRAG()
        questions = [
            "谁出演了《教父 2》(The Godfather II)?",
            # "谁导演了《教父 2》(The Godfather II)?",
            # "《教父》系列一共有几部电影？"
        ]
        results = rag.query_multiple(questions)
        for question, result in zip(questions, results):
            print(f"\n问题: {question}")
            print(f"答案: {result}")
    except Exception as e:
        print(f"执行失败: {str(e)}")