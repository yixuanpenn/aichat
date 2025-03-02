from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import Graph, StateGraph
import json
import time

# 定义状态类型
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_step: str
    analysis: Dict[str, Any]
    next: str

# 创建聊天模型
def create_chat_model(retries=3, delay=2):
    """创建聊天模型，带重试机制"""
    for attempt in range(retries):
        try:
            model = ChatOllama(
                # model="qwen2.5:14b",  # 使用基础模型
                model="deepseek-r1:70b",  # 使用基础模型
                base_url="http://localhost:11434",
                temperature=0.7,
                timeout=30
            )
            # 测试连接
            test_prompt = "简单测试"
            response = model.invoke(test_prompt)
            print("✓ Ollama 模型加载成功")
            return model
        except Exception as e:
            if attempt < retries - 1:
                print(f"⚠️ 尝试连接 Ollama 失败 ({attempt + 1}/{retries}): {str(e)}")
                print(f"等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    "❌ Ollama 连接失败，请确保：\n"
                    "1. Ollama 服务已启动 (ollama serve)\n"
                    "2. 模型已下载 (ollama pull qwen:7b)\n"
                    "3. 服务端口 11434 可访问"
                )

# 创建模型实例
try:
    model = create_chat_model()
except Exception as e:
    print(str(e))
    exit(1)

def safe_invoke(prompt: str, default_response: Dict = None) -> Dict:
    """安全调用模型"""
    try:
        response = model.invoke(prompt)
        if isinstance(response.content, str):
            try:
                return json.loads(response.content)
            except:
                return default_response or {"error": "解析失败"}
        return response.content
    except Exception as e:
        print(f"⚠️ 调用出错: {str(e)}")
        return default_response or {"error": str(e)}

def analyze_question(state: AgentState) -> AgentState:
    """分析用户问题，确定需要的信息类型"""
    messages = state["messages"]
    last_message = messages[-1].content
    
    prompt = f"""
    分析以下问题并返回 JSON 格式结果：
    问题：{last_message}
    
    {{"required_info": ["需要的信息类型"], "complexity": "问题复杂度", "focus_area": "关注领域"}}
    """
    
    analysis = safe_invoke(prompt, {
        "required_info": ["basic"],
        "complexity": "simple",
        "focus_area": "general"
    })
    
    return {
        "messages": messages,
        "current_step": "analyze",
        "analysis": analysis,
        "next": "plan"
    }

def plan_response(state: AgentState) -> AgentState:
    """根据分析结果规划回答策略"""
    analysis = state["analysis"]
    
    prompt = f"""
    基于分析结果规划回答：
    {json.dumps(analysis, ensure_ascii=False)}
    
    返回 JSON 格式：
    {{"steps": ["回答步骤"], "focus_points": ["重点内容"]}}
    """
    
    plan = safe_invoke(prompt, {
        "steps": ["direct_answer"],
        "focus_points": []
    })
    
    return {
        **state,
        "current_step": "plan",
        "analysis": {**analysis, "plan": plan},
        "next": "respond"
    }

def generate_response(state: AgentState) -> AgentState:
    """生成最终回答"""
    analysis = state["analysis"]
    messages = state["messages"]
    question = messages[-1].content
    
    prompt = f"""
    基于以下信息生成回答：
    分析：{json.dumps(analysis, ensure_ascii=False)}
    问题：{question}
    
    请生成清晰、专业的回答。
    """
    
    response = model.invoke(prompt)
    
    return {
        "messages": [*messages, response],
        "current_step": "respond",
        "analysis": analysis,
        "next": "decide"
    }

def decide_next_step(state: AgentState) -> str:
    """决定下一步操作"""
    messages = state["messages"]
    current_step = state["current_step"]
    
    # 如果已经生成了回答，就结束对话
    if current_step == "respond":
        return "end"
    
    # 如果消息中包含"再见"，也结束对话
    last_message = messages[-1]
    if isinstance(last_message.content, str) and "再见" in last_message.content:
        return "end"
        
    return "analyze"

# 构建工作流图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("analyze", analyze_question)
workflow.add_node("plan", plan_response)
workflow.add_node("respond", generate_response)
workflow.add_node("end", lambda x: x)

# 添加边和条件
workflow.add_edge("analyze", "plan")
workflow.add_edge("plan", "respond")
workflow.add_conditional_edges(
    "respond",
    decide_next_step,
    {
        "analyze": "analyze",
        "end": "end"
    }
)

# 设置入口点
workflow.set_entry_point("analyze")

# 编译图
graph = workflow.compile()

def test_conversation():
    """测试多阶段分析对话"""
    messages = [
        HumanMessage(content="请分析张三的学习情况，重点关注他的课程成绩和学习进度。")
    ]
    
    try:
        result = graph.invoke({
            "messages": messages,
            "current_step": "",
            "analysis": {},
            "next": "analyze"
        })
        
        print("\n=== 对话记录 ===")
        print("🤔 问题:", messages[0].content)
        print("\n🔍 分析过程:")
        print(json.dumps(result["analysis"], ensure_ascii=False, indent=2))
        print("\n💡 回答:")
        print(result["messages"][-1].content)
        
    except Exception as e:
        print(f"❌ 运行错误: {str(e)}")

if __name__ == "__main__":
    test_conversation()