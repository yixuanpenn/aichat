from typing import List, Dict, Any, Optional
import networkx as nx
from langchain_community.graphs import NetworkxEntityGraph
from langchain_ollama import OllamaLLM
from langchain.chains import GraphQAChain
from langchain_community.embeddings import HuggingFaceEmbeddings
import yaml
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 设置后端，避免显示问题
import requests
import time

def get_available_device():
    """检测并返回可用的计算设备"""
    if torch.backends.mps.is_available():
        print("📱 使用 Apple Silicon GPU")
        return "mps"
    elif torch.cuda.is_available():
        print("🖥️ 使用 NVIDIA GPU")
        return "cuda"
    print("💻 使用 CPU")
    return "cpu"

class LocalEmbeddings(HuggingFaceEmbeddings):
    """本地 Embedding 模型封装类"""
    def __init__(self, model_name: str, **kwargs):
        # 检查并设置最佳可用设备
        device = kwargs.get('device', 'cpu')
        if device != 'cpu':
            device = get_available_device()
            print(f"使用设备: {device}")
        
        super().__init__(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={
                'normalize_embeddings': kwargs.get('normalize_embeddings', True),
                'device': device
            }
        )

class GraphRAGConfig:
    """配置类"""
    def __init__(self):
        self.config = self._load_config()
        self.llm_config = self.config['llm']
        self.embeddings_config = self.config['embeddings']
        self.graph_config = self.config['graph']
        self.graph = self._create_knowledge_graph()  # 将图谱创建移到这里，避免重复创建

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            # 修改配置文件路径为当前目录
            config_path = Path(__file__).parent / 'rgp_config.yaml'
            
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if not config:
                raise ValueError("配置文件为空")
                
            self._validate_config(config)
            return config
                
        except Exception as e:
            print(f"❌ 配置文件加载失败: {str(e)}")
            raise

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证配置完整性"""
        try:
            if not isinstance(config, dict):
                raise ValueError("配置格式错误")
            
            # 验证 LLM 配置
            if 'llm' not in config:
                raise ValueError("缺少 LLM 配置")
            
            llm_config = config['llm']
            required_fields = ['model_name', 'base_url']
            for field in required_fields:
                if field not in llm_config:
                    raise ValueError(f"LLM 配置缺少字段: {field}")
            
            # 确保 base_url 格式正确
            base_url = llm_config['base_url']
            if not isinstance(base_url, str):
                raise ValueError("base_url 必须是字符串")
            
            # 添加默认参数如果不存在
            if 'parameters' not in llm_config:
                llm_config['parameters'] = {}
            
            default_params = {
                'temperature': 0.7,
                'num_ctx': 4096,
                'num_predict': -1,
                'stop': None,
                'num_keep': 0,
                'stream': True
            }
            
            for key, value in default_params.items():
                if key not in llm_config['parameters']:
                    llm_config['parameters'][key] = value
            
            print("✓ 配置验证通过")
            
        except Exception as e:
            print(f"❌ 配置验证失败: {str(e)}")
            raise

    def _create_llm(self) -> OllamaLLM:
        """创建语言模型"""
        try:
            llm_config = self.config['llm']
            base_url = llm_config['base_url']
            
            # 检查并修正 base_url 格式
            if not base_url.startswith(('http://', 'https://')):
                base_url = f"http://{base_url}"
            base_url = base_url.rstrip('/')
            
            print(f"\n正在检查 Ollama 服务连接...")
            print(f"服务地址: {base_url}")
            
            # 测试连接
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(f"{base_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        print("✓ Ollama 服务连接成功")
                        break
                    else:
                        print(f"⚠️ 服务响应异常: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    if attempt < max_retries - 1:
                        print(f"⚠️ 连接失败 (尝试 {attempt + 1}/{max_retries})")
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        print("\n❌ Ollama 服务连接失败")
                        print("请检查以下内容：")
                        print("1. Ollama 服务是否已启动")
                        print(f"2. 服务地址 {base_url} 是否正确")
                        print("3. 服务器防火墙是否允许连接")
                        print("4. 网络连接是否正常")
                        print("\n如果使用远程服务器，请确保：")
                        print("1. 服务器已开放 11434 端口")
                        print("2. Ollama 服务允许远程连接")
                        print("3. 可以通过以下命令测试连接：")
                        print(f"   curl {base_url}/api/tags\n")
                        raise ConnectionError(f"无法连接到 Ollama 服务: {base_url}")
                except requests.exceptions.Timeout:
                    print(f"⚠️ 连接超时")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        raise TimeoutError(f"连接 Ollama 服务超时: {base_url}")
            
            # 创建 LLM 实例
            parameters = llm_config.get('parameters', {})
            llm = OllamaLLM(
                model=llm_config['model_name'],
                base_url=base_url,
                temperature=parameters.get('temperature', 0.7),
                num_ctx=parameters.get('num_ctx', 4096),
                num_predict=parameters.get('num_predict', -1),
                stop=parameters.get('stop', None),
                num_keep=parameters.get('num_keep', 0),
                stream=parameters.get('stream', True)
            )
            
            print(f"✓ 语言模型创建成功: {llm_config['model_name']}")
            return llm
            
        except Exception as e:
            print(f"❌ 语言模型创建失败: {str(e)}")
            raise

    def _create_knowledge_graph(self):
        """创建纯中文知识图谱"""
        G = nx.DiGraph()
        
        # 1. 核心实体节点 - 全中文属性
        学生信息 = {
            "实体类型": "学生",
            "描述文本": """姓名：张三
个人信息：北京大学计算机系在读学生
年级：大三
专业：计算机科学与技术
特点：成绩优秀，擅长编程和算法
获奖情况：多次获得学习优秀奖学金""",
            "属性": {
                "学号": "2021001",
                "年级": "大三",
                "绩点": "3.9",
                "入学年份": "2021",
                "预计毕业": "2025"
            }
        }
        
        学校信息 = {
            "实体类型": "学校",
            "描述文本": """学校名称：北京大学
办学类型：综合性研究型大学
地理位置：北京市海淀区
办学特色：中国顶尖高等教育机构
师资力量：雄厚
科研实力：国际一流""",
            "属性": {
                "位置": "北京市海淀区",
                "类型": "985工程、双一流",
                "创办时间": "1898年",
                "现任校长": "龚旗煌"
            }
        }
        
        专业信息 = {
            "实体类型": "专业",
            "描述文本": """专业名称：计算机科学与技术
所属院系：信息科学技术学院
培养目标：培养具有扎实理论基础和实践能力的计算机专业人才
特色方向：人工智能、大数据、网络安全
课程体系：理论与实践并重""",
            "属性": {
                "所属院系": "信息科学技术学院",
                "学制": "4年",
                "学位类型": "工学学士",
                "招生人数": "120人/年"
            }
        }
        
        课程信息1 = {
            "实体类型": "课程",
            "描述文本": """课程名称：数据结构
课程简介：计算机专业核心基础课程
主要内容：数据组织方法和相关算法
教学方式：课堂讲授与上机实践相结合
考核方式：理论考试与实验项目评估
学习目标：掌握各种数据结构的原理和应用""",
            "属性": {
                "学分": "4",
                "课程性质": "必修",
                "开课学期": "大二上",
                "周学时": "4",
                "考核方式": "考试+实验"
            }
        }
        
        课程信息2 = {
            "实体类型": "课程",
            "描述文本": """课程名称：算法设计
课程简介：计算机专业核心进阶课程
主要内容：常用算法设计方法与分析技术
教学方式：理论讲解与编程实践
考核方式：期末考试与课程设计
学习目标：培养算法设计与分析能力""",
            "属性": {
                "学分": "4",
                "课程性质": "必修",
                "开课学期": "大二下",
                "周学时": "4",
                "考核方式": "考试+大作业"
            }
        }
        
        # 添加节点
        节点列表 = [
            ("张三", 学生信息),
            ("北京大学", 学校信息),
            ("计算机科学与技术", 专业信息),
            ("数据结构", 课程信息1),
            ("算法设计", 课程信息2)
        ]
        
        # 2. 关系边 - 全中文描述
        关系列表 = [
            ("张三", "北京大学", {
                "关系类型": "就读于",
                "描述文本": """在读状态：本科在读
入学时间：2021年9月
当前情况：学习认真刻苦
综合表现：成绩优异，多次获得奖学金
预计毕业：2025年6月"""
            }),
            ("张三", "计算机科学与技术", {
                "关系类型": "主修",
                "描述文本": """主修专业：计算机科学与技术
入学年级：2021级
学习表现：专业课成绩优异
综合排名：专业前10%
特殊表现：积极参与科研实践"""
            }),
            ("张三", "数据结构", {
                "关系类型": "已修课程",
                "描述文本": """修读情况：已完成课程学习
修读学期：2022年春季学期
考试成绩：95分
实验成绩：优秀
总体评价：理论与实践能力突出"""
            }),
            ("张三", "算法设计", {
                "关系类型": "在修课程",
                "描述文本": """修读情况：正在修读
修读学期：2023年秋季学期
课程进度：已完成60%
当前表现：课堂表现活跃
作业完成：认真出色"""
            }),
            ("计算机科学与技术", "数据结构", {
                "关系类型": "包含课程",
                "描述文本": """课程性质：专业必修课
开课时间：第三学期
先修课程：程序设计基础
教学特点：理论与实践并重
课程目标：掌握数据结构基础"""
            }),
            ("计算机科学与技术", "算法设计", {
                "关系类型": "包含课程",
                "描述文本": """课程性质：专业必修课
开课时间：第四学期
先修课程：数据结构
教学特点：注重实践能力
课程目标：培养算法设计能力"""
            })
        ]
        
        # 构建图谱
        G.add_nodes_from(节点列表)
        G.add_edges_from(关系列表)
        return G
    
    def _create_embeddings(self):
        """
        创建并配置嵌入模型
        Returns:
            HuggingFaceEmbeddings: 配置好的嵌入模型
        """
        embeddings_config = self.config['embeddings']
        
        if embeddings_config['type'] == 'local':
            # 更新设备配置
            embeddings_config['parameters']['device'] = get_available_device()
            
            try:
                return LocalEmbeddings(
                    model_name=embeddings_config['model_name'],
                    **embeddings_config['parameters']
                )
            except Exception as e:
                print(f"警告: 模型加载失败，尝试使用CPU: {str(e)}")
                embeddings_config['parameters']['device'] = 'cpu'
                return LocalEmbeddings(
                    model_name=embeddings_config['model_name'],
                    **embeddings_config['parameters']
                )
        else:
            raise ValueError(f"不支持的 embedding 类型: {embeddings_config['type']}")
    
    def _create_graph_rag(self):
        """创建图谱检索器"""
        return NetworkxEntityGraph(
            graph=self.graph,  # 使用已创建的图谱实例
            embeddings=self._create_embeddings()
        )
    
    def get_formatted_context(self, nodes: List[str]) -> str:
        """生成更详细的上下文信息"""
        context_parts = []
        
        # 1. 添加实体详细信息
        context_parts.append("实体信息：")
        for node in nodes:
            node_data = self.graph.nodes[node]  # 使用已创建的图谱实例
            desc = node_data.get('description', '')
            type_info = node_data.get('type', '')
            
            # 添加所有额外属性
            extra_info = []
            for key, value in node_data.items():
                if key not in ['type', 'description']:
                    extra_info.append(f"{key}: {value}")
            
            context_parts.append(f"- {node}（{type_info}）")
            context_parts.append(f"  描述: {desc}")
            if extra_info:
                context_parts.append(f"  详细信息: {', '.join(extra_info)}")
        
        # 2. 添加关系详细信息
        context_parts.append("\n关系信息：")
        subgraph = self.graph.subgraph(nodes)
        for edge in subgraph.edges(data=True):
            source, target, data = edge
            relation = data.get('relation', '')
            desc = data.get('description', '')
            
            # 添加所有额外属性
            extra_info = []
            for key, value in data.items():
                if key not in ['relation', 'description']:
                    extra_info.append(f"{key}: {value}")
            
            context_parts.append(f"- {source} {relation} {target}")
            context_parts.append(f"  描述: {desc}")
            if extra_info:
                context_parts.append(f"  详细信息: {', '.join(extra_info)}")
        
        return '\n'.join(context_parts)
    
    def query(self, query: str) -> str:
        """处理用户查询"""
        try:
            print(f"\n🤔 问题: {query}")
            print("-" * 50)
            
            if not query.strip():
                return "请输入有效的问题。"
            
            # 查找相关实体和关系
            context = self._get_graph_context(query)
            
            # 构建提示词
            prompt = f"""基于以下知识图谱信息回答问题：

{context}

问题：{query}
请根据上述信息提供详细的回答。"""

            # 使用 LLM 生成回答
            print("💭 回答：")
            response = []
            for chunk in self.llm.stream(prompt):
                if chunk:
                    print(chunk, end='', flush=True)
                    response.append(chunk)
            
            print("\n")
            return ''.join(response)
            
        except Exception as e:
            error_msg = f"❌ 错误: {str(e)}"
            print(error_msg)
            return error_msg

    def _get_graph_context(self, question: str) -> str:
        """获取与问题相关的图谱上下文"""
        context_parts = []
        
        # 1. 获取所有实体信息
        context_parts.append("实体信息：")
        for node, data in self.graph.nodes(data=True):
            # 检查实体是否与问题相关
            if any(keyword in question.lower() for keyword in [node.lower(), data.get('content', '').lower()]):
                context_parts.append(f"\n- {node}（{data.get('type', '未知类型')}）")
                context_parts.append(f"  描述: {data.get('content', '无描述')}")
                
                # 添加额外属性
                extra_info = []
                for key, value in data.items():
                    if key not in ['type', 'content']:
                        extra_info.append(f"{key}: {value}")
                if extra_info:
                    context_parts.append(f"  其他信息: {', '.join(extra_info)}")
        
        # 2. 获取相关关系信息
        context_parts.append("\n关系信息：")
        for src, dst, data in self.graph.edges(data=True):
            # 检查关系是否与问题相关
            if any(keyword in question.lower() for keyword in [src.lower(), dst.lower()]):
                relation = data.get('relation_type', '关联')
                content = data.get('content', '')
                context_parts.append(f"\n- {src} {relation} {dst}")
                if content:
                    context_parts.append(f"  描述: {content}")
                
                # 添加额外属性
                extra_info = []
                for key, value in data.items():
                    if key not in ['relation_type', 'content']:
                        extra_info.append(f"{key}: {value}")
                if extra_info:
                    context_parts.append(f"  其他信息: {', '.join(extra_info)}")
        
        return '\n'.join(context_parts)

    def _get_graph_info(self) -> str:
        """获取图谱信息"""
        info = []
        
        # 获取节点信息
        for node, data in self.graph.nodes(data=True):  # 使用已创建的图谱实例
            info.append(f"实体：{node}")
            for key, value in data.items():
                info.append(f"- {key}: {value}")
        
        # 获取关系信息
        for src, dst, data in self.graph.edges(data=True):
            info.append(f"\n关系：{src} -> {dst}")
            for key, value in data.items():
                info.append(f"- {key}: {value}")
        
        return "\n".join(info)

    def _get_context(self, query: str) -> Optional[str]:
        """获取相关上下文"""
        try:
            # 从配置中获取参数
            graph_config = self.config['graph']
            max_nodes = graph_config['max_nodes']
            similarity_threshold = graph_config['similarity_threshold']
            
            nodes = list(self.graph.nodes(data=True))[:max_nodes]
            edges = list(self.graph.edges(data=True))
            
            context = []
            context.extend(f"节点: {node} (类型: {data['type']})" 
                          for node, data in nodes)
            
            node_names = {n[0] for n in nodes}
            context.extend(f"关系: {src} {data['relation']} {dst}"
                          for src, dst, data in edges
                          if src in node_names and dst in node_names)
            
            return "\n".join(context)
            
        except Exception as e:
            print(f"❌ 获取上下文失败: {str(e)}")
            return None

    def visualize_graph(self, output_path: str = None):
        """将知识图谱可视化并保存为图片"""
        try:
            if output_path is None:
                output_path = Path(__file__).parent / "knowledge_graph.png"
            
            # 确保输出路径是 Path 对象
            if isinstance(output_path, str):
                output_path = Path(output_path)
            
            # 设置matplotlib支持中文
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建新图形
            plt.figure(figsize=(20, 15))
            
            # 使用spring布局
            pos = nx.spring_layout(self.graph, k=2, iterations=100)
            
            # 绘制节点
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                node_color='lightblue',
                node_size=3000,
                alpha=0.7
            )
            
            # 绘制边
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edge_color='gray',
                arrows=True,
                arrowsize=25,
                width=2
            )
            
            # 添加节点标签
            nx.draw_networkx_labels(
                self.graph,
                pos,
                font_size=12,
                font_family='sans-serif',
                font_weight='bold'
            )
            
            # 添加边标签
            edge_labels = nx.get_edge_attributes(self.graph, 'type')
            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels=edge_labels,
                font_size=10,
                font_weight='bold'
            )
            
            # 设置图形样式
            plt.title("知识图谱可视化", fontsize=20, pad=20, fontweight='bold')
            plt.axis('off')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(
                output_path,
                format='png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2,
                transparent=False,
                facecolor='white'
            )
            plt.close()
            
            print(f"✓ 知识图谱已保存至: {output_path.absolute()}")
            
        except Exception as e:
            print(f"❌ 图谱可视化失败: {str(e)}")

    def _generate_graph_visualization(self) -> None:
        """生成知识图谱可视化"""
        try:
            output_path = Path(__file__).parent / "knowledge_graph.png"
            
            # 设置matplotlib后端，避免显示问题
            import matplotlib
            matplotlib.use('Agg')
            
            # 简化字体设置，只使用常见中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 这个字体支持中文
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图形
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # 绘制节点
            nx.draw_networkx_nodes(
                self.graph, 
                pos,
                node_color='lightblue',
                node_size=2000,
                alpha=0.7
            )
            
            # 绘制边
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edge_color='gray',
                arrows=True,
                arrowsize=20
            )
            
            # 添加节点标签
            nx.draw_networkx_labels(
                self.graph,
                pos,
                font_size=10,
                font_family='Arial Unicode MS'  # 指定字体
            )
            
            # 添加边标签
            edge_labels = nx.get_edge_attributes(self.graph, 'relation')
            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels=edge_labels,
                font_size=8,
                font_family='Arial Unicode MS'  # 指定字体
            )
            
            # 设置图形样式
            plt.title("知识图谱可视化", fontsize=16, fontfamily='Arial Unicode MS')  # 指定字体
            plt.axis('off')
            
            # 保存图片
            plt.savefig(
                output_path,
                format='png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1
            )
            plt.close()
            
            print(f"✓ 知识图谱已保存至: {output_path}")
            
        except Exception as e:
            print(f"❌ 图谱可视化失败: {str(e)}")

class GraphRAG:
    def __init__(self):
        """初始化 GraphRAG 系统"""
        try:
            self.config_manager = GraphRAGConfig()
            self.llm = self.config_manager._create_llm()
            self.graph = self.config_manager._create_knowledge_graph()
            
            print("\n=== 创建 NetworkxEntityGraph ===")
            self.entity_graph = NetworkxEntityGraph(self.graph)
            
            # 创建图谱理解提示词
            self.understanding_prompt = """你是一个知识图谱分析专家。请基于以下图谱查询结果，生成一段完整的文本理解。

【图谱查询结果】

1. 节点信息：
{nodes_info}

2. 边信息：
{edges_info}

3. 路径信息：
{paths_info}

4. 子图信息：
{subgraph_info}

5. 属性信息：
{properties_info}

请你：
1. 分析所有实体之间的关系
2. 整合所有相关信息
3. 生成一段完整的文本描述
4. 确保不遗漏重要信息

请按如下格式生成理解：

【实体关系分析】
[分析图谱中的实体及其关系]

【属性信息整合】
[整合各实体的属性信息]

【完整文本理解】
[生成一段连贯的文本，描述图谱包含的所有重要信息]
"""
            
            # 创建问答提示词
            self.qa_prompt = """你是一个知识图谱问答专家。请基于下面的图谱理解回答问题。

【图谱理解文本】
{graph_understanding}

【用户问题】
{query}

请你：
1. 仔细阅读图谱理解文本
2. 找出与问题相关的信息
3. 确保回答完全基于图谱理解
4. 如果信息不足，请明确说明

请按如下格式回答：

【相关信息】
[列出与问题相关的关键信息]

【回答】
[基于图谱理解的完整回答]
"""
            
            print("\n=== 创建 GraphQAChain ===")
            self.qa_chain = GraphQAChain.from_llm(
                llm=self.llm,
                graph=self.entity_graph,
                verbose=True,
                return_intermediate_steps=True
            )
            
            print("✓ 系统初始化完成")
            
        except Exception as e:
            print(f"\n❌ 初始化失败: {str(e)}")
            raise

    def _get_graph_understanding(self, query_results):
        """生成图谱理解文本"""
        try:
            # 使用理解提示词生成文本
            understanding_input = {
                "nodes_info": "\n\n".join(query_results["nodes_info"]),
                "edges_info": "\n\n".join(query_results["edges_info"]),
                "paths_info": "\n\n".join(query_results["paths_info"]),
                "subgraph_info": "\n\n".join(query_results["subgraph_info"]),
                "properties_info": "\n\n".join(query_results["properties_info"])
            }
            
            understanding = self.llm(self.understanding_prompt.format(**understanding_input))
            return understanding
            
        except Exception as e:
            print(f"生成图谱理解失败: {str(e)}")
            return "无法生成图谱理解"

    def query(self, question: str) -> str:
        """使用 GraphRAG 进行问答"""
        try:
            print(f"\n=== 问题: {question} ===")
            print("\n开始查询知识图谱...")
            
            # 1. 获取查询结果
            query_results = self._get_query_results(question)
            
            # 2. 生成图谱理解
            print("\n=== 生成图谱理解 ===")
            graph_understanding = self._get_graph_understanding(query_results)
            print("\n图谱理解文本:")
            print(graph_understanding)
            
            # 3. 基于图谱理解回答问题
            print("\n=== 生成回答 ===")
            qa_input = {
                "query": question,
                "graph_understanding": graph_understanding
            }
            
            result = self.llm(self.qa_prompt.format(**qa_input))
            
            # 4. 显示完整过程
            print("\n=== 查询过程 ===")
            print("\n1. 图谱查询结果:")
            for key, values in query_results.items():
                print(f"\n{key}:")
                print("\n".join(values))
            
            print("\n2. 图谱理解:")
            print(graph_understanding)
            
            print("\n3. 最终回答:")
            print(result)
            
            return result
            
        except Exception as e:
            error_msg = f"查询失败: {str(e)}"
            print(error_msg)
            return error_msg

    def _get_query_results(self, question: str):
        """获取完整的图谱查询结果"""
        results = {
            "nodes_info": [],    # 节点信息
            "edges_info": [],    # 边信息
            "paths_info": [],    # 路径信息
            "subgraph_info": [], # 子图信息
            "properties_info": [] # 属性信息
        }
        
        # 1. 获取节点信息
        for node, data in self.graph.nodes(data=True):
            node_info = [
                f"节点：{node}",
                f"类型：{data.get('实体类型', '未知')}",
                f"描述：{data.get('描述文本', '无描述')}"
            ]
            if '属性' in data:
                node_info.append("属性列表：")
                for k, v in data['属性'].items():
                    node_info.append(f"  - {k}: {v}")
            results["nodes_info"].append("\n".join(node_info))
            
            # 收集属性信息
            if '属性' in data:
                results["properties_info"].append(
                    f"实体 {node} 的属性：\n" + 
                    "\n".join(f"  - {k}: {v}" for k, v in data['属性'].items())
                )
        
        # 2. 获取边信息
        for u, v, data in self.graph.edges(data=True):
            edge_info = [
                f"起点：{u}",
                f"终点：{v}",
                f"关系类型：{data.get('关系类型', '未知')}",
                f"关系描述：{data.get('描述文本', '无描述')}"
            ]
            results["edges_info"].append("\n".join(edge_info))
        
        # 3. 获取路径信息
        for node in self.graph.nodes():
            for target in self.graph.nodes():
                if node != target:
                    try:
                        paths = nx.all_simple_paths(self.graph, node, target, cutoff=2)
                        for path in paths:
                            path_info = []
                            for i in range(len(path)-1):
                                edge_data = self.graph[path[i]][path[i+1]]
                                path_info.append(
                                    f"{path[i]} --[{edge_data.get('关系类型', '未知')}]--> {path[i+1]}"
                                )
                            if path_info:
                                results["paths_info"].append("\n".join(path_info))
                    except nx.NetworkXNoPath:
                        continue
        
        # 4. 获取子图信息
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                subgraph_info = [f"中心节点：{node}"]
                subgraph_info.append("相关联的节点：")
                for neighbor in neighbors:
                    edge_data = self.graph[node][neighbor]
                    subgraph_info.append(
                        f"  - 与 {neighbor} 的关系：{edge_data.get('关系类型', '未知')}"
                    )
                results["subgraph_info"].append("\n".join(subgraph_info))
        
        return results

    def visualize_graph(self, output_path: str = None):
        """将知识图谱可视化并保存为图片"""
        try:
            if output_path is None:
                output_path = Path(__file__).parent / "knowledge_graph.png"
            
            # 确保输出路径是 Path 对象
            if isinstance(output_path, str):
                output_path = Path(output_path)
            
            # 设置matplotlib支持中文
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建新图形
            plt.figure(figsize=(20, 15))
            
            # 使用spring布局
            pos = nx.spring_layout(self.graph, k=2, iterations=100)
            
            # 绘制节点
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                node_color='lightblue',
                node_size=3000,
                alpha=0.7
            )
            
            # 绘制边
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edge_color='gray',
                arrows=True,
                arrowsize=25,
                width=2
            )
            
            # 添加节点标签
            nx.draw_networkx_labels(
                self.graph,
                pos,
                font_size=12,
                font_family='sans-serif',
                font_weight='bold'
            )
            
            # 添加边标签
            edge_labels = nx.get_edge_attributes(self.graph, 'type')
            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels=edge_labels,
                font_size=10,
                font_weight='bold'
            )
            
            # 设置图形样式
            plt.title("知识图谱可视化", fontsize=20, pad=20, fontweight='bold')
            plt.axis('off')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(
                output_path,
                format='png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2,
                transparent=False,
                facecolor='white'
            )
            plt.close()
            
            print(f"✓ 知识图谱已保存至: {output_path.absolute()}")
            
        except Exception as e:
            print(f"❌ 图谱可视化失败: {str(e)}")

def test_graph_rag():
    """测试 GraphRAG 系统"""
    try:
        print("\n=== 开始 GraphRAG 测试 ===")
        rag = GraphRAG()
        
        # 测试问题
        test_questions = [
            "请详细描述张三在北京大学的学习情况。",
            "张三在数据结构和算法设计这两门课程的表现如何？",
            "计算机科学与技术专业的课程体系是怎样的？",
            "根据图谱信息，张三的整体学习表现如何？",
            "北京大学计算机科学与技术专业有什么特点？"
        ]
        
        print("\n=== 开始测试查询 ===")
        for i, question in enumerate(test_questions, 1):
            print(f"\n测试 {i}" + "="*40)
            rag.query(question)
            print("="*50)
        
        print("\n✓ 测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    test_graph_rag()

