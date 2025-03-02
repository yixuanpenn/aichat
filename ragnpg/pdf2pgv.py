import os
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from sqlalchemy import text
from config.config_manager import ConfigManager

class VectorDBHandler:
    # 预定义的文档类型和对应的collection
    DOCUMENT_TYPES = {
        "财务报告": "finance_report",
        "人事档案": "hr_documents",
        "技术文档": "tech_docs",
        "会议记录": "meeting_minutes",
        "规章制度": "regulations",
        "其他文档": "misc_docs"
    }

    def __init__(self):
        print("\n=== 初始化 VectorDBHandler ===")
        try:
            # 加载配置
            self.config = ConfigManager()
            
            # 构建连接字符串
            db_config = self.config.get_db_config()
            self.connection_string = (
                f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['name']}"
            )
            print(f"数据库连接串: {self.connection_string}")
            
            # 初始化 embedding 模型
            emb_config = self.config.get_embedding_config()
            print(f"\n正在初始化 Embedding 模型...")
            print(f"使用本地模型: {emb_config['path']}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=emb_config['path'],
                model_kwargs={'device': emb_config['device']},
                encode_kwargs={'normalize_embeddings': emb_config['normalize']}
            )
            print("Embedding 模型初始化成功！")
            
            # 初始化数据库实例字典
            self.db_instances = {}
            
        except Exception as e:
            print(f"\n错误: 初始化失败")
            print(f"错误类型: {type(e)}")
            print(f"错误信息: {str(e)}")
            raise

    def _get_db_instance(self, doc_type: str) -> Optional[PGVector]:
        """获取或创建特定文档类型的数据库实例"""
        collection_name = self.DOCUMENT_TYPES.get(doc_type)
        if not collection_name:
            print(f"错误: 未知的文档类型 {doc_type}")
            return None
            
        if collection_name not in self.db_instances:
            try:
                self.db_instances[collection_name] = PGVector(
                    collection_name=collection_name,
                    connection_string=self.connection_string,
                    embedding_function=self.embeddings
                )
                print(f"创建新的collection实例: {collection_name}")
            except Exception as e:
                print(f"创建collection失败: {str(e)}")
                return None
                
        return self.db_instances[collection_name]

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的 MD5 哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _check_file_exists(self, db: PGVector, file_hash: str) -> Tuple[bool, Optional[str]]:
        """检查文件是否已存在于数据库中"""
        try:
            # 使用文件哈希值进行查询
            results = db.similarity_search(
                "CHECK_FILE_EXISTS",
                k=1,
                filter={"file_hash": file_hash}
            )
            
            if results and len(results) > 0:
                # 从元数据中获取文件信息
                metadata = results[0].metadata
                file_info = (
                    f"文件名: {metadata.get('original_filename')}\n"
                    f"添加时间: {metadata.get('added_date')}\n"
                    f"文档类型: {metadata.get('doc_type')}"
                )
                return True, file_info
            
            return False, None
            
        except Exception as e:
            print(f"检查文件存在性时出错: {str(e)}")
            return False, None

    def add_document(self, file_path: str, doc_type: str = "其他文档") -> bool:
        """添加文档到指定类型的collection"""
        print(f"\n=== 处理文档: {file_path} ===")
        print(f"文档类型: {doc_type}")
        
        try:
            # 检查文件类型和大小
            if not self.config.is_supported_file_type(file_path):
                print(f"不支持的文件类型: {file_path}")
                return False
                
            if not self.config.check_file_size(file_path):
                print(f"文件超过大小限制: {file_path}")
                return False
            
            # 获取数据库实例
            db = self._get_db_instance(doc_type)
            if not db:
                return False
                
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"错误: 文件不存在: {file_path}")
                return False
            
            # 计算文件哈希值
            file_hash = self._calculate_file_hash(file_path)
            exists, file_info = self._check_file_exists(db, file_hash)
            
            if exists:
                print(f"\n文件已存在于知识库中:")
                print(file_info)
                print("\n是否重新处理? (y/n)")
                if input().lower() != 'y':
                    return False
            
            # 加载PDF
            print("\n正在加载PDF...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            print(f"PDF加载成功，共 {len(documents)} 页")
            
            # 使用配置的分块参数
            chunk_config = self.config.get_chunk_config()
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_config['chunk_size'],
                chunk_overlap=chunk_config['chunk_overlap'],
                separator="\n"
            )
            split_docs = text_splitter.split_documents(documents)
            
            # 添加元数据
            for doc in split_docs:
                doc.metadata.update({
                    "file_hash": file_hash,
                    "original_filename": os.path.basename(file_path),
                    "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "doc_type": doc_type,
                    "file_path": file_path
                })
            
            print(f"文档分割完成，共 {len(split_docs)} 个片段")
            
            # 如果文件已存在，先删除旧数据
            if exists:
                print("\n正在删除旧数据...")
                self._delete_document(db, file_hash)
            
            # 添加到向量数据库
            print("\n正在添加到向量数据库...")
            db.add_documents(split_docs)
            print(f"文档成功添加到 {doc_type} collection！")
            
            return True
            
        except Exception as e:
            print(f"\n错误: 文档处理失败")
            print(f"错误类型: {type(e)}")
            print(f"错误信息: {str(e)}")
            return False

    def _delete_document(self, db: PGVector, file_hash: str) -> bool:
        """删除指定哈希值的文档"""
        try:
            # 注意：这里需要根据具体的 PGVector 实现来调整删除方法
            # 可能需要直接执行 SQL 删除语句
            with db.connection.connect() as conn:
                conn.execute(
                    text("""
                    DELETE FROM langchain_pg_embedding 
                    WHERE collection_id = :collection_id 
                    AND metadata->>'file_hash' = :file_hash
                    """),
                    {
                        "collection_id": db.collection_id,
                        "file_hash": file_hash
                    }
                )
            return True
        except Exception as e:
            print(f"删除文档时出错: {str(e)}")
            return False

    def list_documents(self, doc_type: str = None) -> List[Dict]:
        """列出知识库中的所有文档"""
        try:
            results = []
            types_to_check = [doc_type] if doc_type else self.DOCUMENT_TYPES.keys()
            
            for dtype in types_to_check:
                db = self._get_db_instance(dtype)
                if not db:
                    continue
                
                # 获取唯一的文件哈希值
                query = """
                SELECT DISTINCT 
                    metadata->>'file_hash' as file_hash,
                    metadata->>'original_filename' as filename,
                    metadata->>'added_date' as added_date,
                    metadata->>'doc_type' as doc_type
                FROM langchain_pg_embedding 
                WHERE collection_id = :collection_id
                """
                
                with db.connection.connect() as conn:
                    result = conn.execute(text(query), {"collection_id": db.collection_id})
                    for row in result:
                        results.append({
                            "file_hash": row.file_hash,
                            "filename": row.filename,
                            "added_date": row.added_date,
                            "doc_type": row.doc_type
                        })
            
            return results
            
        except Exception as e:
            print(f"获取文档列表失败: {str(e)}")
            return []

    def ask(self, question: str, doc_type: str = "其他文档") -> str:
        """从指定collection中查询答案"""
        print(f"\n=== 处理问题 ===")
        print(f"问题: {question}")
        print(f"查询文档类型: {doc_type}")
        
        try:
            # 获取数据库实例
            db = self._get_db_instance(doc_type)
            if not db:
                return "文档类型错误"

            # 使用配置的LLM参数
            llm_config = self.config.get_llm_config()
            llm = Ollama(
                model=llm_config['name'],
                base_url=llm_config['base_url'],
                temperature=llm_config['temperature']
            )
            
            # 设置检索器
            print("\n正在设置检索器...")
            retriever = db.as_retriever(search_kwargs={"k": 3})
            
            # 使用配置的提示词模板
            prompt = PromptTemplate(
                template=self.config.get_qa_template(),
                input_variables=["context", "question"]
            )
            
            # 构建RAG链
            chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = {"context": retriever, "question": RunnablePassthrough()} | chain
            
            # 生成答案
            print("\n正在生成答案...")
            answer = rag_chain.invoke(question)
            print("答案生成完成！")
            
            return answer
            
        except Exception as e:
            print(f"\n错误: 问答失败")
            print(f"错误类型: {type(e)}")
            print(f"错误信息: {str(e)}")
            return "处理失败"

def main():
    print("\n=== 启动 RAG 系统 ===")
    
    # 初始化处理器
    handler = VectorDBHandler()

    # 列出现有文档
    print("\n=== 现有文档 ===")
    docs = handler.list_documents()
    for doc in docs:
        print(f"\n文件名: {doc['filename']}")
        print(f"添加时间: {doc['added_date']}")
        print(f"文档类型: {doc['doc_type']}")   
    
    # 示例：处理不同类型的文档
    documents = [
        {
            "path": "./RAG/files/二级单位财务运行报告.pdf",
            "type": "财务报告"
        },
        {
            "path": "./RAG/files/人事制度.pdf",
            "type": "人事档案"
        }
    ]
    
    # 添加文档
    for doc in documents:
        if handler.add_document(doc["path"], doc["type"]):
            print(f"\n成功添加文档: {doc['path']} 到 {doc['type']} collection")
        else:
            print(f"\n添加文档失败: {doc['path']}")
    
    # 测试不同collection的问答
    questions = [
        {
            # "question": "财务运行情况如何？",
            # "question": "请针对财务运行情况的报告生成一份完整的提示词模版，实现报告的自动生成",
            "question": "有哪些板块？",
            "type": "财务报告"
        },
        # {
        #     "question": "人事制度有哪些规定？",
        #     "type": "人事档案"
        # }
    ]
    
    # 执行问答
    for q in questions:
        print(f"\n问题: {q['question']}")
        print(f"查询collection: {q['type']}")
        answer = handler.ask(q['question'], q['type'])
        print(f"回答: {answer}")

if __name__ == "__main__":
    main()