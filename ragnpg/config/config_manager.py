import os
import yaml
from typing import Dict, Any

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'config'):
            config_path = os.path.join(
                os.path.dirname(__file__),
                'config.yaml'
            )
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print("配置加载成功")
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            raise
    
    def get_db_config(self) -> Dict[str, str]:
        """获取数据库配置"""
        return self.config['database']
    
    def get_collection_name(self, doc_type: str) -> str:
        """获取集合名称"""
        return self.config['vector_store']['collections'].get(
            doc_type,
            self.config['vector_store']['collections']['其他文档']
        )
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """获取嵌入模型配置"""
        return self.config['model']['embedding']
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.config['model']['llm']
    
    def get_chunk_config(self) -> Dict[str, int]:
        """获取文档分块配置"""
        return {
            'chunk_size': self.config['vector_store']['chunk_size'],
            'chunk_overlap': self.config['vector_store']['chunk_overlap']
        }
    
    def get_qa_template(self) -> str:
        """获取问答模板"""
        return self.config['prompts']['qa_template']
    
    def is_supported_file_type(self, file_path: str) -> bool:
        """检查文件类型是否支持"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.config['document']['supported_types']
    
    def check_file_size(self, file_path: str) -> bool:
        """检查文件大小是否在限制范围内"""
        max_size = self.config['document']['max_file_size']
        return os.path.getsize(file_path) <= max_size 