import os
from typing import List, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

class DocumentLoader:
    """文档加载器，支持 PDF、Word 和 CSV 文件"""
    
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n"
        )
    
    def load_document(self, file_path: str) -> Optional[List[Document]]:
        """
        根据文件类型加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Document]: 文档列表，如果处理失败则返回 None
        """
        print(f"\n=== 处理文件：{file_path} ===")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"错误：文件不存在：{file_path}")
                return None
            
            # 获取文件扩展名（转换为小写）
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # 根据文件类型选择加载器
            if file_extension == '.pdf':
                print("检测到 PDF 文件，使用 PDF 加载器")
                return self._load_pdf(file_path)
                
            elif file_extension in ['.doc', '.docx']:
                print("检测到 Word 文件，使用 Word 加载器")
                return self._load_word(file_path)
                
            elif file_extension == '.csv':
                print("检测到 CSV 文件，使用 CSV 加载器")
                return self._load_csv(file_path)
                
            else:
                print(f"不支持的文件类型：{file_extension}")
                print("目前仅支持 PDF、Word(doc/docx) 和 CSV 文件")
                return None
                
        except Exception as e:
            print(f"文件处理失败：{str(e)}")
            print(f"错误类型：{type(e)}")
            return None
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """加载 PDF 文件"""
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"PDF 加载成功，共 {len(pages)} 页")
        
        # 分割文档
        docs = self.text_splitter.split_documents(pages)
        print(f"文档分割完成，共 {len(docs)} 个片段")
        return docs
    
    def _load_word(self, file_path: str) -> List[Document]:
        """加载 Word 文件"""
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load()
        print(f"Word 文档加载成功")
        
        # 分割文档
        split_docs = self.text_splitter.split_documents(docs)
        print(f"文档分割完成，共 {len(split_docs)} 个片段")
        return split_docs
    
    def _load_csv(self, file_path: str) -> List[Document]:
        """加载 CSV 文件"""
        loader = CSVLoader(
            file_path,
            csv_args={
                'delimiter': ',',
                'quotechar': '"',
                'encoding': 'utf-8'
            }
        )
        docs = loader.load()
        print(f"CSV 加载成功，共 {len(docs)} 行")
        return docs 