# -*- coding: utf-8 -*-
from query_nebulagraph import create_graph_client
import time

# 修改样例数据，使用中文属性名
学生信息 = {
    "属性": {
        "学号": "21001",          # string
        "年级": "大三",           # string
        "绩点": 3.9,              # double
        "入学年份": 2021,         # int
        "预计毕业": 2025          # int
    }
}

学校信息 = {
    "属性": {
        "地点": "北京市海淀区",    # string
        "类型": "综合性大学",      # string
        "建校时间": "1898年",     # string
        "现任校长": "龚旗煌"      # string
    }
}

专业信息 = {
    "属性": {
        "所属院系": "信息科学技术学院",  # string
        "学制": 4,                      # int
        "学位类型": "工学学士",         # string
        "招生人数": 120                 # int
    }
}

课程信息1 = {
    "属性": {
        "学分": 4,                     # int
        "性质": "必修",               # string
        "开课学期": "大二上",         # string
        "周学时": 4,                  # int
        "考核方式": "考试"           # string
    }
}

课程信息2 = {
    "属性": {
        "学分": 4,                     # int
        "性质": "必修",               # string
        "开课学期": "大二下",         # string
        "周学时": 4,                  # int
        "考核方式": "考试"           # string
    }
}

关系列表 = [
    {"关系类型": "就读于", "属性": {
        "入学年份": 2021              # int
    }},
    {"关系类型": "主修", "属性": {
        "入学年份": 2021              # int
    }},
    {"关系类型": "已修课程", "属性": {
        "成绩": "优"                  # string
    }},
    {"关系类型": "在修课程", "属性": {
        "进度": "60%"                 # string
    }},
    {"关系类型": "包含课程", "属性": {
        "学分": 4                      # int
    }},
    {"关系类型": "包含课程", "属性": {
        "学分": 4                      # int
    }}
]

def encode_chinese(text):
    """处理中文编码"""
    return text.encode('utf-8').decode('utf-8')

def create_and_verify_tag(session, tag_name, tag_definition):
    """创建并验证标签"""
    print(f"\nCreating {tag_name} tag...")
    # 对标签定义中的中文属性名进行编码处理
    encoded_definition = tag_definition
    create_query = f'CREATE TAG IF NOT EXISTS `{tag_name}`({encoded_definition})'
    print(f"Executing: {create_query}")
    result = session.execute(create_query)
    if result.error_code() != 0:
        raise Exception(f"Failed to create tag {tag_name}: {result.error_msg()}")
    time.sleep(5)
    
    # 验证标签创建
    print(f"Verifying {tag_name} tag...")
    verify_result = session.execute(f'DESCRIBE TAG `{tag_name}`')
    if verify_result.is_empty():
        raise Exception(f"Tag {tag_name} verification failed")
    print(f"Tag {tag_name} created and verified successfully")
    time.sleep(2)

def setup_schema(session):
    try:
        # 创建图空间
        print("Creating space...")
        session.execute('DROP SPACE IF EXISTS studentinfo')
        session.execute('CREATE SPACE studentinfo(partition_num=1, replica_factor=1, vid_type=FIXED_STRING(30))')
        
        # 等待图空间创建完成
        print("Waiting for the space creation to complete...")
        time.sleep(20)
        
        # 切换到新创建的图空间并验证
        print("Switching to studentinfo space...")
        result = session.execute('USE studentinfo')
        if result.error_code() != 0:
            raise Exception(f"Failed to switch to studentinfo space: {result.error_msg()}")
        
        # 创建标签，每个标签创建后验证
        print("\nCreating tags...")
        
        create_and_verify_tag(session, '学生', 
            '`学号` string, `年级` string, `绩点` double, `入学年份` int, `预计毕业` int')
        
        create_and_verify_tag(session, '学校',
            '`地点` string, `类型` string, `建校时间` string, `现任校长` string')
        
        create_and_verify_tag(session, '专业',
            '`所属院系` string, `学制` int, `学位类型` string, `招生人数` int')
        
        create_and_verify_tag(session, '课程',
            '`学分` int, `性质` string, `开课学期` string, `周学时` int, `考核方式` string')
        
        # 创建边类型
        print("\nCreating edge types...")
        def create_and_verify_edge(edge_name, edge_definition):
            print(f"\nCreating {edge_name} edge...")
            create_query = f'CREATE EDGE IF NOT EXISTS `{edge_name}`({edge_definition})'
            print(f"Executing: {create_query}")
            result = session.execute(create_query)
            if result.error_code() != 0:
                raise Exception(f"Failed to create edge {edge_name}: {result.error_msg()}")
            time.sleep(5)
            
            # 验证边类型创建
            print(f"Verifying {edge_name} edge...")
            verify_result = session.execute(f'DESCRIBE EDGE `{edge_name}`')
            if verify_result.is_empty():
                raise Exception(f"Edge {edge_name} verification failed")
            print(f"Edge {edge_name} created and verified successfully")
            time.sleep(2)
        
        create_and_verify_edge('就读于', '`入学年份` int')
        create_and_verify_edge('主修', '`入学年份` int')
        create_and_verify_edge('已修课程', '`成绩` string')
        create_and_verify_edge('在修课程', '`进度` string')
        create_and_verify_edge('包含课程', '`学分` int')
        
        print("\nSchema setup completed successfully!")
        
    except Exception as e:
        print(f"An error occurred during schema setup: {str(e)}")
        print("Full error details:")
        import traceback
        traceback.print_exc()
        raise e

def verify_insert(session, query):
    """验证插入操作是否成功"""
    print(f"\nExecuting query: {query}")
    try:
        result = session.execute(query)
        print(f"Result error code: {result.error_code()}")
        print(f"Result error message: {result.error_msg()}")
        print(f"Result is empty: {result.is_empty()}")
        if not result.is_empty():
            print(f"Result data: {result.rows()}")
        
        if result.error_code() != 0:
            error_msg = result.error_msg()
            print(f"Insert failed with error code {result.error_code()}: {error_msg}")
            raise Exception(f"Insert failed: {error_msg}")
            
        return result
    except Exception as e:
        print(f"Exception during query execution: {str(e)}")
        raise

def insert_vertex(session, vertex_label, vertex_id, properties):
    try:
        # 打印输入参数
        print(f"\nInserting vertex:")
        print(f"Label: {vertex_label}")
        print(f"ID: {vertex_id}")
        print(f"Properties: {properties}")
        
        # 构建属性列表
        props_list = []
        props_keys = []
        for k, v in properties.items():
            # 对中文属性名进行编码处理
            encoded_key = f"`{encode_chinese(k)}`"
            props_keys.append(encoded_key)
            if isinstance(v, (int, float)):
                props_list.append(str(v))
            elif isinstance(v, str):
                props_list.append(f"'{encode_chinese(v)}'")
            else:
                props_list.append(f"'{str(v)}'")
        
        props_string = ", ".join(props_list)
        props_keys_string = ", ".join(props_keys)
        
        query = f'INSERT VERTEX `{vertex_label}` ({props_keys_string}) VALUES "{vertex_id}":({props_string})'
        
        print(f"\nExecuting vertex insert query:")
        verify_insert(session, query)
        print(f"Successfully inserted vertex {vertex_id}")
        
    except Exception as e:
        print(f"Error inserting vertex {vertex_id}: {str(e)}")
        print("Full error details:")
        import traceback
        traceback.print_exc()
        raise e

def insert_edge(session, src_id, dst_id, edge_type, properties):
    try:
        # 构建属性列表
        props_list = []
        props_keys = []
        for k, v in properties.items():
            props_keys.append(f"`{k}`")  # 使用反引号包裹属性名
            if isinstance(v, (int, float)):
                props_list.append(str(v))  # 数字不需要引号
            elif isinstance(v, str):
                # 处理字符串值中的特殊字符
                v = v.replace("'", "\\'")  # 转义单引号
                props_list.append(f"'{v}'")  # 使用单引号
            else:
                props_list.append(f"'{str(v)}'")
        
        # 构建查询语句
        props_string = ", ".join(props_list)
        props_keys_string = ", ".join(props_keys)
        
        # 使用反引号包裹边类型名和属性名
        query = f'INSERT EDGE `{edge_type}` ({props_keys_string}) VALUES "{src_id}" -> "{dst_id}":({props_string})'
        
        print(f"\nExecuting edge insert query:")
        verify_insert(session, query)
        print(f"Successfully inserted edge from {src_id} to {dst_id}")
        
    except Exception as e:
        print(f"Error inserting edge from {src_id} to {dst_id}: {str(e)}")
        print("Full error details:")
        import traceback
        traceback.print_exc()
        raise e

def main():
    client = create_graph_client()
    session = client.get_session('root', 'nebula')
    try:
        # 设置schema
        setup_schema(session)
        
        # 确保在正确的图空间中
        print("\nSwitching to studentinfo space...")
        session.execute('USE studentinfo')
        
        # 等待schema完全生效
        print("Waiting for schema to take effect...")
        time.sleep(10)
        
        print("\nInserting vertices...")
        insert_vertex(session, "学生", "张三", 学生信息['属性'])
        insert_vertex(session, "学校", "北京大学", 学校信息['属性'])
        insert_vertex(session, "专业", "计算机科学与技术", 专业信息['属性'])
        insert_vertex(session, "课程", "数据结构", 课程信息1['属性'])
        insert_vertex(session, "课程", "算法设计", 课程信息2['属性'])

        # 等待顶点插入完成
        time.sleep(5)

        print("\nInserting edges...")
        insert_edge(session, "张三", "北京大学", "就读于", 关系列表[0]["属性"])
        insert_edge(session, "张三", "计算机科学与技术", "主修", 关系列表[1]["属性"])
        insert_edge(session, "张三", "数据结构", "已修课程", 关系列表[2]["属性"])
        insert_edge(session, "张三", "算法设计", "在修课程", 关系列表[3]["属性"])
        insert_edge(session, "计算机科学与技术", "数据结构", "包含课程", 关系列表[4]["属性"])
        insert_edge(session, "计算机科学与技术", "算法设计", "包含课程", 关系列表[5]["属性"])

        print("\nAll operations completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
    finally:
        session.release()
        client.close()
        print("\nConnection closed.")

if __name__ == "__main__":
    main()
