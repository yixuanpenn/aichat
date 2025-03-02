from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.common import ttypes

def create_graph_client():
    """创建并返回 NebulaGraph 客户端"""
    config = Config()
    config.max_connection_pool_size = 10
    # 使用提供的URL，假设URL已经包含了正确的端口映射
    connection_pool = ConnectionPool()
    # 注意：这里我们假设代理服务已经处理了端口映射，因此不需要指定端口号
    if not connection_pool.init([('27.tcp.cpolar.top', 12459)], config):
        raise Exception("Failed to connect to NebulaGraph")
    return connection_pool

def query_data(client):
    """查询图谱数据"""
    session = client.get_session('root', 'nebula')  # 使用提供的用户名和密码
    try:
        # 切换到demo_movie_recommendation数据库
        session.execute("USE demo_movie_recommendation;")
        
        # 执行查询，这里需要根据实际情况修改查询语句
        # 假设我们想要查询所有电影的标题和发布年份
        result = session.execute("MATCH ()-[e]->() RETURN e LIMIT 10;")
        if result is None or result.is_empty():
            print("No data returned from the query.")
            return
        print("Query results:")
        for record in result:
            # 使用正确的方法访问Record对象中的数据
            print(f"结果: {record}")
    finally:
        session.release()

def main():
    client = create_graph_client()
    query_data(client)
    client.close()

if __name__ == "__main__":
    main()