from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

def create_and_check_schema():
    # 配置连接池
    config = Config()
    config.max_connection_pool_size = 10
    print("Configuring connection pool...")
    # NebulaGraph服务的地址和端口
    connection_pool = ConnectionPool()
    if not connection_pool.init([('27.tcp.cpolar.top', 12459)], config):
        print("Failed to connect to NebulaGraph")
        return
    else:
        print("Connected to NebulaGraph successfully.")

    # 获取session
    session = connection_pool.get_session('root', 'nebula')
    if session:
        print("Session created successfully.")
    else:
        print("Failed to create session.")
        return

    try:
        # 检查图空间是否已存在
        print("Checking if the 'studentinfo' graph space already exists...")
        show_spaces = "SHOW SPACES;"
        spaces_result = session.execute(show_spaces)
        if spaces_result.is_empty():
            print("No spaces found.")
            return

        existing_spaces = {record.values[0].as_string() for record in spaces_result}
        if 'studentinfo' in existing_spaces:
            print("Graph space 'studentinfo' already exists.")
        else:
            # 创建图空间
            print("Creating graph space 'studentinfo'...")
            session.execute("CREATE SPACE IF NOT EXISTS studentinfo (partition_num=10, replica_factor=1, vid_type=fixed_string(30));")
            session.execute("USE studentinfo;")
            print("Graph space 'studentinfo' created.")

        # 创建顶点类型
        print("Creating vertex types...")
        session.execute("""
        CREATE TAG IF NOT EXISTS Student(
            student_id string, 
            grade string, 
            gpa double, 
            enrollment_year int, 
            expected_graduation int
        );
        CREATE TAG IF NOT EXISTS School(location string, type string, established string, current_principal string);
        CREATE TAG IF NOT EXISTS Major(department string, duration int, degree_type string, enrollment int);
        CREATE TAG IF NOT EXISTS Course(credits int, nature string, semester string, weekly_hours int, assessment_method string);
        """)
        print("Vertex types created.")

        # 创建边类型
        print("Creating edge types...")
        session.execute("""
        CREATE EDGE IF NOT EXISTS StudiesAt(enrollment_year int);
        CREATE EDGE IF NOT EXISTS MajorsIn(enrollment_year int);
        CREATE EDGE IF NOT EXISTS CompletedCourse(grade string);
        CREATE EDGE IF NOT EXISTS EnrolledCourse(progress string);
        CREATE EDGE IF NOT EXISTS IncludesCourse(credits int);
        """)
        print("Edge types created.")

        print("Schema creation attempted in 'studentinfo' space.")

        # 查询顶点类型
        print("Querying vertex types...")
        show_tags = "SHOW TAGS;"
        tag_result = session.execute(show_tags)
        if tag_result.is_empty():
            print("No tags found in 'studentinfo'.")
        else:
            print("Tags in 'studentinfo':")
            for record in tag_result:
                print(record.values[0].as_string())

        # 查询边类型
        print("Querying edge types...")
        show_edges = "SHOW EDGES;"
        edge_result = session.execute(show_edges)
        if edge_result.is_empty():
            print("No edges found in 'studentinfo'.")
        else:
            print("Edges in 'studentinfo':")
            for record in edge_result:
                print(record.values[0].as_string())

    except Exception as e:
        print(f"Failed to manage schema in 'studentinfo': {str(e)}")
    finally:
        session.release()
        print("Session released.")
        connection_pool.close()
        print("Connection pool closed.")

if __name__ == "__main__":
    create_and_check_schema()