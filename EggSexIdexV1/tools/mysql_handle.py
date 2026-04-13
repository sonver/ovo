import mysql.connector


class MysqlHandle:

    def __init__(self, host, port, user, passwd, db):
        super().__init__()
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db

    def connect_mysql(self):
        """
        获取mysql连接句柄和游标
        :return: conn 连接句柄  cursor游标
        """
        conn = mysql.connector.connect(host=self.host, port=self.port, user=self.user, password=self.passwd, db=self.db)
        cursor = conn.cursor(dictionary=True)
        return conn, cursor
