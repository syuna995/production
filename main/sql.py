'''
Created on 2017/12/13

@author: AMO-TOYAMA
'''

import mysql.connector


class sqlconnector:

    def __init__(self):
        self.connect = mysql.connector.connect(user='****',
                                               password='****',
                                               host='****',
                                               database='****',
                                               charset='utf8')
        self.cursor = self.connect.cursor(dictionary=True)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        self.connect.close()
        self.cursor.close()

    def check(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def change(self, sql):
        self.cursor.execute(sql)
        self.connect.commit()


if __name__ == '__main__':
    db = sqlconnector()

    dblist = db.check('SELECT * FROM calculatedlist WHERE code = 1301')
    for d in dblist:
        print(d)
