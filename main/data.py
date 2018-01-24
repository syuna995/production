'''
Created on 2017/10/03

@author: AMO-TOYAMA
'''
import csv
import numpy as np
from main.sql import sqlconnector


class databank:

    def __init__(self):
        pass

    def dataset_fromCSV(self):
        train_x = list()
        train_y = list()
        test_x = list()
        test_y = list()

        with open('../input/train.csv', 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                tmp = list()
                for i in range(len(row)):
                    if i == 100:
                        train_y.append(float(row[i]))
                    else:
                        tmp.append(float(row[i]))
                train_x.append(np.array(tmp))

        with open('../input/test.csv', 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                tmp = list()
                for i in range(len(row)):
                    if i == 100:
                        test_y.append(float(row[i]))
                    else:
                        tmp.append(float(row[i]))
                test_x.append(np.array(tmp))

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        return train_x[::-1], train_y[::-1], test_x[::-1], test_y[::-1]

    def dataset_fromSQL(self):
        train_x = list()
        train_y = list()
        test_x = list()
        test_y = list()

        with sqlconnector() as sql:
            codelist = sql.check(
                'SELECT code FROM calculatedlist GROUP BY code LIMIT 11')
            print(codelist)

        with sqlconnector() as sql:
            for i in range(1, len(codelist)):
                plist = sql.check(
                    'select * from calculatedlist where code = ' + str(codelist[i]['code']) + ' \
                    and date >= "2000-01-01" and date <= "2014-12-31"')

                zscored = list()
                for i in range(len(plist)):
                    tmplist = list()
                    tmplist.append(plist[i]['open'])
                    tmplist.append(plist[i]['high'])
                    tmplist.append(plist[i]['low'])
                    tmplist.append(plist[i]['close'])
                    tmplist.append(plist[i]['in5'])
                    tmplist.append(plist[i]['in6'])
                    tmplist.append(plist[i]['in7'])
                    tmplist.append(plist[i]['in8'])
                    tmplist.append(plist[i]['in9'])
                    tmplist.append(plist[i]['in10'])
                    tmplist.append(plist[i]['in11'])
                    tmplist.append(plist[i]['in12'])
                    tmplist.append(plist[i]['in13'])
                    tmplist.append(plist[i]['in14'])
                    tmplist.append(plist[i]['in15'])
                    tmplist.append(plist[i]['in16'])
                    tmplist.append(plist[i]['in17'])
                    tmplist.append(plist[i]['in18'])
                    tmplist.append(plist[i]['in19'])
                    tmplist.append(plist[i]['in20'])
                    tmplist.append(plist[i]['in21'])
                    tmplist.append(plist[i]['in22'])
                    tmplist.append(plist[i]['in23'])
                    tmplist.append(plist[i]['in24'])
                    tmplist.append(plist[i]['in25'])
                    tmplist.append(plist[i]['in26'])
                    tmplist.append(plist[i]['in27'])
                    tmplist.append(plist[i]['in28'])
                    tmplist.append(plist[i]['in29'])
                    tmplist.append(plist[i]['in30'])
                    tmplist.append(plist[i]['in31'])
                    tmplist.append(plist[i]['in32'])
                    tmplist.append(plist[i]['in33'])
                    tmplist.append(plist[i]['in34'])
                    tmplist.append(plist[i]['in35'])
                    tmplist.append(plist[i]['in36'])
                    zscored.append(np.array(tmplist))

                zscored = np.array(zscored)
                mean = np.mean(zscored, 0)
                std = np.std(zscored, 0)
                for i in range(len(zscored)):
                    zscored[i, 0:] = (zscored[i, 0:] - mean) / std

                for i in range(len(zscored) - 20):
                    train_x.append(zscored[i:i + 20, 0:])
                    train_y.append(zscored[i + 20, 3])

            del zscored

            plist = sql.check(
                'select * from calculatedlist where code = ' + str(codelist[0]['code']) + ' \
                    and date >= "2000-01-01" and date <= "2014-12-31"')

            zscored = list()
            for i in range(len(plist)):
                tmplist = list()
                tmplist.append(plist[i]['open'])
                tmplist.append(plist[i]['high'])
                tmplist.append(plist[i]['low'])
                tmplist.append(plist[i]['close'])
                tmplist.append(plist[i]['in5'])
                tmplist.append(plist[i]['in6'])
                tmplist.append(plist[i]['in7'])
                tmplist.append(plist[i]['in8'])
                tmplist.append(plist[i]['in9'])
                tmplist.append(plist[i]['in10'])
                tmplist.append(plist[i]['in11'])
                tmplist.append(plist[i]['in12'])
                tmplist.append(plist[i]['in13'])
                tmplist.append(plist[i]['in14'])
                tmplist.append(plist[i]['in15'])
                tmplist.append(plist[i]['in16'])
                tmplist.append(plist[i]['in17'])
                tmplist.append(plist[i]['in18'])
                tmplist.append(plist[i]['in19'])
                tmplist.append(plist[i]['in20'])
                tmplist.append(plist[i]['in21'])
                tmplist.append(plist[i]['in22'])
                tmplist.append(plist[i]['in23'])
                tmplist.append(plist[i]['in24'])
                tmplist.append(plist[i]['in25'])
                tmplist.append(plist[i]['in26'])
                tmplist.append(plist[i]['in27'])
                tmplist.append(plist[i]['in28'])
                tmplist.append(plist[i]['in29'])
                tmplist.append(plist[i]['in30'])
                tmplist.append(plist[i]['in31'])
                tmplist.append(plist[i]['in32'])
                tmplist.append(plist[i]['in33'])
                tmplist.append(plist[i]['in34'])
                tmplist.append(plist[i]['in35'])
                tmplist.append(plist[i]['in36'])
                zscored.append(np.array(tmplist))

            zscored = np.array(zscored)
            mean = np.mean(zscored, 0)
            std = np.std(zscored, 0)

            del zscored

            plist = sql.check(
                'select * from calculatedlist where code = ' + str(codelist[0]['code']) + ' and \
                date >= "2015-01-01" and date <= "2016-12-31"')

            zscored = list()
            for i in range(len(plist)):
                tmplist = list()
                tmplist.append(plist[i]['open'])
                tmplist.append(plist[i]['high'])
                tmplist.append(plist[i]['low'])
                tmplist.append(plist[i]['close'])
                tmplist.append(plist[i]['in5'])
                tmplist.append(plist[i]['in6'])
                tmplist.append(plist[i]['in7'])
                tmplist.append(plist[i]['in8'])
                tmplist.append(plist[i]['in9'])
                tmplist.append(plist[i]['in10'])
                tmplist.append(plist[i]['in11'])
                tmplist.append(plist[i]['in12'])
                tmplist.append(plist[i]['in13'])
                tmplist.append(plist[i]['in14'])
                tmplist.append(plist[i]['in15'])
                tmplist.append(plist[i]['in16'])
                tmplist.append(plist[i]['in17'])
                tmplist.append(plist[i]['in18'])
                tmplist.append(plist[i]['in19'])
                tmplist.append(plist[i]['in20'])
                tmplist.append(plist[i]['in21'])
                tmplist.append(plist[i]['in22'])
                tmplist.append(plist[i]['in23'])
                tmplist.append(plist[i]['in24'])
                tmplist.append(plist[i]['in25'])
                tmplist.append(plist[i]['in26'])
                tmplist.append(plist[i]['in27'])
                tmplist.append(plist[i]['in28'])
                tmplist.append(plist[i]['in29'])
                tmplist.append(plist[i]['in30'])
                tmplist.append(plist[i]['in31'])
                tmplist.append(plist[i]['in32'])
                tmplist.append(plist[i]['in33'])
                tmplist.append(plist[i]['in34'])
                tmplist.append(plist[i]['in35'])
                tmplist.append(plist[i]['in36'])
                zscored.append(np.array(tmplist))

            zscored = np.array(zscored)
            for i in range(len(zscored)):
                zscored[i, 0:] = (zscored[i, 0:] - mean) / std

            for i in range(len(zscored) - 20):
                test_x.append(zscored[i:i + 20, 0:])
                test_y.append(zscored[i + 20, 3])

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    db = databank()
    train_x, train_y, test_x, test_y = db.dataset_fromSQL()
    print(__name__)
    pass
