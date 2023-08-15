from sklearn.cluster import DBSCAN
import numpy as np
import collections

class Clustering():
    def __init__(self, eps_1=200, eps_2=100):
        self.cluster_1 = DBSCAN(eps=eps_1, min_samples=1)
        self.cluster_2 = DBSCAN(eps=eps_2, min_samples=1)
        self.cluster_3 = DBSCAN(eps=0.5, min_samples=1)


    def update(self, box):
        """
        @dets      t,l,b,r,s
        """
        bboxes = box
        outputs = bboxes.copy()
        outputs[:, 2:4] -= outputs[:, :2]
        areas = outputs[:, 2:4]     # w, h
        # areas[:, 0] = 1
        areas = areas.astype(int)
        outputs[:, 0] = outputs[:, 0] + 0.5 * outputs[:, 2]
        outputs[:, 1] = outputs[:, 1] + 0.5 * outputs[:, 3]
        X = np.copy(outputs[:, :2])     # center_x, center_y
        X = X.astype(int)
        C = self.dbscan_lib(X, areas)

        return C


    def dbscan_lib(self, dataSet, areas):
        """
        @brief      利用sklearn包计算DNSCAN
        @param      dataSet  The data set
        @param      eps      The eps
        @param      minPts   The minimum points
        @return     { description_of_the_return_value }
        """
        if len(dataSet) == 0 or len(areas) == 0:
            return (np.array([]))
        C_2 = self.cluster_2.fit_predict(areas)

        dict_area = {}
        for key in C_2:
            dict_area[key] = dict_area.get(key, 0) + 1
        # optimization single targetg
        index = 0
        for i, a in enumerate(dict_area.values()):
            if a == 1:
                if i == 0:
                    C_2[0] = 2
                elif i == len(C_2)-1:
                    C_2[-1] = C_2[-2]
                else:
                    if areas[index-1, 0]*areas[index-1, 1] - areas[index-2, 0]*areas[index-2, 1] \
                            > areas[index, 0]*areas[index, 0] - areas[index-1, 0]*areas[index-1, 1]:
                        C_2[index-1] = C_2[index]
                    else:
                        C_2[index-1] = C_2[index-2]
            index += a

        dict_area_1 = {}
        for key in C_2:
            dict_area_1[key] = dict_area_1.get(key, 0) + 1
        area_list = []
        dataSet_list = []
        index = 0
        for i in dict_area_1.values():
            area_list.append(areas[index:index+i])
            dataSet_list.append(dataSet[index:index+i])
            index += i

        len_dis = []
        for i, dis in enumerate(area_list):
            a = DBSCAN(eps=int(sum(dis[:, 0])/len(dis)*2), min_samples=1).fit_predict(dataSet_list[i])
            if i >0:
                a += (max(len_dis[i-1])+1)
            len_dis.append(a)

        C_3 = np.hstack(len_dis)
        return C_3


