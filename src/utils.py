import numpy as np


class Loader:
    n_cities_a = 48
    n_cities_b = 100

    def __init__(self, unit_test=False):
        self.path_cost = "./../datasets/cost.txt"
        self.path_dist = "./../datasets/distance.txt"
        self.o1_path = './../datasets/euclidA100.tsp'
        self.o2_path = './../datasets/euclidB100.tsp'
        self.front_path = './../datasets/best.euclidAB100.tsp'

        if unit_test:
            self.path_cost = '.' + self.path_cost
            self.path_dist = '.' + self.path_dist
            self.o1_path = '.' + self.o1_path
            self.o2_path = '.' + self.o2_path
            self.front_path = '.' + self.front_path

    def load_dataset_a(self):
        distances = np.zeros((self.n_cities_a, self.n_cities_a), dtype='uint16')
        costs = np.zeros((self.n_cities_a, self.n_cities_a), dtype='uint16')
        with open(self.path_dist, "rt") as in_file:
            dist_t = in_file.read()
        with open(self.path_cost, "rt") as in_file:
            cost_t = in_file.read()
        dist_rows = dist_t.split("\n")
        cost_rows = cost_t.split("\n")

        for i in range(self.n_cities_a):
            dist_cols = dist_rows[i].split("\t")[:i + 1]
            cost_cols = cost_rows[i].split("\t")[:i + 1]
            distances[i][:i + 1] = dist_cols
            costs[i][:i + 1] = cost_cols

        # Mirror arrays by diagonal
        for i in range(self.n_cities_a):
            for j in range(i, self.n_cities_a):
                distances[i][j] = distances[j][i]
                costs[i][j] = costs[j][i]
        return distances, costs

    def load_dataset_b(self):
        o1_cords = np.zeros((self.n_cities_b, 2), dtype='int16')
        o2_cords = np.zeros((self.n_cities_b, 2), dtype='int16')
        front = np.zeros((1719, 2), dtype='uint32')
        with open(self.o1_path, "rt") as in_file:
            o1_file = in_file.read()
        with open(self.o2_path, "rt") as in_file:
            o2_file = in_file.read()
        with open(self.front_path, "rt") as in_file:
            front_t = in_file.read()
        o1_cords_rows = o1_file.split("\n")[6:]
        o2_cords_rows = o2_file.split("\n")[6:]
        front_rows = front_t.split("\n")
        for i in range(self.n_cities_b):
            o1_cords[i] = o1_cords_rows[i].split(" ")[1:]
            o2_cords[i] = o2_cords_rows[i].split(" ")[1:]
        for i in range(1719):
            front[i] = front_rows[i].split("\t")

        o1 = np.zeros((self.n_cities_b, self.n_cities_b), dtype='uint64')
        o2 = np.zeros((self.n_cities_b, self.n_cities_b), dtype='uint64')

        for i in range(len(o1_cords)):
            o1_city_a_coord = o1_cords[i]
            o2_city_a_coord = o2_cords[i]
            for j in range(i+1, len(o1_cords)):
                o1_city_b_coord = o1_cords[j]
                o2_city_b_coord = o2_cords[j]
                o1_dist = np.sqrt((o1_city_a_coord[0]-o1_city_b_coord[0])**2 +
                                  (o1_city_a_coord[1]-o1_city_b_coord[1])**2)
                o2_dist = np.sqrt((o2_city_a_coord[0]-o2_city_b_coord[0])**2 +
                                  (o2_city_a_coord[1]-o2_city_b_coord[1])**2)
                o1[i][j] = o1_dist
                o1[j][i] = o1_dist
                o2[i][j] = o2_dist
                o2[j][i] = o2_dist

        return o1, o2, front
