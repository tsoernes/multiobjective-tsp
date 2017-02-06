from ea_genome import TSPGenome
from ea_problem import TSPProblem
from utils import Loader
from ea_population import TSPPopulation
from nsga_utils import rank_assign_sort, rank_assign_sort2
import unittest
import time
from copy import deepcopy
import numpy as np
from itertools import chain


def time_fn(fn, *args, **kwargs):
    start = time.clock()
    results = fn(*args, **kwargs)
    end = time.clock()
    fn_name = fn.__module__ + "." + fn.__name__
    print(fn_name + ": " + str(end-start) + "s")
    return results


class TestNSGAUtils(unittest.TestCase):
    def setUp(self):
        print("Setting up")
        loader = Loader(True)
        distances, costs, front = loader.load_dataset_b()
        self.problem = TSPProblem(distances, costs)
        self.problem.generation_limit = 10
        self.problem.population_size = 1000
        population = TSPPopulation(self.problem)
        population.evaluate_fitnesses()
        self.pool = population.children
        self.pool.append(deepcopy(self.pool[0]))
        self.pool.append(deepcopy(self.pool[0]))
        self.pool.append(deepcopy(self.pool[0]))

    def test_rank_assign_sort(self):
        print("Testing rank sort")

        pool_sorted_a = time_fn(rank_assign_sort2, deepcopy(self.pool))
        TestNSGAUtils.check_ranking(pool_sorted_a)
        self.check_rank_len(pool_sorted_a)
        TestNSGAUtils.check_rank_att(pool_sorted_a)
        TestNSGAUtils.check_same_fitness_same_rank(pool_sorted_a)

    @staticmethod
    def check_same_fitness_same_rank(pool_sorted):
        """
        Individuals with equal fitness should have the same rank
        :param pool_sorted:
        :return:
        """
        # Flatten rank sorted array and keep reconstruction indexes
        flattened = [ind.fitnesses
                     for front in pool_sorted
                     for ind in front]
        reconstruct_indexes = [(i, j)
                               for i, front in enumerate(pool_sorted)
                               for j in range(len(front))]

        # Find individuals with equal fitnesses
        a = np.asarray(flattened)
        all_arr = np.concatenate(a).reshape(-1, a[0].size)
        ids = np.ravel_multi_index(all_arr.T, all_arr.max(0) + 1)
        _, unqids, counts = np.unique(ids, return_inverse=True, return_counts=True)
        sidx = unqids.argsort()
        mask = np.in1d(unqids, np.where(counts > 1)[0])
        out = np.split(sidx[mask[sidx]], counts[counts > 1].cumsum())[:-1]

        for same_fitness in out:
            for i, ind_a in enumerate(same_fitness):
                for ind_b in same_fitness[i+1:]:
                    y_a, x_a = reconstruct_indexes[ind_a]
                    y_b, x_b = reconstruct_indexes[ind_b]
                    assert pool_sorted[y_a][x_a].rank == pool_sorted[y_b][x_b].rank

    @staticmethod
    def check_ranking(pool_sorted):
        for i, front_a in enumerate(pool_sorted):
            for ind_a_i in range(len(front_a)):
                ind_a = front_a[ind_a_i]

                for front_b in pool_sorted[i:]:
                    for ind_b in front_b:
                        # Ind_a should not be dominated by any individual in the same or worse fronts
                        assert not ind_b.dominates(ind_a), (ind_a.fitnesses, ind_b.fitnesses, ind_a.rank, ind_b.rank)
                for ind_b in pool_sorted[i]:
                    # Ind_a should not dominate any individual in the same front
                    assert not ind_a.dominates(ind_b), (ind_a.fitnesses, ind_b.fitnesses, ind_a.rank, ind_b.rank)
        for i in range(len(pool_sorted)-1, 1, -1):
            for ind_a in pool_sorted[i]:
                dom_count = 0
                for ind_b in pool_sorted[i-1]:
                    if ind_b.dominates(ind_a):
                        dom_count += 1
                # Every individual a should be dominated by at least one member b of the next best front
                assert dom_count > 0, ("Rank", ind_a.rank, "i", i, len(pool_sorted), ind_a.fitnesses,
                                       [b.fitnesses for b in pool_sorted[i-1]])

    def check_rank_len(self, pool_sorted):
        count = sum(len(front) for front in pool_sorted)
        assert count == len(self.pool)

    @staticmethod
    def check_rank_att(pool_sorted):
        for i, front_a in enumerate(pool_sorted):
            for j, ind_a in enumerate(front_a):
                # Rank attribute should correspond to the index of pool_sorted
                assert ind_a.rank == i, (ind_a.rank, i, j, len(front_a))
                # Individuals within the same front should have the same rank
                for ind_b in front_a[j:]:
                    assert ind_a.rank == ind_b.rank, (ind_a.rank, ind_b.rank)
                # Individuals in lesser fronts should have lesser rank
                for front_b in pool_sorted[i+1:]:
                    for ind_b in front_b:
                        assert ind_a.rank < ind_b.rank, (ind_a.rank, ind_b.rank)

    def check_crowding_distance(self, pool, rank_indexes):
        # Check crowding distance sorting within ranks
        for rank in range(len(rank_indexes)):
            front = self.get_front(pool, rank_indexes, rank)
            for j in range(len(front)-1):
                assert front[j].crowding_distance >= front[j+1].crowding_distance

    @staticmethod
    def get_front(pool, rank_indexes, rank):
        if rank >= 1:
            start = rank_indexes[rank - 1]
        else:
            start = 0
        return pool[start:rank_indexes[rank]]
