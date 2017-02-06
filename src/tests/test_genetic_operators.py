import numpy as np
from ea_genome import TSPGenome
from ea_problem import TSPProblem
from utils import Loader
from ea_population import TSPPopulation
import unittest
from copy import deepcopy
from ea_genetic_operators import ordered_crossover, displacement_mutation


class TestNSGAUtils(unittest.TestCase):
    n_cities = 48

    def setUp(self):
        loader = Loader(True)
        distances, costs = loader.load_dataset_a()
        self.problem = TSPProblem(distances, costs)
        self.problem.generation_limit = 10
        self.problem.population_size = 100
        population = TSPPopulation(self.problem)
        self.pool = population.children
        self.pool.append(deepcopy(self.pool[0]))

    def test_ordered_crossover(self):
        counter = 0
        for i in range(0, len(self.pool)-1, 2):
            ind_a = self.pool[i].genotype
            ind_b = self.pool[i+1].genotype
            c_a, c_b = ordered_crossover(ind_a, ind_b, 1)
            self.genome_is_valid(c_a)
            self.genome_is_valid(c_b)
            if any(c_a != ind_a) and any(c_b != ind_b):
                counter += 1
        print("Mutated", counter, "pairs")
        assert counter > 0

    def test_ordered_crossover_none(self):
        for i in range(len(self.pool), 2):
            ind_a = self.pool[i].genotype
            ind_b = self.pool[i+1].genotype
            c_a, c_b = ordered_crossover(ind_a, ind_b, 0)
            self.genome_is_valid(c_a)
            self.genome_is_valid(c_b)
            assert all(ind_a == c_a)
            assert all(ind_b == c_b)

    def test_displacement_mutation(self):
        counter = 0
        for i, ind in enumerate(self.pool):
            mut = displacement_mutation(ind.genotype, 1)
            self.genome_is_valid(mut)
            # Assert that mutation does not alter original genome
            if any(mut != ind.genotype):
                counter += 1
        assert counter > 0
        print(counter, " of ", len(self.pool), " was mutated")

    def test_displacement_mutation_none(self):
        for ind in self.pool:
            mut = displacement_mutation(ind.genotype, 0)
            assert all(mut == ind.genotype)

    def genome_is_valid(self, genotype):
        """
        A route is valid if and only if it visits each city exactly once and returns to the origin city
        :return: Boolean
        """
        """
        Since there is a direct path between each and every city (for this problem),
        then it will always be possible to travel directly back to the origin city.
        Therefore, the only requirement for a valid solution is that it only visits each city exactly once.
        """
        assert len(genotype) == self.n_cities, ("Invalid genome length:", len(genotype))
        set_diff = len(np.setdiff1d(np.arange(self.n_cities), genotype, assume_unique=True))
        assert set_diff == 0, ("Genome invalid: ", genotype, " missing values: ", set_diff)
