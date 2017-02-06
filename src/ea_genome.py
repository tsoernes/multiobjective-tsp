import numpy as np


class TSPGenome:
    __slots__ = ['n_cities', 'fitnesses', 'dominates_list',
                 'inverse_domination_count', 'rank', 'crowding_distance',
                 'genotype']

    """
    A 1-D list of the cities to visit (in order)
    """
    n_objectives = 2

    def __init__(self, n_cities, genotype=None):
        self.n_cities = n_cities
        self.fitnesses = np.zeros(self.n_objectives, dtype='uint32')
        self.dominates_list = None  # List of individuals that this individual dominates
        self.inverse_domination_count = float('-inf')  # Number of individuals that dominate this individual
        self.rank = -1  # Member of the n'th pareto front; 0 being the best
        self.crowding_distance = -1

        if genotype is None:
            self.genotype = np.random.permutation(
                np.arange(self.n_cities, dtype='uint8'))
        else:
            self.genotype = genotype

    def dominates(self, individual_b):
        """
        Individual_a dominates individual_f if both:
            a is no worse than b in regards to all fitnesses
            a is strictly better than b in regards to at least one fitness
        Assumes that lower fitness is better, as is the case with cost-distance-TSP.
        :param self:
        :param individual_b:
        :return: True if individual_a dominates individual_b
                 False if individual_b dominates individual_a or neither dominate each other
        """
        a_no_worse_b = 0  # a <= b
        a_strictly_better_b = 0  # a < b
        n_objectives = len(self.fitnesses)
        for fitness_i in range(n_objectives):
            f_a = self.fitnesses[fitness_i]
            f_b = individual_b.fitnesses[fitness_i]
            if f_a < f_b:
                a_no_worse_b += 1
                a_strictly_better_b += 1
            elif f_a == f_b:
                a_no_worse_b += 1
            else:
                return False
        return a_no_worse_b == n_objectives and a_strictly_better_b >= 1

    def __lt__(self, other):
        """ Even though A < B, that does not indicate that A.dominates(B),
        as A may have a lower value for fit. func. 1 but greater value for
        fit. func 2 and therefore neither dominate each other. """
        for i in range(len(self.fitnesses)):
            if self.fitnesses[i] < other.fitnesses[i]:
                return True
            if self.fitnesses[i] > other.fitnesses[i]:
                return False
        return False

    def __gt__(self, other):
        for i in range(len(self.fitnesses)):
            if self.fitnesses[i] > other.fitnesses[i]:
                return True
            if self.fitnesses[i] < other.fitnesses[i]:
                return False
        return False

    def __eq__(self, other):
        for i in range(len(self.fitnesses)):
            if self.fitnesses[i] != other.fitnesses[i]:
                return False
        return True

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __ne__(self, other):
        for i in range(len(self.fitnesses)):
            if self.fitnesses[i] != other.fitnesses[i]:
                return True
        return False
