import numpy as np
from itertools import islice


class TSPPopulation:
    """
    A population of potential solutions (i.e. individuals)
    """

    def __init__(self, problem):
        self.problem = problem
        self.distances = problem.distances
        self.costs = problem.costs
        self.n_cities = len(self.distances)
        self.population_size = problem.population_size
        self.genome = problem.genome
        self.crossover = problem.crossover_method
        self.mutate = problem.mutation_method
        self.crossover_rate = problem.crossover_rate
        self.mutation_rate = problem.mutation_rate
        self.generation = 0
        self.children = [self.problem.genome(self.n_cities)
                         for _ in range(self.population_size)]
        self.adults = []
        self.parents = []
        # A list of indexes of the end position for each rank, such that
        # adults[:rank_indexes[0]] will yield the first front,
        # adults[rank_indexes[0]:rank_indexes[1]] the second and so forth
        self.rank_indexes = []
        self.f_mins = [float('inf'), float('inf')]
        self.f_maxes = [float('-inf'), float('-inf')]

    def evaluate_fitnesses(self):
        """
        Evaluate the fitnesses of the phenotypes.
        If the evaluation is a stochastic process, then adults should also be
        evaluated each run, in order to weed out phenotypes with
        a lucky evaluation.
        """
        for child in self.children:
            self.evaluate_fitness(child)
            for fitness_i in range(len(child.fitnesses)):
                f = child.fitnesses[fitness_i]
                if f < self.f_mins[fitness_i]:
                    self.f_mins[fitness_i] = f
                if f > self.f_maxes[fitness_i]:
                    self.f_maxes[fitness_i] = f

    def select_adults(self):
        self.adults, self.rank_indexes = self.problem.adult_select_method(self.children,
                                                                          self.adults,
                                                                          self.f_mins,
                                                                          self.f_maxes)

    def select_parents(self):
        """
        Select adults to become parents, e.g. to mate.
        """
        self.parents = self.problem.parent_select_method(self.adults,
                                                         **self.problem.parent_select_params)

    def reproduce(self):
        """
        Generate children from the selected parents by first
        crossing genes then mutating
        """
        # An individual can reproduce with itself. Probably not optimal.
        '''
        self.children = \
            [self.genome(self.n_cities, genotype=self.mutate(child_genome, self.problem.mutation_rate))
                for parent_a, parent_b in zip(islice(self.parents, 0, None, 2), islice(self.parents, 1, None, 2))
                for child_genome in self.crossover(parent_a.genotype, parent_b.genotype, self.crossover_rate)]
        '''
        # untested refactoring
        self.children = []
        couples = zip(islice(self.parents, 0, None, 2),
                      islice(self.parents, 1, None, 2))
        for parent_a, parent_b in couples:
            crossed_genomes = self.crossover(parent_a.genotype,
                                             parent_b.genotype,
                                             self.crossover_rate)
            for crossed_genome in crossed_genomes:
                mutated_genome = self.mutate(crossed_genome,
                                             self.problem.mutation_rate)
                self.children.append(self.genome(self.n_cities,
                                                 mutated_genome))

    def evaluate_fitness(self, child):
        total_distance = 0
        total_cost = 0
        for i in range(self.n_cities-1):
            city_a = child.genotype[i]
            city_b = child.genotype[i+1]
            #  The cost of travelling from a to b is equal to the cost of travelling from b to a
            total_distance += self.distances[city_a][city_b]
            total_cost += self.costs[city_a][city_b]

        child.fitnesses[0] = total_distance
        child.fitnesses[1] = total_cost

    @property
    def n_fronts(self):
        return len(self.rank_indexes)

    def get_front(self, rank):
        if rank >= 1:
            start = self.rank_indexes[rank - 1]
        else:
            start = 0
        return self.adults[start:self.rank_indexes[rank]]

    @staticmethod
    def area_metric(front):
        """
        Return a metric for the 'total fitness' of a pareto front. This metric
        can only be compared to the area metric
        of other fronts if the f_mins and f_maxes are the same.

        For two dimensional pareto fronts, the normalized area under the two
        pareto frontiers is a very nice metric.
        This means that whenever one Pareto front approximation dominates
        another, the are of the former is less (if both fitness functions are
        to be minimized) than that of the latter.
        """
        pass

    @staticmethod
    def min_fitness(pool, fitness_func_i):
        return min(pool, key=lambda i: i.fitnesses[fitness_func_i])

    @staticmethod
    def max_fitness(pool, fitness_func_i):
        return max(pool, key=lambda i: i.fitnesses[fitness_func_i])

    @staticmethod
    def n_unique(front):
        """
        Number of solutions with different fitness in the given front.

        :param front:
        :return:
        """
        a = np.asarray([ind.fitnesses for ind in front])
        unique = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))) \
            .view(a.dtype) \
            .reshape(-1, a.shape[1])
        return len(unique)
