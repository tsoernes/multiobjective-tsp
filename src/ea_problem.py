from ea_adult_selection import adult_select_pareto_crowding_distance
from ea_genetic_operators import displacement_mutation, ordered_crossover
from ea_genome import TSPGenome
from ea_parent_selection import parent_select_crowding_tournament


class InvalidDatasetError(Exception):
    pass


class TSPProblem:
    def __init__(self, distances, costs):
        if len(distances) != len(costs):
            raise InvalidDatasetError('Length of data sets not equal')

        self.costs = costs
        self.distances = distances
        self.n_cities = len(costs)

        self.population_size = 60
        self.generation_limit = 20

        self.genome = TSPGenome
        self.genome_params = {
        }

        self.adult_select_method = adult_select_pareto_crowding_distance
        self.parent_select_method = parent_select_crowding_tournament
        self.parent_select_params = {
            'tournament_size': 2
        }

        self.mutation_method = displacement_mutation
        self.crossover_method = ordered_crossover
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8


        # mut: 0.001, 0.005, 0.01, 0.05, 0.1
        # cross: 0.5, 0.6, 0.7, 0.8, 0.9

        # best: mut 0.01, cross: 0.8

        # pop/gen: 400/500, 200/1000, 100/2000, 50/4000
        # best: 100/2000