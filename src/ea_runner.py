import logging
import pickle
import itertools
import timeit

from matplotlib import pyplot as plt
import numpy as np

from ea_population import TSPPopulation
from utils import Loader
from ea_problem import TSPProblem


class EARunner:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.loader = Loader(False)
        self.run_problem()

    def run_problem(self):
        distances, costs = self.loader.load_dataset_a()
        problem = TSPProblem(distances, costs)
        save_path = str(problem.population_size) + ' ' \
                    + str(problem.generation_limit) + ' ' \
                    + str(problem.crossover_rate) + ' ' \
                    + str(problem.mutation_rate) + ' report'
        self.run(problem, plot=True)
        # self.run(problem, plot=True, save_path="../results/" + save_path)

    def run_true_front(self):
        distances, costs, front = self.loader.load_dataset_b()
        problem = TSPProblem(distances, costs)
        self.run(problem, plot=True, true_front=front)

    def load_results(self):
        paths = ["../results/50 4000 0.7 0.05 report-0.pickle",
                 "../results/100 2000 0.8 0.01 report-0.pickle",
                 "../results/200 1000 0.8 0.05 report-1.pickle"]
        self.load_results_stats(paths)
        self.load_results_plot(paths)

    @staticmethod
    def run(problem, true_front=None, plot=True, save_path=None):
        """
        :param problem:
        :param plot:
        :param true_front: actual optimal front (for comparison with discovered/calculated front)
        :param save_path: Save the first front of the final population to file with the given path
        :return:
        """
        # Generate the initial population
        population = TSPPopulation(problem)
        logging.info("Generations: %s, Pop. size: %s, Cross. rate: %s, Mut. rate: %s",
                     problem.generation_limit,
                     problem.population_size,
                     problem.crossover_rate, problem.mutation_rate)
        fronts = []

        def main_loop():
            while population.generation < problem.generation_limit:
                population.generation += 1
                population.evaluate_fitnesses()  # Calculate total cost and total distance for each route/individual
                population.select_adults()
                population.select_parents()
                population.reproduce()
                if population.generation % (problem.generation_limit / 5) == 0:
                    logging.info("\t\t Generation %s/%s", population.generation, problem.generation_limit)
                    fronts.append(population.get_front(0))

        logging.info("\tExecution time: %s", timeit.timeit(main_loop, number=1))
        logging.info("\t(Min/Max) Distance: %s/%s; Cost: %s/%s",
                     TSPPopulation.min_fitness(population.adults, 0).fitnesses[0],
                     TSPPopulation.max_fitness(population.adults, 0).fitnesses[0],
                     TSPPopulation.min_fitness(population.adults, 1).fitnesses[1],
                     TSPPopulation.max_fitness(population.adults, 1).fitnesses[1])

        if save_path:
            with open(save_path + "-" + str(np.random.randint(10)) + '.pickle', 'wb') as f:
                pickle.dump(population.get_front(0), f)
        if plot:
            EARunner.plot([population.adults], save_path=save_path)
            EARunner.plot([population.get_front(0)],
                          name='Fitnesses, final Pareto-front',
                          save_path=save_path)
            # EARunner.plot(fronts, true_front=true_front, dash=True,
            #              name='Fitnesses, final Pareto-front per 20% progress', save_path=save_path)
            plt.show()

    @staticmethod
    def plot(pools, true_front=None, dash=False, name='Fitnesses', save_path=None):
        """
        :param true_front:
        :param pools: NOT instance of TSPPopulations, but a list of lists of individuals (lists of population.adults)
        :param dash: dash lines between each individual in each pool
        :param name: Plot legend
        :param save_path:
        :return:
        """
        marker = itertools.cycle(('o', ',', '+', '.', '*'))
        color = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))

        if dash:
            linestyle = "--"
            for pool_i in range(len(pools)):
                pools[pool_i] = sorted(pools[pool_i],
                                       key=lambda ind: ind.fitnesses[0])
        else:
            linestyle = ""

        plt.figure()
        plt.title(name)

        for i, pool in enumerate(pools):
            c = next(color)
            plt.plot([individual.fitnesses[0] for individual in pool],
                     [individual.fitnesses[1] for individual in pool],
                     marker=next(marker), linestyle=linestyle, color=c,
                     label=str((i + 1) * 20) + "%-" + str(len(pool))
                           + "sols-" + str(TSPPopulation.n_unique(pool)) + "uniq")
            min_dist = TSPPopulation.min_fitness(pool, 0).fitnesses
            max_dist = TSPPopulation.max_fitness(pool, 0).fitnesses
            min_cost = TSPPopulation.min_fitness(pool, 1).fitnesses
            max_cost = TSPPopulation.max_fitness(pool, 1).fitnesses
            if not dash:
                c = 'r'
            plt.plot([min_dist[0]], [min_dist[1]], marker='D', color=c)
            plt.plot([max_dist[0]], [max_dist[1]], marker='D', color=c)
            plt.plot([min_cost[0]], [min_cost[1]], marker='D', color=c)
            plt.plot([max_cost[0]], [max_cost[1]], marker='D', color=c)
        if true_front is not None:
            plt.plot([i[0] for i in true_front], [i[1] for i in true_front],
                     linestyle="--", label="True front")
            # if dash:
            # plt.legend(loc="best")
        plt.xlabel("Distance")
        plt.xticks(np.arange(30000, 120001, 10000))
        plt.ylabel("Cost")
        plt.yticks(np.arange(300, 1401, 100))
        if save_path:
            plt.savefig(save_path + "-" + str(np.random.randint(10)) + ".png")

    @staticmethod
    def load_results_plot(paths):
        populations = []
        for path in paths:
            with open(path, 'rb') as f:
                population = pickle.load(f)
                populations.append(population)
        EARunner.plot(populations, dash=True,
                      name="Final pareto fronts, 3 configurations")
        plt.show()

    @staticmethod
    def load_results_stats(paths):
        for path in paths:
            with open(path, 'rb') as f:
                population = pickle.load(f)
                logging.info("\t(Min/Max) Distance: %s/%s; Cost: %s/%s",
                             TSPPopulation.min_fitness(population, 0).fitnesses[0],
                             TSPPopulation.max_fitness(population, 0).fitnesses[0],
                             TSPPopulation.min_fitness(population, 1).fitnesses[1],
                             TSPPopulation.max_fitness(population, 1).fitnesses[1])


if __name__ == "__main__":
    runner = EARunner()
