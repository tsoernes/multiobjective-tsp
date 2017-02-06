import numpy as np


def parent_select_crowding_tournament(adults, tournament_size=2):
    """
    Parent selection based on the crowded-comparison operator.
    Creates as many parents as there are adults.
    :param adults:
    :param tournament_size:
    :return:
    """
    parents = []
    n = len(adults)
    for i in range(n):
        group = np.random.choice(adults, size=tournament_size)
        best = group[0]
        for individual in group[1:]:
            if individual.rank < best.rank:
                best = individual
            elif individual.rank == best.rank and \
                    individual.crowding_distance > best.crowding_distance:
                best = individual
        parents.append(best)

    return parents
