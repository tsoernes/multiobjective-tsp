from itertools import chain

from nsga_utils import rank_assign_sort, crowding_distance_assign


def adult_select_pareto_crowding_distance(children, parents, f_mins, f_maxes):
    """
    Select adults from a composite pool of children and their parents
    (from the previous generation) based on first on rank and second on
    crowding distance. This ensures elitism.
    """
    # The number of parents is 0 the first iteration. So, assume that the child
    # pool has the same size as the parent pool.
    n_adults = len(children)
    # Combine children and adults into same pool. Sort the pool into non-dominated fronts.
    pool = rank_assign_sort(list(chain(parents, children)))
    adults = []
    rank_indexes = []
    for non_dominated_front in pool:
        # Crowding distance needs to be calculated for all fronts since it will later be used for tournament selection.
        crowding_distance_assign(non_dominated_front, f_mins, f_maxes)

        # Add non-dominated fronts to empty adult pool until no more complete fronts can be added.
        if len(adults) + len(non_dominated_front) <= n_adults:
            adults.extend(non_dominated_front)
            rank_indexes.append(len(adults))
        # Then, add individuals from the last front based on their crowding distance.
        else:
            non_dominated_front = sorted(non_dominated_front,
                                         key=lambda ind: ind.crowding_distance,
                                         reverse=True)
            adults.extend(non_dominated_front[:n_adults - len(adults)])
            rank_indexes.append(len(adults))
            break
    return adults, rank_indexes
