
def rank_assign_sort(pool):
    """
    Very fast Non-Dominant sort with binary insertion as per
    NSGA-II, Jensen 2003.
    Assign ranks to the members of the pool and sort it by rank,
    e.g. member of the n'th non-dominated front (ascending; lower is better)
    :param pool:
    :return:
    """
    # Sort pool in ascending order by fitness function 1, and if two solutions
    # has the same fitness for fitness function 1,
    # then sort in ascending order by fitness function 2
    pool = sorted(pool)

    fronts = [[pool[0]]]
    # The individual with the lowest value for fitness func. 1 must belong to the first front
    pool[0].rank = 0
    current_rank = 0
    for ind_a in pool[1:]:
        for ind_b in fronts[current_rank]:
            if ind_b.dominates(ind_a):
                current_rank += 1
                fronts.append([ind_a])
                ind_a.rank = current_rank
                break
        else:
            # Find the lowest front, index/rank b, where ind_a is not dominated.
            b = _bisect_fronts(fronts, ind_a)
            fronts[b].append(ind_a)
            ind_a.rank = b

    return fronts


def _bisect_fronts(fronts, ind):
    """
    :param fronts:
    :param ind:
    :return:
    """
    lo = 0
    hi = len(fronts)
    while lo < hi:
        mid = (lo+hi)//2
        if not fronts[mid][-1].dominates(ind):
            hi = mid
        else:
            lo = mid+1
    return lo


def crowding_distance_assign(non_dominated_front, f_mins, f_maxes):
    """
    Assign crowding distance to the individuals in a non-dominated front
    and return a sorted front
    :param non_dominated_front:
    :return: The non-dominated front sorted by crowding distance in
    descending order (higher is better)
    """
    for individual in non_dominated_front:
        individual.crowding_distance = 0

    for objective_i in range(len(non_dominated_front[0].fitnesses)):
        # Sort front by current fitness function
        objective_sorted = sorted(non_dominated_front,
                                  key=lambda ind: ind.fitnesses[objective_i])
        objective_sorted[0].crowding_distance = float('inf')
        objective_sorted[-1].crowding_distance = float('inf')
        #fitness_factor = objective_sorted[-1].fitnesses[objective_i] - objective_sorted[0].fitnesses[objective_i]
        fitness_factor = f_maxes[objective_i] - f_mins[objective_i]
        for individual_i in range(1, len(non_dominated_front)-1):
            prev_ind = objective_sorted[individual_i-1].fitnesses[objective_i]
            next_ind = objective_sorted[individual_i+1].fitnesses[objective_i]
            scaled_dist = (next_ind - prev_ind) / fitness_factor
            objective_sorted[individual_i].crowding_distance += scaled_dist
