from itertools import chain

import numpy as np


def displacement_mutation(genome, mutation_rate):
    """
    A sub-tour is selected at random, taken out and
    reinserted at random position.
    :param genome:
    :param mutation_rate:
    :return:
    """
    # todo this can probably be implemented faster
    if np.random.rand() < mutation_rate:
        i_left, i_right = _random_indices(len(genome))
        part = genome[i_left:i_right]
        mutated_genome = np.concatenate([genome[:i_left], genome[i_right:]])
        insert_pos = np.random.randint(0, len(mutated_genome)+1)
        mutated_genome = np.insert(mutated_genome, insert_pos, part)
        return mutated_genome
    return genome


def ordered_crossover(genome_a, genome_b, crossover_rate):
    """
    Ordered crossover ("OX-1")
    First, a sub-tour is selected at random from each route. The two sub-tours
    are at the same position in both routes,
    e.g. from visited city #3 to visited city #8 (0-indexed, high-exclusive).
    parent_a = [8, 4, 7, 3, 6, 2, 5, 1, 9, 0] -> child_a = [3, 6, 2, 5, 1]
    parent_b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> child_b = [3, 4, 5, 6, 7]
    Then, starting from the end position of the sub-tours, looping around to
    the end position of the sub-tours (exclusive), cities are added from the
    opposite parent to the child genome if the city is not yet present
    in the child genomes route. This ensures that a child has two sub-tours,
    one from each parent, and that each sub-tour has the same order
    as that in the parent.
    """
    if np.random.rand() >= crossover_rate:
        return [genome_a, genome_b]
    size = len(genome_a)
    i_left, i_right = _random_indices(size)
    child_a = np.ones(size, dtype='uint16')*-1
    child_b = np.ones(size, dtype='uint16')*-1
    child_a[i_left:i_right] = genome_a[i_left:i_right]
    child_b[i_left:i_right] = genome_b[i_left:i_right]

    # indexes in child to be filled from other parent
    child_chain = np.concatenate((np.arange(i_right, size),
                                  np.arange(0, i_left)))
    # indexes in parent in the order they should be added to the other child
    parent_chain = np.concatenate((np.arange(i_right, size),
                                   np.arange(0, i_right)))

    for chain_i, i in enumerate(child_chain):
        # Add genes from the opposite parent in the same order they appear in the opposite parent
        for j in parent_chain[chain_i:]:
            #  if genome_b[j] not in child_a:
            if not np.any(genome_b[j] == child_a):
                child_a[i] = genome_b[j]
                break
        for j in parent_chain[chain_i:]:
            #  if genome_a[j] not in child_b:
            if not np.any(genome_a[j] == child_b):
                child_b[i] = genome_a[j]
                break

    return [child_a, child_b]


def _random_indices(length):
    n1 = np.random.randint(0, length)
    n2 = np.random.randint(n1+1, length+1)
    return n1, n2
