from itertools import chain
from timeit import timeit
import logging
import bisect

import numpy as np


class Number:
    def __init__(self, value=None):
        if value:
            self.value = value
        else:
            self.value = np.random.randint(10)
        self.rank = -1
        self.strictly_better_than_list = None
        self.strictly_worse_than_count = -1

        self.distance = np.random.randint(10)

    def strictly_better_than(self, other):
        # Placeholder for similar inexpensive computation
        if self.value < other.value:
            return True
        return False


def assign_sort_rank(numbers):
    # Needs flat list
    #assert isinstance(numbers[0], Number)

    rank_sorted = [[]]
    for number_a in numbers:
        assert type(number_a) == Number, type(number_a)
        number_a.strictly_better_than_list = []
        number_a.strictly_worse_than_count = 0
        for number_b in numbers:
            assert type(number_b) == Number, type(number_b)
            if number_a.strictly_better_than(number_b):
                number_a.strictly_better_than_list.append(number_b)
            elif number_b.strictly_better_than(number_a):
                number_a.strictly_worse_than_count += 1
        if number_a.strictly_worse_than_count == 0:
            number_a.rank = 0
            rank_sorted[0].append(number_a)
    i = 0
    while rank_sorted[i]:
        current_front = rank_sorted[i]
        next_front = []
        for number_a in current_front:
            for number_b in number_a.strictly_better_than_list:
                number_b.strictly_worse_than_count -= 1
                assert number_b.strictly_worse_than_count >= 0, number_b.strictly_worse_than_count
                if number_b.strictly_worse_than_count == 0:
                    number_b.rank = i + 1
                    next_front.append(number_b)
        rank_sorted.append(next_front)
        i += 1
    # The last front will always be empty
    return rank_sorted[:-1]


def assign_sort_rank_2(numbers):
    """
    As per NSGA-II, Jensen 2003
    Assign ranks to the members of the pool and sort it by rank (e.g. member of the n'th non-dominated front)
    (ascending; lower is better)
    :param numbers:
    :return:
    """
    pool = sorted(numbers, key=lambda n: n.value)

    fronts = [[pool[0]]]
    pool[0].rank = 0
    e = 0
    for ind_a in pool[1:]:
        ind_a.inverse_domination_count = 0
        for ind_b in fronts[e]:
            # b dominates a
            if ind_b.strictly_better_than(ind_a):
                ind_a.inverse_domination_count += 1
        if ind_a.inverse_domination_count == 0:
            # ind_a belongs in one of the fronts [0, e] (inclusive)
            # find lowest rank b where ind_a is not dominated.
            # since each front is sorted in increasing order of fit[0] and decreasing order of fit[1}
            # it is only necessary to find the insertion point for the second fitness
            b = -1

            for r in range(len(fronts)):
                if ind_a.fitnesses[1] <= fronts[r][-1].fitnesses[1]:
                    b = r
                    break

            """
            #  naive approach
            for r in range(len(fronts)):
                x_i, x_l = 0, len(fronts[r])
                for i, ind_b in enumerate(fronts[r]):
                    x_i = i
                    if ind_b.dominates(ind_a):
                        break
                else:
                    b = r
                    assert ind_a.fitnesses[0] >= fronts[r][x_i].fitnesses[0]
                    assert ind_a.fitnesses[1] <= fronts[r][x_i].fitnesses[1]
                    break
            """
            assert b != -1
            fronts[b].append(ind_a)
            ind_a.rank = b
        else:
            e += 1
            fronts.append([])
            fronts[e].append(ind_a)
            ind_a.rank = e
    return fronts


def random_selection(li):
    # Need flat list ??
    numbers = []
    for _ in range(len(li)):
        numbers.append(np.random.choice(li))
    return numbers


def rank_distance_selection(nu1, nu2):
    # Need flat lists
    assert isinstance(nu1[0], Number)
    if nu2:
        assert isinstance(nu2[0], Number)

    n = len(nu1)  # == len(li2)
    rank_sorted = assign_sort_rank(list(chain(nu1, nu2)))
    new_numbers = []
    rank_indexes = []
    for front in rank_sorted:
        # This sort is a placeholder for a custom sort that can be done by 'sorted',
        # but it still sorts by the same value (distance)
        front = sorted(front, key=lambda num: num.distance)
        # Add fronts to empty number pool until no more complete fronts can be added.
        if len(new_numbers) + len(front) <= n:
            new_numbers.extend(front)
            rank_indexes.append(len(new_numbers))
        # Then, add individuals from the last front based on their distance.
        else:
            new_numbers.extend(front[:n - len(new_numbers)])
            rank_indexes.append(len(new_numbers))
            break
    # Return flat, rank sorted list and a list with indexes of the end position for each rank
    # e.g. first front is new_numbers[:rank_indexes[0]]
    return new_numbers, rank_indexes


def generate_from_numbers(numbers):
    # Takes a flat list, but can be made to work a nested list
    new_numbers = []
    for i in range(0, len(numbers) - 1, 2):
        number_a = numbers[i]
        number_b = numbers[i+1]
        mangled_numbers = mangle(number_a, number_b)
        new_numbers.extend(mangled_numbers)
    return new_numbers


def mangle(number_a, number_b):
    number_1 = Number(number_a.value - number_b.value)
    number_2 = Number(number_a.value + number_b.value)
    return number_1, number_2


def view_numbers(numbers, rank_indexes):
    # Need flat rank sorted list
    # assert that numbers is really rank sorted
    for num_i in range(len(numbers)-1):
        assert numbers[num_i].rank <= numbers[num_i+1].rank

    print("\n\nView", len(numbers), "numbers by rank:", [number.rank for number in numbers], "rank_indexes:", rank_indexes)
    count = 0
    for rank in range(len(rank_indexes)):
        front = get_front(numbers, rank_indexes, rank)
        for number in front:
            print(number.rank)
            count += 1
        print("\n")
    print("counted", count)


def get_front(numbers, rank_indexes, front_n):
    if front_n >= 1:
        start = rank_indexes[front_n-1]
    else:
        start = 0
    return numbers[start:rank_indexes[front_n]]


def loop(m=100, n=50):
    a = [Number() for _ in range(m)]
    b = []
    c = []
    for i in range(n):
        #print("Iteration:", i)
        b, rank_indexes = rank_distance_selection(a, b)
        #view_numbers(b, rank_indexes)
        c = random_selection(b)
        a = generate_from_numbers(c)
        #assert len(a) == len(b) == len(c) == m, \
        #    (len(a), len(b), len(c))

logging.basicConfig(level=logging.DEBUG)
print(timeit(loop, number=1))
