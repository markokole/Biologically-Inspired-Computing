import csv
import time
import itertools as it
import numpy as np
import math
import random

class tsp:
    def __init__(self):
        pass

    # load data into a numpy array
    def load_data(self, file):
        with open(file, "r") as f:
            data_from_file = np.array(list(csv.reader(f, delimiter=';')))  # convert list to numpy array

        self.data = data_from_file
        self.cities = self.data[:1]  # array of all cities

        print("Data loaded!")
        return self.data #, self.cities

    # create the distance matrix according to the number of cities
    def initialize_matrix(self, whole_distance_matrix, n_cities):
        row = n_cities + 1
        column = n_cities

        self.distance_matrix = np.array(whole_distance_matrix[:row, :column])
        self.cities = self.distance_matrix[0]  # names of the cities

        return self.distance_matrix, self.cities

    # function return the name of the city behind the index
    def get_city_name(self, cities, index):
        return cities[index]

    # calculate sum, mean and min of an array
    # used in genetic algorithm part of the code
    def calculate_stats(self, lov):
        sum_ = round(sum(lov), 1)
        avg_ = round(np.mean(lov), 2)
        min_ = min(lov)
        max_ = max(lov)
        dev_ = round(np.std(np.array(lov), ddof=1), 4)
        return sum_, avg_, min_, max_, dev_

    # function returns distance between two cities
    def get_distance_between_two_cities(self, distance_matrix, start, stop):
        return float(distance_matrix[start + 1, stop])

    # function calculates total distance of the route which is given as an input parameter.
    # The return to the fist city from the last one is also considered in the function
    def calculate_total_distance(self, distance_matrix, cities, order_of_visits):
        route = ""  # string for describing the route with city names
        route_index = []  # list with order of city indexes to describer route
        total_distance = 0
        order = order_of_visits + [order_of_visits[0]]
        for i in range(len(order) - 1):
            start_city_index = order[i]  # index of start city
            stop_city_index = order[i + 1]  # index of stop city
            start_city = self.get_city_name(cities=cities, index=start_city_index)  # name of start city
            #stop_city = self.get_city_name(cities=cities, index=stop_city_index)  # name of stop city
            distance = self.get_distance_between_two_cities(distance_matrix, start_city_index, stop_city_index)  # get distance between cities
            total_distance += distance  # add current distance to the total
            route = route + start_city + " "  # add current city to the route

            route_index.append(i)

        return round(total_distance, 2), route.rstrip()

    ###########################
    #### EXHAUSTIVE SEARCH ####
    ###########################
    # takes number of cities as a parameter and checks EVERY single permutation.
    # The permutation with shortest travelled distance is the preferred route
    def exhaustive_search(self, distance_matrix, cities, n_cities):
        number_of_all_routes = math.factorial(n_cities)
        start = time.time()
        counter = 0
        min_distance = 9999999
        min_route = []
        city_index = [i for i in range(n_cities)] #range(number_of_cities)  # create list of city indexes

        # loop through all permutations
        # permutations function takes in two parameters: list of candidates and number of elements in each permutation.
        # For example: permutations([1,2,3,4], 2) -> (1,2), (1,3)...
        for l in list(it.permutations(city_index, n_cities)):
            counter += 1
            distance, route = self.calculate_total_distance(distance_matrix=distance_matrix, cities=cities, order_of_visits=list(l))
            if distance < min_distance:
                min_distance = distance
                min_route = route

        end = time.time()
        run_time = round((end - start), 8)

        print("EXHAUSTIVE SEARCH. Number of cities:", n_cities)
        print("Number of all routes:", number_of_all_routes)
        print("Shortest route: %s\nRoute: %s\nTime:%s seconds" % (min_distance, min_route, run_time))

    #######################
    #### HILL CLIMBING ####
    #######################
    # function runs a hill climbing algorithm
    def hill_climbing(self, distance_matrix, cities, number_of_cities, number_of_swaps):
        list_of_city_index = [i for i in range(number_of_cities)]  # range(number_of_cities)  # create list of city indexes

        ##An initial solution is generated using random shuffle
        np.random.shuffle(list_of_city_index)  # random shuffle to get initial solution
        inital_solution = list_of_city_index  # prepare initial solution
        min_distance, optimal_route = self.calculate_total_distance(distance_matrix=distance_matrix, cities=cities, order_of_visits=inital_solution)

        # print "Start swapping:"
        for i in range(1, number_of_swaps + 1):
            # print i
            first_city = np.random.randint(0, number_of_cities)
            second_city = np.random.randint(0, number_of_cities)
            # print first_city, "<->", second_city
            # print "Swap cities: %s <-> %s" % (list_of_city_index[first_city], list_of_city_index[second_city])
            list_of_city_index[first_city], list_of_city_index[second_city] = list_of_city_index[second_city], \
                                                                              list_of_city_index[first_city]
            # print list_of_city_index
            temp_distance, temp_route = self.calculate_total_distance(distance_matrix=distance_matrix, cities=cities, order_of_visits=list_of_city_index)
            # print temp_distance, temp_route
            if (temp_distance < min_distance):
                min_distance = temp_distance
                optimal_route = temp_route

        return min_distance, optimal_route

    # function runs hill_climbing function 20 times and puts out statistics
    def run_hill_climbing(self, distance_matrix, cities, n_cities, n_swaps):
        import time
        list_of_results_distance = []  # list for distances
        list_of_results_route = []  # list for lists of routes

        start2 = time.time()

        # first 10 cities with 1000 swaps
        for i in range(1, 21):  # run 20 times
            distance, route = self.hill_climbing(distance_matrix=distance_matrix, cities=cities, number_of_cities=n_cities, number_of_swaps=n_swaps)  # get distance and route
            list_of_results_distance.append(distance)  # append to the list
            list_of_results_route.append(route)  # append to the list

        end2 = time.time()
        run_time = round((end2 - start2), 4)

        min_route = min(list_of_results_distance)
        max_route = max(list_of_results_distance)
        list_distance_np = np.array(list_of_results_distance)  # convert to numpy array
        mean_route = np.mean(list_distance_np)
        std_dev = round(np.std(list_distance_np, ddof=1), 4)  # calculate standard deviation using numpy

        index_of_min_route = list_of_results_distance.index(min_route)  # get index of the shortest distance
        min_route_cities = list_of_results_route[index_of_min_route]  # city to city route that has shortest distance

        print("HILL CLIMBING. Number of cities: %s, number of swaps: %s" % (n_cities, n_swaps))
        print("Shortest route: %s\nLongest route: %s\nMean distance: %s\nStandard deviation: %s" % (min_route, max_route, mean_route, std_dev))
        print("Route: %s" % (min_route_cities))
        print("Time needed: %s" % (run_time))

    #######################
    #### COMPARISONS ####
    #######################
    # function runs a comparison check for 10 cities between exhaustive search and hill climbing
    def compare_es_and_hc_with_10_cities(self, distance_matrix, cities, n_swaps):
        print("Calculating best routes for 10 cities...\n")
        number_of_cities = 10
        self.exhaustive_search(distance_matrix=distance_matrix, cities=cities, n_cities=number_of_cities)
        self.run_hill_climbing(distance_matrix=distance_matrix, cities=cities, n_cities=number_of_cities, n_swaps=n_swaps)

    #########################
    ### GENETIC ALGORITHM ###
    #########################
    # calculate probability of elements in an array
    def probability(self, fitness_values, sum_fitness):
        return map(lambda x: round(x / float(sum_fitness), 5), fitness_values)

    # generate genotypes using numerical distribution from probability_list
    def generate_mating_pool(self, population, probability):
        mating_pool = []
        # do this to ensure probability sum to be equal 1.0
        #print(probability)
        probs = np.array(probability)
        #print(probs)
        probs /= probs.sum()
        # create indexed list for random choice
        indexes = range(0, len(population))
        for i in range(len(population)):
            rand_index = np.random.choice(np.array(indexes), p=probs)  # get random index
            route = population[rand_index]  # get route for the index value
            mating_pool.append(route)  # append route to mating pool

        return mating_pool
    #end probability

    # Order crossover algorithm
    def order_xover(self, parent_1, parent_2, xover_point_1, subset_length):

        xover_point_2 = xover_point_1 + subset_length  # Second crossover point
        offspring = [''] * len(parent_1)  # initialize offspring

        parent_1_subset = parent_1[xover_point_1: xover_point_2]  # Parent 1 subset

        offspring[
        xover_point_1: xover_point_2] = parent_1_subset  # implant subset from Parent 1 into offspring genome
        # rearrange order of genes in parent 2 before adding missing genes to offspring
        reshuffled_parent2 = parent_2[xover_point_2:] + parent_2[:xover_point_2]

        # use list comprehension to get the values in Parent 2 that are not in offspring already
        not_in_offspring = [item for item in reshuffled_parent2 if item not in offspring]

        # populate the tail first (part from subset to the end of the list)
        for i in range(xover_point_2, len(offspring)):
            offspring[i] = not_in_offspring[0]
            not_in_offspring.pop(0)

        # populate the head (part from start to the subset)
        for i in range(0, xover_point_1):
            if len(not_in_offspring) > 0:
                offspring[i] = not_in_offspring[0]
            not_in_offspring.pop(0)

        return offspring
    #end order_xover

    # there is a 1% chance for a mutation to happen for a given list
    # scramble mutation takes a random subset of elements from a list and randomly reorders the sublist.
    # The sublist is placed back into the list
    def scramble_mutation(self, list_of_city_indexes):
        scrambled_list = list_of_city_indexes
        is_mutating = random.randint(1, 100)
        if is_mutating == 1:  # 1% chance for mutation to happen
            length = len(list_of_city_indexes)
            start_subset = np.random.randint(0, length)  # define the start of the subset that will be scrambled
            end_subset = np.random.randint(start_subset, length)  # define the end of the subset that will be scrambled
            subset = list_of_city_indexes[start_subset:end_subset]  # get the subset for scramble
            random_shuffle = random.sample(subset, len(subset))  # do a random shuffle of the subset
            scrambled_list = list_of_city_indexes[:start_subset] + random_shuffle + list_of_city_indexes[end_subset:]

        return scrambled_list

    # 90% - get 2 fittest from the family (shortest route)
    # 10% - get both children
    def get_2_fittest(self, distance_matrix, cities, p1, p2, o1, o2):
        elite_selection = random.randint(1, 100)
        if elite_selection > 10:  # 90% chance for elitism
            family_fitness = []
            family_route = []
            for e in (p1, p2, o1, o2):
                distance, route = self.calculate_total_distance(distance_matrix=distance_matrix, cities=cities, order_of_visits=e)
                family_fitness.append(distance)
                family_route.append(e) #used numbers instead of cities
            family_fitness_sorted = sorted(family_fitness)  # sort fitnes min -> max
            shortest_distance_index = family_fitness.index(family_fitness_sorted[0])  # index to shortest distance
            second_shortest_distance_index = family_fitness.index(family_fitness_sorted[1])  # index to second shortest distance

            shortest_route = family_route[shortest_distance_index]  # shortest route
            second_shortest_route = family_route[second_shortest_distance_index]  # second shortest route
            return shortest_route, second_shortest_route #self.phenotype_to_genotype(cities=cities, list_of_cities=shortest_route), self.phenotype_to_genotype(cities==cities, list_of_cities=second_shortest_route)
        else:  # 10% chance for using children further in population
            return o1, o2

    # number_of_cities defines number of cities a route has
    # population_size defines amount of individuals
    def genetic_algorithm(self, distance_matrix, cities, population_size, number_of_generations, n_cities):
        initial_population = []
        population_fitness = []
        fitness_of_best_fit_ind = []

        start3 = time.time()

        city_index = range(n_cities)  # generate city index
        all_permutations = list(it.permutations(city_index, n_cities))
        if population_size > len(all_permutations):
            print("Population size larger than number of permutations! Population size corrected to %s factorial" % (n_cities))
            population_size = math.factorial(n_cities)
        print("GENETIC ALGORITHM. Number of cities in route: %s" % (n_cities))
        print("Population size: %s" % (population_size))

        for i in range(population_size):
            current_route = list(random.choice(all_permutations))
            initial_population.append(current_route)  # populate initial population with random permutations
            distance, route = self.calculate_total_distance(distance_matrix=distance_matrix, cities=cities, order_of_visits=current_route)  # get distance from route
            population_fitness.append(distance)  # append every routes fitness to the list

        sum_fitness, avg_fitness, min_fitness, max_fitness, dev_fitness = self.calculate_stats(population_fitness)
        fitness_of_best_fit_ind.append(min_fitness)  # put the min fitness from the initial choice

        # probability based on individual fitness and while fitness
        probability_list = list(self.probability(population_fitness, sum_fitness))

        print("Initial population's sum: %s, average: %s and minimum: %s" % (sum_fitness, avg_fitness, min_fitness))

        #mating_pool = []  # mating pool has 4 individual routes selected from initial population using probability factor

        mating_pool = self.generate_mating_pool(initial_population, probability_list)

        new_generation = mating_pool
        best_fitness = min_fitness  # best fitness is the fitness of the shortest route in random population

        for i in range(1, number_of_generations + 1):
            offspring_after_scrambled_mutation = []

            ###scrambled mutation
            for i in range(len(new_generation)):  # loop through whole generation
                scrambled_offspring = self.scramble_mutation(new_generation[i])
                offspring_after_scrambled_mutation.append(scrambled_offspring)  # append new offspring

            offspring_after_xover = []  # list with offsprings after order crossover

            ###order crossover
            i = 0  # counter
            while i < population_size:
                one_point_xover = np.random.randint(1, n_cities / 2)  # get random point for xover. Not the first one nor last one
                subset_length = np.random.randint(1, n_cities - one_point_xover - 1)  # get subset length

                offspring_1 = self.order_xover(offspring_after_scrambled_mutation[i],
                                               offspring_after_scrambled_mutation[i + 1], one_point_xover,
                                               subset_length)
                offspring_2 = self.order_xover(offspring_after_scrambled_mutation[i + 1],
                                               offspring_after_scrambled_mutation[i], one_point_xover,
                                               subset_length)

                parent_1 = new_generation[i]
                parent_2 = new_generation[i + 1]
                choice_1, choice_2 = self.get_2_fittest(distance_matrix=distance_matrix, cities=cities, p1=parent_1, p2=parent_2, o1=offspring_1, o2=offspring_2)

                # append both new offsprings to the list
                offspring_after_xover.append(choice_1)
                offspring_after_xover.append(choice_2)

                i += 2  # works with pairs so counter increases by 2
            #end while i < population_size

            # initialize
            offspring_fitness = []
            offspring_phenotype = []
            offspring_genotype = []
            for i in range(len(offspring_after_xover)):
                offspring_distance, offspring_route = self.calculate_total_distance(distance_matrix=distance_matrix, cities=cities, order_of_visits=offspring_after_xover[i])
                offspring_fitness.append(offspring_distance)
                offspring_phenotype.append(offspring_route)
                offspring_genotype.append(offspring_after_xover[i])

            fitness_of_best_fit_ind.append(min(offspring_fitness))

            new_generation = offspring_genotype  # generated generation becomes new generation
            random.shuffle(new_generation)  # shuffle the items in the new generation
        #end for i in range(1, number_of_generations + 1)

        end3 = time.time()
        run_time = round((end3 - start3), 4)

        sum_, avg_offspring_fitness, min_offspring_fitness, max_offspring_fitness, dev_offspring_fitness = self.calculate_stats(offspring_fitness)

        print("Fitness of the best individual in last generation (shortest distance to travel): %s" % (min_offspring_fitness))
        print("Longest distance to travel: %s" % (max_offspring_fitness))
        print("Average distance: %s" % (avg_offspring_fitness))
        print("Standard deviation: %s" % (dev_offspring_fitness))
        min_offspring_fitness_index = offspring_fitness.index(min_offspring_fitness)
        max_offspring_fitness_index = offspring_fitness.index(max_offspring_fitness)
        print("Shortest route (phenotype): %s" % (offspring_phenotype[min_offspring_fitness_index]))
        print("Longest route (phenotype): %s" % (offspring_phenotype[max_offspring_fitness_index]))
        # print "Shortest route (genotype): %s" % offspring_genotype[min_offspring_fitness_index]
        # print "Longest route (genotype): %s" % offspring_genotype[max_offspring_fitness_index]
        # print "fitness_of_best_fit_ind:", fitness_of_best_fit_ind
        print("Time needed: %s" % (run_time))

############
### RUN ####
############

filename = "data/european_cities.csv"
number_of_cities = 10
number_of_swaps = 10000 #for hill climbing
t = tsp()
whole_distance_matrix = t.load_data(filename)
distance_matrix, cities = t.initialize_matrix(whole_distance_matrix=whole_distance_matrix,n_cities=number_of_cities)
#print(t.get_city_name(cities=cities, index=2))
#print(t.get_distance_between_two_cities(distance_matrix=distance_matrix, start=0, stop=2))

#print(t.calculate_total_distance(distance_matrix=distance_matrix, cities=cities, order_of_visits=[0,2,1]))

#t.exhaustive_search(distance_matrix=distance_matrix, cities=cities, n_cities=number_of_cities)
#print(t.hill_climbing(distance_matrix=distance_matrix, cities=cities, number_of_cities=number_of_cities, number_of_swaps=number_of_swaps))
#t.run_hill_climbing(distance_matrix=distance_matrix, cities=cities, n_cities=number_of_cities, n_swaps=number_of_swaps)
#t.compare_es_and_hc_with_10_cities(distance_matrix=distance_matrix, cities=cities, n_swaps=number_of_swaps)
population_size= 100
number_of_generations=100
t.genetic_algorithm(distance_matrix=distance_matrix, cities=cities, population_size=population_size, number_of_generations=number_of_generations, n_cities=number_of_cities) #population_size, number_of_generations
