#An assignment for course Biologically inspired computing
#Link to the assignment page: http://www.uio.no/studier/emner/matnat/ifi/INF3490/h17/assignment-1/

import csv
import time
import itertools as it
import numpy as np
import math
import random

class tsp(object):

    number_of_cities = 6
    swap_quantity=1000
    data = []
    cities = []
    subset = []

    def number_of_visits(self, number_of_cities=6):
        self.number_of_cities = number_of_cities

    def number_of_swaps(self, swap_quantity=1000):
        self.swap_quantity = swap_quantity

    #load data from file into a numpy array
    def load_data(self, file):
        with open(file, "r") as f:
            data_from_file = np.array(list(csv.reader(f, delimiter=';'))) #convert list to numpy array

        self.data = data_from_file

        self.cities = self.data[:1] # array of all cities
        print "Data loaded!"
        return self.data

    def get_list_of_cities(self):
        return self.cities

    def get_city_name(self, i):
        #type numpy.ndarray
        return self.cities.item(i)

    #choose a subset size by choosing number of cities
    def data_subset(self):
        row = self.number_of_cities + 1
        column = self.number_of_cities
        self.subset = self.data[:row,:column]
        self.cities = self.subset[:1]
        return self.subset

    #calculate sum, mean and min of an array
    def calculate_stats(self, lov):
        sum_ = round(sum(lov), 1)
        avg_ = round(np.mean(lov), 2)
        min_ = min(lov)
        max_ = max(lov)
        dev_ = round(np.std(np.array(lov), ddof=1), 4)
        return sum_, avg_, min_, max_, dev_

    #function returns distance between two cities
    def get_distance_between_two_cities(self, start, stop):
        return float(self.subset[start+1,stop])

    #function calculates total distance of the route which is given as an input parameter.
    #The return to the fist city from the last one is also considered in the function
    def calculate_total_distance(self, order_of_visits):
        route = [] #route = "" #string for descibing the route with city names
        route_index = [] #list with order of city indexes to describer route
        total_distance = 0
        order = order_of_visits + [order_of_visits[0]]

        for i in xrange(len(order)-1):
            start_city_index = order[i] #index of start city
            stop_city_index = order[i+1] #index of stop city
            start_city = self.get_city_name(start_city_index) #name of start city
            stop_city = self.get_city_name(stop_city_index) #name of stop city
            #print "start city:", start_city
            #print "stop_city", stop_city
            distance = self.get_distance_between_two_cities(start_city_index, stop_city_index) #get distance between cities
            #print "distance", distance
            total_distance += distance #add current distance to the total
            route.append(start_city)#route = route + start_city + " " #add current city to the route

            route_index.append(i)

        return round(total_distance, 2), route


###########################
#### EXHAUSTIVE SEARCH ####
###########################
    #takes number of cities as a parameter and checks EVERY single permutation.
    #The permutation with shortest travelled distance is the preferred route
    def exhaustive_search(self):
        number_of_all_routes = math.factorial(self.number_of_cities)

        start = time.time()
        counter = 0
        min_distance = 9999999
        min_route = []
        city_index = range(self.number_of_cities)

        #loop through all permutations
        #permutations function takes in two parameters: list of candidates and number of elements in each permutation.
        #For example: permutations([1,2,3,4], 2) -> (1,2), (1,3)...
        for l in list(it.permutations(city_index, self.number_of_cities)):
            counter += 1
            distance, route = self.calculate_total_distance(list(l))
            if distance < min_distance:
                min_distance = distance
                min_route = route

        end = time.time()
        run_time = round((end - start), 8)

        print "EXHAUSTIVE SEARCH. Number of cities:", self.number_of_cities
        print "Shortest route: %s\nRoute: %s\nTime:%s seconds\n" % (min_distance, min_route, run_time)

        return min_distance, min_route, run_time

#######################
#### HILL CLIMBING ####
#######################
    #function runs a hill climbing algorithm
    def hill_climbing(self):

        list_of_city_index = range(self.number_of_cities) #create list of city indexes
        #print list_of_city_index

        np.random.shuffle(list_of_city_index) #random shuffle to get initial solution
        inital_solution = list_of_city_index #prepare initial solution
        #print inital_solution
        min_distance, optimal_route = self.calculate_total_distance(inital_solution)
        #print min_distance, optimal_route

        max_swaps = self.swap_quantity + 1
        for i in range(1, max_swaps):
            #print i
            first_city = np.random.randint(0, self.number_of_cities) #random city index
            second_city = np.random.randint(0, self.number_of_cities) #random city index
            #swap city indexes with with each other
            list_of_city_index[first_city], list_of_city_index[second_city] = list_of_city_index[second_city], list_of_city_index[first_city]
            #print list_of_city_index
            temp_distance, temp_route = self.calculate_total_distance(list_of_city_index)
            #print temp_distance, temp_route
            if (temp_distance < min_distance): #archive minimal travelled distance and route
                min_distance = temp_distance
                optimal_route = temp_route

        return min_distance, optimal_route


    #function runs hill_climbing function 20 times and puts out statistics
    def run_hill_climbing(self):

        list_of_results_distance = [] #list for distances
        list_of_results_route = [] #list for lists of routes

        start2 = time.time()

        #first 10 cities with 1000 swaps
        for i in xrange(1,21): #run 20 times
            distance, route = self.hill_climbing() #get distance and route
            list_of_results_distance.append(distance) #append to the list
            #print distance, route
            list_of_results_route.append(route) #append to the list

        end2 = time.time()
        run_time = round((end2 - start2), 4)

        #collect statistics
        sum_route, mean_route, min_route, max_route, std_dev = self.calculate_stats(list_of_results_distance)

        index_of_min_route = list_of_results_distance.index(min_route) #get index of the shortest distance
        min_route_cities = list_of_results_route[index_of_min_route] #city to city route that has shortest distance

        print "HILL CLIMBING. Number of cities: %s, number of swaps: %s" % (self.number_of_cities, self.swap_quantity)
        print "Shortest route: %s\nLongest route: %s\nMean distance: %s\nStandard deviation: %s" % (min_route, max_route, mean_route, std_dev)
        print "Route: %s" % (min_route_cities)
        print "Time needed: %s\n" % (run_time)


#########################
### GENETIC ALGORITHM ###
#########################
    #calculate probability of elements in an array
    def probability(self, lov, sum_val):
        return map(lambda x: round(x/float(sum_val), 5), lov)

    #generate genotypes using numerical distribution from probability_list
    def generate_mating_pool(self, population, probability):
        mating_pool = []
        #do this to ensure probability sum to be equal 1.0
        probs = np.array(probability)
        probs /= probs.sum()
        #create indexed list for random choice
        indexes = range(0, len(population))
        for i in xrange(len(population)):
            rand_index = np.random.choice(np.array(indexes), p=probs) #get random index
            route = population[rand_index] #get route for the index value
            mating_pool.append(route) #append route to mating pool

        return mating_pool


    #Order crossover algorithm
    def order_xover(self, parent_1, parent_2, xover_point_1, subset_length):

        xover_point_2 = xover_point_1 + subset_length #Second crossover point
        offspring = [''] * len(parent_1) #initialize offspring

        parent_1_subset = parent_1[xover_point_1 : xover_point_2] #Parent 1 subset

        offspring[xover_point_1 : xover_point_2] = parent_1_subset #implant subset from Parent 1 into offspring genome
        #rearrange order of genes in parent 2 before adding missing genes to offspring
        reshuffled_parent2 = parent_2[xover_point_2:] + parent_2[:xover_point_2]

        #use list comprehension to get the values in Parent 2 that are not in offspring already
        not_in_offspring = [item for item in reshuffled_parent2 if item not in offspring]

        #populate the tail first (part from subset to the end of the list)
        for i in xrange(xover_point_2, len(offspring)):
            offspring[i] = not_in_offspring[0]
            not_in_offspring.pop(0)

        #populate the head (part from start to the subset)
        for i in xrange(0, xover_point_1):

            if len(not_in_offspring) > 0:
                offspring[i] = not_in_offspring[0]
            not_in_offspring.pop(0)

        return offspring

    #there is a 1% chance for a mutation to happen for a given list
    #scramble mutation takes a random subset of elements from a list and randomly reorders the sublist.
    #The sublist is placed back into the list
    def scramble_mutation(self, list_of_city_indexes):
        scrambled_list = list_of_city_indexes
        is_mutating = random.randint(1,100)
        if is_mutating == 1: # 1% chance for mutation to happen
            length = len(list_of_city_indexes)
            start_subset = np.random.randint(0, length) #define the start of the subset that will be scrambled
            end_subset = np.random.randint(start_subset, length) #define the end of the subset that will be scrambled
            subset = list_of_city_indexes[start_subset:end_subset] #get the subset for scramble
            random_shuffle = random.sample(subset, len(subset)) #do a random shuffle of the subset
            scrambled_list = list_of_city_indexes[:start_subset] + random_shuffle + list_of_city_indexes[end_subset:]

        return scrambled_list

    #encoding
    def phenotype_to_genotype(self, list_of_cities):
        list_of_city_index = []
        #print self.get_list_of_cities().tolist()
        for city in list_of_cities:
            #print city
            idx = self.cities.tolist()[0].index(city)
            list_of_city_index.append(idx)
        return list_of_city_index


    #90% - get 2 fittest from the family (shortest route)
    #10% - get both children
    def get_2_fittest(self, p1, p2, o1, o2):

        elite_selection = random.randint(1,100)
        if elite_selection > 10: # 90% chance for elitism
            family_fitness = []
            family_route = []
            for e in (p1, p2, o1, o2):
                d,r = self.calculate_total_distance(e)
                family_fitness.append(d)
                family_route.append(r)
            family_fitness_sorted = sorted(family_fitness) #sort fitnes min -> max
            shortest_distance_index = family_fitness.index(family_fitness_sorted[0]) #index to shortest distance
            second_shortest_distance_index = family_fitness.index(family_fitness_sorted[1]) #index to second shortest distance

            shortest_route = family_route[shortest_distance_index] #shortest route
            second_shortest_route = family_route[second_shortest_distance_index] #second shortest route

            return self.phenotype_to_genotype(shortest_route), self.phenotype_to_genotype(second_shortest_route)
        else: #10% chance for using children further in population
            return o1, o2


    #number_of_cities defines number of cities a route has
    #population_size defines amount of individuals
    def genetic_algorithm(self, population_size, number_of_generations):
        initial_population = []
        population_fitness = []
        fitness_of_best_fit_ind = []

        start3 = time.time()

        city_index = range(self.number_of_cities) #generate city index
        all_permutations = list(it.permutations(city_index, self.number_of_cities))
        if population_size > len(all_permutations):
            print "Population size larger than number of permutations! Population size corrected to %s factorial" % (self.number_of_cities)
            population_size = math.factorial(self.number_of_cities)
        print "GENETIC ALGORITHM. Number of cities in route: %s" % (self.number_of_cities)
        print "Population size: %s" % (population_size)

        for i in xrange(population_size):
            current_route = list(random.choice(all_permutations))
            initial_population.append(current_route) #populate initial population with random permutations
            distance, route = self.calculate_total_distance(current_route) #get distance from route
            population_fitness.append(distance) #append every routes fitness to the list

        #print "Initial population in genotypes:\n %s:" % (initial_population)
        #print "Initial population's fitness: %s:" % population_fitness

        sum_fitness, avg_fitness, min_fitness, max_fitness, dev_fitness = self.calculate_stats(population_fitness)
        fitness_of_best_fit_ind.append(min_fitness) #put the min fitness from the initial choice

        #probability
        probability_list = self.probability(population_fitness, sum_fitness)

        print "Initial population's sum: %s, average: %s and minimum: %s" % (sum_fitness, avg_fitness, min_fitness)


        mating_pool = [] #mating pool has 4 individual routes selected from initial population using probability factor
        mating_pool = self.generate_mating_pool(initial_population, probability_list)
        #print "Mating pool:\n%s" % (mating_pool)

        new_generation = mating_pool
        best_fitness = min_fitness # best fitness is the fitness of the shortest route in random population

        for i in xrange(1, number_of_generations+1):
            #print "GENERATION %s:" % i

            offspring_after_scrambled_mutation = []

            ###scrambled mutation
            for i in xrange(len(new_generation)): #loop through whole generation
                scrambled_offspring = self.scramble_mutation(new_generation[i])
                offspring_after_scrambled_mutation.append(scrambled_offspring) #append new offspring

            offspring_after_xover = [] #list with offsprings after order crossover

            ###order crossover
            i = 0 #counter
            while i < population_size:
                one_point_xover = np.random.randint(1, self.number_of_cities/2) #get random point for xover. Not the first one nor last one
                #print "One point crossover: %s" % one_point_xover
                subset_length = np.random.randint(1, self.number_of_cities-one_point_xover -1) #get subset length
                #print "Length of subset: %s" % subset_length

                offspring_1 = self.order_xover(offspring_after_scrambled_mutation[i], offspring_after_scrambled_mutation[i+1], one_point_xover, subset_length)
                offspring_2 = self.order_xover(offspring_after_scrambled_mutation[i+1], offspring_after_scrambled_mutation[i], one_point_xover, subset_length)

                parent_1 = new_generation[i]
                parent_2 = new_generation[i+1]
                #print "FAMILY:", parent_1, parent_2, offspring_1, offspring_2
                choice_1, choice_2 = self.get_2_fittest(parent_1, parent_2, offspring_1, offspring_2)

                #append both new offsprings to the list
                offspring_after_xover.append(choice_1)
                offspring_after_xover.append(choice_2)

                i += 2 #works with pairs so counter increases by 2
            #initialize
            offspring_fitness = []
            offspring_phenotype = []
            offspring_genotype = []
            for i in xrange(len(offspring_after_xover)):
                offspring_distance, offspring_route = self.calculate_total_distance(offspring_after_xover[i])
                offspring_fitness.append(offspring_distance)
                offspring_phenotype.append(offspring_route)
                offspring_genotype.append(self.phenotype_to_genotype(offspring_route))

            fitness_of_best_fit_ind.append(min(offspring_fitness))

            new_generation = offspring_genotype #generated generation becomes new generation
            random.shuffle(new_generation) #shuffle the items in the new generation

        end3 = time.time()
        run_time = round((end3 - start3), 4)


        sum_, avg_offspring_fitness, min_offspring_fitness, max_offspring_fitness, dev_offspring_fitness = self.calculate_stats(offspring_fitness)

        print "Fitness of the best individual in last generation (shortest distance to travel): %s" % min_offspring_fitness
        print "Longest distance to travel: %s" % max_offspring_fitness
        print "Average distance: %s" % avg_offspring_fitness
        print "Standard deviation: %s" % dev_offspring_fitness
        min_offspring_fitness_index = offspring_fitness.index(min_offspring_fitness)
        max_offspring_fitness_index = offspring_fitness.index(max_offspring_fitness)
        print "Shortest route (phenotype): %s" % offspring_phenotype[min_offspring_fitness_index]
        print "Longest route (phenotype): %s" % offspring_phenotype[max_offspring_fitness_index]
        #print "Shortest route (genotype): %s" % offspring_genotype[min_offspring_fitness_index]
        #print "Longest route (genotype): %s" % offspring_genotype[max_offspring_fitness_index]
        #print "fitness_of_best_fit_ind:", fitness_of_best_fit_ind
        print "Time needed: %s" % (run_time)

#######################
#### COMPARISONS ####
#######################

    #function runs a comparison check for x number of cities between exhaustive search and hill climbing
    def compare_es_and_hc_with_x_cities(self):
        print "Comparing Exhaustive search and Hill climbing:"
        print "Calculating best routes for %s cities...\n" % self.number_of_cities
        #number_of_cities=10
        self.exhaustive_search()

        #initialize_matrix(number_of_cities)
        self.run_hill_climbing()



input_no = input("Type number of cities to visit (6-10 is recommended): ")
print("Input for hill climbing")
input_swap = input("Type number of swaps for hill climbing algorithm (500-2000 is recommended): ")
print("Input for genetic algorithm")
input_pop_size = input("Type population size (between 50-5000 was tested): ")
input_number_gen = input("Type number of generations (tested on 20, should be a reasonable number): ")

d = tsp()
d.number_of_visits(input_no) #6 is default - no parameter value -> 6 cities
d.number_of_swaps(input_swap) #1000 is default - no parameter value -> 1000 swaps in hill climbing
##NB!! If both values are default, the hill climbing algorithm will always find the minimum, because: 6! = 720 < 1000
##And thats not fun

#data = d.load_data("/user/marko/emne/inf-3490/european_cities.csv") #path to the dataset
data = d.load_data("european_cities.csv") #path to the dataset
d.data_subset()
#print d.number_of_cities
#print d.get_list_of_cities()
#print d.get_distance_between_two_cities(0, 1)
#print d.get_city_name(3)
#print d.calculate_total_distance([1,2,3])
d.exhaustive_search()
#print d.hill_climbing()
d.run_hill_climbing()
#print d.calculate_stats([10, 20, 30, 40])
#d.compare_es_and_hc_with_x_cities()
#print d.generate_mating_pool([1,2,3,4], [0.1,0.2,0.5,0.2])
#print d.order_xover([1,2,3,4,5,6,7,8], [6,5,4,3,2,1,7,8], 2, 4)
#print d.scramble_mutation([1,2,3,4,5,6,7,8])
#print d.phenotype_to_genotype(['Barcelona', 'Berlin'])
d.genetic_algorithm(input_pop_size, input_number_gen)
#d.genetic_algorithm(50, 20) #population_size, number_of_generations
#d.genetic_algorithm(500, 20)
#d.genetic_algorithm(1000, 20)
#d.genetic_algorithm(5000, 20)
#print d.get_2_fittest([5,2,0], [3,2,4], [3,1,2], [2,0,1])
