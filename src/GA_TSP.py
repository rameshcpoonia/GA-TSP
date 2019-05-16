# Copyright 2018, Magnus Gribbestad, All rights reserved.
# For questions contact me at magnus@gribbestad.no

import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
import copy
import warnings

'''
Copyright 2018, Magnus Gribbestad, All rights reserved.
For questions or permission usage contact me at magnus@gribbestad.no

GA_TSP is a class that can solve TSP, designed and tested for different travelling salesman problems.

How to use the class:
    1. Instantiate an object of this class GA_TSP
    2. To run the GA use the function: runGA()

    The runGA() function has several input parameters, some of them are optional.
    The possible input parameters are:

    map: List of tuples that contains x and y position of all the cities. Format: [(x1, y1), (x2, y2), ..., (xn, yn)]
    genLimit: Integer: Generation limit, which is the limit of where the GA will stop.
    stallLimit: Integer: Number of consecutive iterations without improvement that will terminate the GA
    popSize: Integer: Population Size
    nStops (optional): Integer: nCities to travel to. If not defined all cities in the map will be considered
    pCross (optional): Float [0-1]: Probability of doing crossover. Default = 0.7
    pMut (optional): Float [0-1]: Probability of doing mutation. Default = 0.05
    eliteRatio (optional): Float [0-1]: Ratio of the best chromosomes that should be copied to next generation. Default = 0.2)
    optimalRoute (optional): Float: Distance of the best route. Default = None
    crossoverMethod (optional): String: Name of the crossover method to use. Included: pmx and ox. Default = pmx
    mutationMethod (optional): String: Name of the mutation method to use. Included: twors, rsm, cim, psm. Default = rsm
    plotting (optional): Boolean: Real-time plotting of best route each generation. Reduces effectiveness. Default = False
    map_name (optional): String with the name of the cities locations, for instance: Norway.  Default = None

'''
class GA_TSP(object):
    def __init__(self):
        self.best_each_gen = []       # Holds best (min) distance each generation
        self.best_route_each_gen = [] # Holds best route each generation
        self.avg_each_gen = []        # Holds the average distance each generation
        self.time_elapsed = 0         # Time elapsed for the GA
        self.best_route = None        # Will hold the best route after running
        self.best_distance = None     # Will hold the best (min) distance after running

    def runGA(self, map, genLimit, stallLimit, popSize, nStops = None, pCross = 0.7, pMut = 0.05, eliteRatio = 0.2, optimalRoute = None, crossover_method="pmx", mutation_method="rsm", plotting=True, map_name=None):
        '''
        This function runs the GA with the parameters specified as input.

        :param map: List of tuples that contains x and y position of all the cities. Format: [(x1, y1), (x2, y2), ..., (xn, yn)]
        :param genLimit: Integer: Generation limit, which is the limit of where the GA will stop.
        :param stallLimit: Integer: Number of consecutive iterations without improvement that will terminate the GA
        :param popSize: Integer: Population Size
        :param nStops: (optional): Integer: nCities to travel to. If not defined all cities in the map will be considered
        :param pCross: (optional): Float [0-1]: Probability of doing crossover. Default = 0.7
        :param pMut: (optional): Float [0-1]: Probability of doing mutation. Default = 0.05
        :param eliteRatio: (optional): Float [0-1]: Ratio of the best chromosomes that should be copied to next generation. Default = 0.2)
        :param optimalRoute: (optional): Float: Distance of the best route. Default = None
        :param crossover_method: (optional): String: Name of the crossover method to use. Included: pmx and ox. Default = pmx
        :param mutation_method: optional): String: Name of the mutation method to use. Included: twors, rsm, cim, psm. Default = rsm
        :param plotting: (optional): Boolean: Real-time plotting of best route each generation. Reduces effectiveness. Default = False
        :param map_name: (optional): String with the name of the cities locations, for instance: Norway.  Default = None

        :return: When the function is finished running the best route's distance will be printed. In addition, will
         some plots appear that shows the best route. The route (city indexes and distance) can also be reached from
         the function "get_best_route()".
        '''
        self.map = map # Holds the position of all the cities
        self.best_each_gen = [] # Reset list
        self.best_route_each_gen = [] # Reset list
        self.avg_each_gen = [] # Reset list
        self.time_elapsed = 0 # Reset timer

        self.prev_best = 10 ** 20 # Sets previous best to "infinite" large number
        self.stall_counter = 0 # Stall counter - keeps track of iterations ran without improvement

        # Updates the GA settings from the function inputs
        gen_limit = genLimit
        gen_for_converged = stallLimit
        pop_size = popSize
        if pop_size % 2 != 0: pop_size + 1
        p_cross = pCross
        p_mut = pMut
        elite_ratio = eliteRatio


        self.selection = self.select_selection() # Holds the function for the selection method (roulette wheel)
        self.crossover = self.select_crossover(crossover_method) # Holds the selected crossover function
        self.mutation = self.select_mutation(mutation_method) # Holds the selected mutation function

        # If number of stops is undefined all cities is evaluated. If not the first nStops cities is selected.
        if nStops is None:
            nStops = len(self.map)
        else:
            nStops = nStops

        # Indexing the cities from 0 to nStops.
        cities = list(range(0, len(self.map)))[0:nStops + 1]
        random.seed(5)

        # Generate the initial population
        prev_pop = self.generate_population(pop_size, cities)

        # Calculates the distance and fitness
        prev_distance = self.get_cost(prev_pop, map)
        prev_fitness = self.get_fitness(prev_distance)

        # Updates the list holdign the best distances and routes
        self.best_each_gen.append(min(prev_distance))
        best_init_route = prev_pop[prev_distance.index(min(prev_distance))]
        self.best_route_each_gen.append(best_init_route)
        self.avg_each_gen.append(sum(prev_distance) / float(len(prev_distance)))

        # If real-time plotting is selected it is initialised.
        # In addition a warning will be sent, that says that real-time plotting reduces efficiency
        if plotting == True:
            self.plotting_info = self.initialise_while_running_plot(best_init_route, map_name)
            warnings.warn(" \n Real-time plotting reduced algorithm performance! \n", UserWarning)


        criteria = False # Stopping criteria is initially set to false
        gen_num = 1 # Generation number is set to 0
        start_time = time.clock()  # Start measuring time

        # GA Loop starts here - runs until criteria is met.
        while not criteria:
            cur_pop = []  # List for holding

            # Perform elitism - copy best X% from previous population into the new population
            cur_pop = cur_pop + self.elitism(prev_pop, prev_distance, elite_ratio)
            #test = self.get_cost(cur_pop, self.map)
            #print("A: ", test[0:3], " ", cur_pop[0])

            # Run loop until the population size is the correct size
            while len(cur_pop) < pop_size:
                # Select parents for mating with the selection method
                parent1 = copy.deepcopy(prev_pop[self.selection(prev_fitness)])
                parent2 = copy.deepcopy(prev_pop[self.selection(prev_fitness)])

                # Create offspring from the crossover function with the 2 selected parents
                child1, child2 = self.crossover(parent1, parent2, p_cross)

                # Perform mutation on the two new offsprings
                child1 = self.mutation(child1, p_mut)
                child2 = self.mutation(child2, p_mut)

                # Add the two offsprings to the new population
                cur_pop.append(child1)
                cur_pop.append(child2)

            # Set the new population as the previous
            prev_pop = cur_pop

            # Calculate distance and fitness
            prev_distance = self.get_cost(prev_pop, self.map)
            prev_fitness = self.get_fitness(prev_distance)

            #test = copy.deepcopy(prev_distance)
            #test.sort()
            #print("C: ", test[0:3], " ", prev_pop[0])

            min_distance = min(prev_distance)  # Find the shortest tour
            #print(min_distance)
            self.best_each_gen.append(min_distance)  # Add shortest tour to the list
            current_best = prev_pop[prev_distance.index(min(prev_distance))]  # Find the best route in the cur pop
            self.best_route_each_gen.append(current_best) # Add best route into the list
            self.avg_each_gen.append(sum(prev_distance) / float(len(prev_distance))) # Add average distance to the list

            # If plotting is activated update the plot with the best route for this generation.
            if plotting:
                self.show_best_while_running(prev_pop[prev_distance.index(min(prev_distance))])

            # Check if the stopping criteria is met - based on max generations and if GA converged.
            criteria = self.check_criteria(gen_num, gen_limit, min_distance, gen_for_converged)

            gen_num += 1  # Increase generation number

        # ------------- Results ------------------
        # Measure elapsed time and print it
        self.time_elapsed = time.clock() - start_time

        # Plot graph that shows the shortest distance for each population
        #self.plot_best_generations()

        self.best_route = prev_pop[prev_distance.index(min(prev_distance))]
        self.best_distance = min(self.best_each_gen)

        # If real-time plotting is turned on close the plot when ga is finished.
        if plotting:
            plt.close(self.plotting_info[1])
            plt.ioff()


    def get_best_route(self):
        '''
        Returns best route, shortest distance and the time elapsed for the previous GA run.

        :return: best_route (array of city indexes), best_distance (float), time_elapsed in seconds(float)
        '''
        return self.best_route, self.best_distance, self.time_elapsed

    def generate_population(self, pop_size, cities):
        '''
        Generates a initial population of size "pop_size", each chromosome in the population
        is a permuation of the city indexes given in "cities".

        :param pop_size: Integer - Population size
        :param cities: List - Cities, list of indexes of all cities
        :return: initial population - list with pop_size chromosomes
        '''
        # Generate initial population
        org_pop = []
        for i in range(0, pop_size):
            # Add a new chromosome to the original population
            org_pop.append(list(np.random.permutation(cities)))

        return org_pop

    def get_cost(self, pop, map):
        '''
        Calculates the distance (cost) of each tour (permutation) in the population.
        The distance is calculated by using the euclidean distance bewteen each pair of cities.
        The distance includes the distance between the last and the first city.

        :param pop: Population that holds x chromosomes (tours)
        :param map: Map that holds the x- and y-position of each city
        :return: A list with the distance of each tour in the population
        '''
        pop_size = len(pop)
        distances = []
        for i in range(0, pop_size):
            new_distance = 0
            cities = pop[i]
            for k in range(0, len(pop[0]) - 1):
                city1 = cities[k]
                city2 = cities[k + 1]
                p1 = map[city1]
                p2 = map[city2]
                new_distance += math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

            # Add distance back to first city
            new_distance += math.sqrt(((map[cities[0]][0] - map[cities[k + 1]][0]) ** 2) +
                                      ((map[cities[0]][1] - map[cities[k + 1]][1]) ** 2))
            distances.append(new_distance)
        return distances


    def get_fitness(self, distances):
        '''
        Converts the distances to a fitness

        :param distances: A list with distances
        :return: The distances converted to fitness (bigger is better)
        '''
        fitnesses = [(1 / distance) * len(distances) for distance in distances]
        return fitnesses

    def elitism(self, pop, distances, ratio):
        '''
        Selects the X% best chromosomes from a population. These best chromosomes are returned.

        :param pop: Population of tours (chromosomes)
        :param distances: List of distances
        :param ratio:  Ratio [0-1] - The ratio of the best chromosomes to return
        :return: elite - The best x (ratio) chromosomes in the pop
        '''
        # Makes sure that the number of chromosomes to keep are an even number
        to_keep = round(len(pop) * ratio)
        if to_keep % 2 != 0: to_keep + 1
        # Sort indexes
        sorted_indexes = sorted(range(len(distances)), key=lambda k: distances[k])

        # Choose the elite
        elite = [pop[i] for i in sorted_indexes[0:to_keep]]

        return elite

    def select_selection(self):
        '''
        Select selection is used to select the selection method to use for selecting chromosomes for mating.
        Currently only the selection wheel function is supported

        :return: Function for selecting
        '''
        # Only one selection technique supported
        return self.fortune_wheel


    def fortune_wheel(self, weights):
        '''
        Fortune wheel function selects an index, based on probability. Better chromosomes (higher weights) has
        a bigger probability of being chosen.

        :param weights: List of fitnesses
        :return: Index of the selected chromosomes
        '''
        accumulation = np.cumsum(weights)
        p = random.random() * accumulation[-1];
        for index in range(0, len(accumulation)):
            if (accumulation[index] > p):
                return index

    def select_crossover(self, method):
        '''
        Select crossover methods based on the chosen method.
        PMX, OX, Single-point, 2-point and uniform crossover is implemented so far.

        If unknown method is chosen pxm will be chosen and a warning wil be outputted.

        Single-point, 2-point and uniform crossover are permuation crossover functions, therefore
        if these methods are chosen an alternative chromosome encoding method is used to make sure
        take tours contain no duplicates. This is often referred to as an inversion technique. To read about the
        technique go to this paper: http://user.ceng.metu.edu.tr/~ucoluk/research/publications/tspnew.pdf.

        :param method: String either "pmx" or "ox".
        :return: crossover method in the form of a variable
        '''
        if method == "pmx":
            crossover = self.crossover_pmx
        elif method == "ox":
            crossover = self.crossover_ox
        elif method == "1p":
            crossover = self.crossover_sp
        elif method == "2p":
            crossover = self.crossover_2p
        elif method == "uniform":
            crossover = self.crossover_uniform
        else:
            crossover = self.crossover_pmx
            warnings.warn(" \n The crossover method: '" + method + "' is unknown. \n Crossover method PMX is selected instead."
                                                               "Available crossover methods are 'pmx' and 'ox'. \n", UserWarning)
        return crossover

    def crossover_pmx(self, parent1, parent2, p_cross):
        '''
        Partially Matched Crossover (PMX),
        Implementation doing PMX for two parents.

        :param parent1: Chromosome to mate
        :param parent2: Chromosome to mate
        :param p_cross: Probability to mate
        :return: Returns two new childs
        '''
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Perform crossover only with the probability p_cross
        rnd = random.random()
        if rnd < p_cross:
            point = np.random.randint(0, len(parent1))
            swaps = len(child1[0:point])

            for i in range(0, swaps):
                child1_value = parent2[i]
                child2_value = parent1[i]

                child1[i], child1[parent1.index(child1_value)] = child1[parent1.index(child1_value)], child1[i]
                child2[i], child2[parent2.index(child2_value)] = child2[parent2.index(child2_value)], child2[i]

            return child1, child2
        else:
            return parent1, parent2

    def crossover_ox(self, parent1, parent2, p_cross):
        '''
        Implements the ordred crossover (OX).
        Crossover between two parents, to produce two offspring.
        Crossover happens with the probability p_cross.

        :param parent1: One of the parent chromosomes
        :param parent2: One of the parent chromosomes
        :param p_cross: Probability for doing crossover
        :return: Two offsprings
        '''
        rnd = random.random()
        if rnd < p_cross:
            child1 = [None] * len(parent1)
            child2 = [None] * len(parent1)

            # Select two crossover points
            points = np.random.randint(0, len(parent1))

            # Order 1 Crossover
            child1[0:points + 1] = parent1[0:points + 1]
            child1_rem = [item for item in parent2 if item not in child1]
            child2[0:points + 1] = parent2[0:points + 1]
            child2_rem = [item for item in parent1 if item not in child2]
            ind1 = 0
            for i in range(0, len(child1)):
                if child1[i] is None:
                    child1[i] = child1_rem[ind1]
                    child2[i] = child2_rem[ind1]
                    ind1 += 1
            return child1, child2
        else:
            return parent1, parent2

    def select_mutation(self, method):
        '''
        Select mutation method based on the chosen method.
        PSM, RSM, TWORS and CIM are implemented so far.
        If unknown method is chosen rsm will be chosen and a warning wil be outputed.

        :param method: String: 'psm', 'rsm', 'twors', 'cim'.
        :return: mutation method in the form of a variable
        '''
        if method == "psm":
            mutation = self.mutation_psm
        elif method == "rsm":
            mutation = self.mutation_rsm
        elif method == "twors":
            mutation = self.mutation_twors
        elif method == "cim":
            mutation = self.mutation_cim
        else:
            mutation = self.mutation_rsm
            warnings.warn(" \n The mutation method: '" + method + "' is unknown. \n Mutation method rsm is selected instead."
                                                               "Available crossover methods are 'pmx' and 'ox'. \n",
                          UserWarning)
        return mutation

    def mutation_twors(self, child, p_mut):
        '''
        TWORS Mutation method

        :param child: Chromosome to mutate
        :param p_mut: Mutation probability
        :return: Mutated child
        '''
        rnd = random.random()
        if rnd < p_mut:
            city1 = random.randint(0, len(child) - 1)
            city2 = random.randint(0, len(child) - 1)
            child[city1], child[city2] = child[city2], child[city1]

        return child

    def mutation_cim(self, child, p_mut):
        '''
        Centre inverse mutation method

        :param child: Chromosome to mutate
        :param p_mut: Mutation probability
        :return: Mutated child
        '''
        # Centre inverse mutation
        rnd = random.random()
        if rnd < p_mut:
            point = random.randint(0, len(child) - 1)
            child = list(reversed(child[0:point])) + list(reversed(child[point::]))
        return child

    def mutation_rsm(self, child, p_mut):
        '''
        Reverse Sequence Mutation method

        :param child: Chromosome to mutate
        :param p_mut: Mutation probability
        :return: Mutated child
        '''
        rnd = random.random()
        if rnd < p_mut:
            points = np.random.randint(0, len(child), (2))
            points.sort(axis=0)
            child[points[0]:points[1] + 1] = list(reversed(child[points[0]:points[1] + 1]))

        return child

    def mutation_psm(self, child, p_mut):
        '''
        Partial Shuffle Mutation method

        :param child: Chromosome to mutate
        :param p_mut: Mutation probability
        :return: Mutated child
        '''
        for i in range(0, len(child)):
            rnd = random.random()
            if rnd < p_mut:
                j = random.randint(0, len(child) - 1)
                child[i], child[j] = child[j], child[i]

        return child

    def check_criteria(self, gen_num, max_gen, best_this_gen, stall_limit):
        '''
        Checks if stopping criteria is met.
        Function includes two criteras:
            1. Max generations (iterations)
            2. Result has stalled for the last "stall_limit" iterations

        :param gen_num: Current generation number
        :param max_gen: Maximum number of generations
        :param best_this_gen: Shortest distance for this generation
        :param stall_limit: Limit for number of stalled generations before stopping
        :return: True for criteria is met (GA should stop) and false otherwise.
        '''
        criteria = False

        # Checks if max generations is reached
        if gen_num >= max_gen:
            criteria = True

        # Checks if GA has stalled (converged)
        if best_this_gen < self.prev_best:
            self.stall_counter = 0
            self.prev_best = best_this_gen
        else:
            self.stall_counter += 1
            if self.stall_counter > stall_limit:
                criteria = True

        return criteria

    def plot_best_generations(self):
        '''
        Plots a graph that shows the best route for each generation.
        '''
        plt.figure()
        plt.plot(list(range(0, len(self.best_each_gen))), self.best_each_gen)
        plt.xlabel("Generation number [n]")
        plt.ylabel("Distance [km]")
        plt.title("Best route each generation")
        plt.show()


    def plot_avg_generations(self):
        '''
        Plots a graph that shows the average route for each generation
        '''
        plt.figure()
        plt.plot(list(range(0, len(self.avg_each_gen))), self.avg_each_gen)
        plt.xlabel("Generation number [n]")
        plt.ylabel("Distance [km]")
        plt.title("Avg route each generation")
        plt.show()

    def plot_best_route(self, title = None):
        '''
        Plots the best route on the map - shows the tour (line between cities).

        :param title: Title on the plot
        '''
        route = self.best_route

        plt.figure()
        x, y = zip(*self.map)
        plt.scatter(x, y, s=25, edgecolors='black')
        x_route = [x[i] for i in route]
        y_route = [y[i] for i in route]
        x_route.append(x[route[0]])
        y_route.append(y[route[0]])
        plt.plot(x_route, y_route, color="red", linewidth=1)

        if title is None:
            plt.title("Tour through cities")
        else:
            plt.title("Tour through cities in " + title)
        plt.show()

    def initialise_while_running_plot(self, route, title = None):
        '''
        Initialises real-time plot for the routes.

        :param route: Initial route to plot
        :param title: Title of the plot
        :return: axis and figured - used for updating the plot during GA
        '''
        x, y = zip(*self.map)
        x_route = [x[i] for i in route]
        y_route = [y[i] for i in route]
        x_route.append(x[route[0]])
        y_route.append(y[route[0]])

        plt.ion()
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=25, edgecolors='black')
        ax.plot(x_route, y_route, color="red", linewidth=1)

        return ax, fig

    def show_best_while_running(self, route):
        '''
        Updates graph showing the best route for each generation.

        :param route: New route to draw
        '''
        ax, fig = self.plotting_info

        x, y = zip(*self.map)
        x_route = [x[i] for i in route]
        y_route = [y[i] for i in route]
        x_route.append(x[route[0]])
        y_route.append(y[route[0]])

        ax.clear()
        ax.scatter(x, y, s=25, edgecolors='black')
        ax.plot(x_route, y_route, color="red", linewidth=1)
        fig.canvas.draw()

        self.plotting_info = (ax, fig)

    '''
    ----------------------------------------------------------
    Functions for inversion technique
    ----------------------------------------------------------
    '''

    def decode(self, inv):
        '''
        Decodes from the alternative chromosome encoding to the normal representation.
        Uses the method presented in this paper:
                - http://user.ceng.metu.edu.tr/~ucoluk/research/publications/tspnew.pdf

        :param inv: Inverted chromosome
        :return: Decoded chromosome
        '''
        tour = [None] * len(inv)
        pos = [0] * len(tour) * len(tour)
        for i in reversed(range(1, len(tour) + 1)):
            for j in range(1, len(tour) + 1):
                m = i + j
                if pos[m - 1] >= (inv[i - 1] + 1):
                    pos[m - 1] = pos[m - 1] + 1
                pos[i - 1] = inv[i - 1] + 1

        for i in range(1, len(tour) + 1):
            tour[pos[i - 1] - 1] = i

        tour = [x - 1 for x in tour]

        return tour

    def encode(self, tour):
        '''
        Encodes from the normal representation to the alternative chromosome encoding.
        Uses the method presented in this paper:
                - http://user.ceng.metu.edu.tr/~ucoluk/research/publications/tspnew.pdf

        :param tour: Tour (chromosome) - List of city index
        :return: Encoded chromosome
        '''
        tour = [x + 1 for x in tour]

        inv = [None] * len(tour)
        for i in range(1, len(tour) + 1):
            inv[i - 1] = 0
            m = 1
            while tour[m - 1] != i:
                if tour[m - 1] > i:
                    inv[i - 1] = inv[i - 1] + 1
                m = m + 1
        return inv

    def crossover_sp(self, parent1, parent2, p_cross):
        '''
        Singe-point crossover. Takes one section from parent 1 and the other from parent 2 into child 1.
        Crossover happens with the given probability p_cross.

        :param parent1: First parent chromosome
        :param parent2: Seconds parent chromosome
        :param p_cross: Probability for crossover happening
        :return: Two new decoded childs
        '''
        parent1 = self.encode(parent1)
        parent2 = self.encode(parent2)
        rnd = random.random()
        if rnd < p_cross:
            cp = random.randint(0, len(parent1))
            child1 = parent1[0:cp] + parent2[cp::]
            child2 = parent2[0:cp] + parent1[cp::]
        else:
            child1 = parent1
            child2 = parent2

        child1 = self.decode(child1)
        child2 = self.decode(child2)
        return child1, child2

    # Two-point crossover
    def crossover_2p(self, parent1, parent2, p_cross):
        '''
        2-point crossover. Takes two section from parent 1 and the other from parent 2 into child 1.
        Crossover happens with the given probability p_cross.

        :param parent1: First parent chromosome
        :param parent2: Seconds parent chromosome
        :param p_cross: Probability for crossover happening
        :return: Two new decoded childs
        '''
        parent1 = self.encode(parent1)
        parent2 = self.encode(parent2)
        rnd = random.random()
        if (rnd < p_cross):
            points = np.random.randint(0, len(parent1), (2))
            points.sort(axis=0)
            child1 = parent1[0:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]::]
            child2 = parent2[0:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]::]
        else:
            child1 = parent1
            child2 = parent2

        child1 = self.decode(child1)
        child2 = self.decode(child2)
        return child1, child2

    def crossover_uniform(self, parent1, parent2, p_cross):
        '''
        Uniform crossover.
        This method goes through every gene, and determines if it should be inherited from parent 1 or 2.
        If the probability is set to 0.5, each gene would have 50% chance of being from parent 1.

        :param parent1: First parent chromosome
        :param parent2: Seconds parent chromosome
        :param p_cross: Probability for crossover happening
        :return: Two new decoded childs
        '''
        parent1 = self.encode(parent1)
        parent2 = self.encode(parent2)

        child1 = [None] * len(parent1)
        child2 = [None] * len(parent2)
        for i in range(0, len(parent1)):
            rnd = random.random()
            if rnd > p_cross:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                child1[i] = parent2[i]
                child2[i] = parent1[i]

        child1 = self.decode(child1)
        child2 = self.decode(child2)
        return child1, child2

    def get_tsp_data(self, dataset):
        '''
        Returns a dataset from http://www.math.uwaterloo.ca/tsp/world/countries.html

        Included datasets can be retireved by using the input parameter wi29, dj38 or qa194.

        :param dataset:  wi29, dj38 or qa194
        :return: List of tuples, each tuple represent a city location in the form (x, y)
        '''
        if dataset == "wi29":
            file = open("tspdata/wi29.txt", "r")
            optimal_route = 27603
        elif dataset == "dj38":
            file = open("tspdata/dj38.txt", "r")
            optimal_route = 6656
        elif dataset == "uy734":
            file = open("tspdata/uy734.txt", "r")
            optimal_route = 79114
        else:
            file = open("tspdata/qa194.txt", "r")
            optimal_route = 9352
        map = []
        for line in file.readlines():
            ind, y, x = line[0:-1].split(" ")
            map.append((float(x), float(y)))
        return map, optimal_route

    def get_tsp_data_help(self):
        '''
        Help funtion for getting information about the included datasets.

        :return: List of possible inputs to teh get_tsp_data functions.
        '''
        print("Four datasets are included in this package, \n"
              "they are collected from: http://www.math.uwaterloo.ca/tsp/world/countries.html. \n"
              " - For Western Sahara use input 'wi29' \n"
              " - For Djibouti use input 'dj38' \n"
              " - For Qatar use input 'qa194' \n \n "
              "Possible inputs are returned as a list of strings")
        return ["wi29", "dj38", "qa194"]

    def how_to_use(self):
        help_string = "------ HOW TO USE ------- \n" \
                      "This is a small guide explaining how to use the GA_TSP package. \n \n " \
                      "GA_TSP is a class that can solve TSP, designed and tested for different " \
                      "travelling salesman problems. \n \n" \
                      "EXAMPLE USE: \n" \
                      " # Make instance of the class: \n" \
                      "     - ga = GA_TSP() \n" \
                      " # Run the ga with the runGA function: \n" \
                      "     - ga.runGA([(1,2),(2,3),(7,8),(12,1),(12,12)], 1000, 100, 200, 5, 0.7, 0.05) \n \n" \
                      "How to use the class: \n" \
                      "1. Instantiate an object of this class GA_TSP \n" \
                      "2. To run the GA use the function: runGA() \n" \
                      "The runGA() function has several input parameters, some of them are optional. \n" \
                      "The possible input parameters are \n \n:" \
                      "map: List of tuples that contains x and y position of all the cities. " \
                      "Format: [(x1, y1), (x2, y2), ..., (xn, yn)] \n" \
                      "genLimit: Integer: Generation limit, which is the limit of where the GA will stop. \n" \
                      "stallLimit: Integer: Number of consecutive iterations without improvement " \
                      "that will terminate the GA \n" \
                      "popSize: Integer: Population Size \n" \
                      "nStops (optional): Integer: nCities to travel to. If not defined all cities in " \
                      "the map will be considered \n" \
                      "pCross (optional): Float [0-1]: Probability of doing crossover. Default = 0.7 \n" \
                      "pMut (optional): Float [0-1]: Probability of doing mutation. Default = 0.05 \n" \
                      "eliteRatio (optional): Float [0-1]: Ratio of the best chromosomes that should be copied " \
                      "to next generation. Default = 0.2) \n" \
                      "optimalRoute (optional): Float: Distance of the best route. Default = None \n" \
                      "crossoverMethod (optional): String: Name of the crossover method to use. Included: pmx and ox." \
                      " Default = pmx \n" \
                      "mutationMethod (optional): String: Name of the mutation method to use. " \
                      "Included: twors, rsm, cim, psm. Default = rsm \n" \
                      "plotting (optional): Boolean: Real-time plotting of best route each generation. " \
                      "Reduces effectiveness. Default = False \n" \
                      "map_name (optional): String with the name of the cities locations, for instance: " \
                      "Norway.  Default = None \n"
        print(help_string)

    def help(self):
        self.how_to_use()

# Copyright 2018, Magnus Gribbestad, All rights reserved.
# For questions or permission usage contact me at magnus@gribbestad.no