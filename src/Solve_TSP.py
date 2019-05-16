'''
This is a simple script that shows some example use of the GA_TSP package
'''

from GA_TSP import GA_TSP

def load_map(dataset):
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

def test_ga_methods(ga):
    '''
    Big loop that tests different combinations of crossover and mutation methods.
    '''
    settings = [['pmx', 'twors'],
                ['pmx', 'cim'],
                ['pmx', 'psm'],
                ['pmx', 'rsm'],
                ['ox', 'twors'],
                ['ox', 'cim'],
                ['ox', 'psm'],
                ['ox', 'rsm']]

    settings = [['pmx', 'rsm']]

    cities, optimal_route = load_map("qa194")
    gen_limit = 10000
    gen_for_converged = 1000
    pop_size = 1000
    nStops = 194  # 29 # 38 # 194
    p_cross = 0.7
    p_mut = 0.05
    elite_ratio = 0.2

    for setting in settings:
        print("------------------------------    Start    -----------------------------------------")
        best_routes = []
        differences = []
        differences_per = []
        times = []
        for i in range(0,10):
            crossover, mutation = setting
            ga.runGA(cities, gen_limit, gen_for_converged, pop_size, nStops, p_cross, p_mut, elite_ratio, optimal_route,
                     crossover_method=crossover, mutation_method=mutation, plotting=False)
            route, distance, time = ga.get_best_route()

            best_routes.append(distance)
            times.append(time)
            diff = optimal_route - distance
            diff_per = (1-(optimal_route/(distance)))*100
            print("Difference: ", diff_per)
            differences.append(diff)
            differences_per.append(diff_per)
            ga.plot_best_route()
            ga.plot_best_generations()


        print("-------------------------------------------------------------------------------------")
        print("Crossover: ", crossover, " Mutation: ", mutation)
        print("Best tour: ", min(best_routes), " Time: ", times[best_routes.index(min(best_routes))], " Diff: ", differences_per[best_routes.index(min(best_routes))])
        print("Average tour: ", sum(best_routes)/len(best_routes), " Time: ", sum(times)/len(times), " Diff: ", (1-(optimal_route/(sum(best_routes)/len(best_routes))))*100)
        print("-------------------------------------------------------------------------------------")


if __name__ == '__main__':
    ga = GA_TSP()

    # Simple GA example
    cities, optimal_route = load_map("wi29")
    gen_limit = 1000
    gen_for_converged = 200
    pop_size = 300
    nStops = 29  # 29 # 38 # 194
    p_cross = 0.7
    p_mut = 0.05
    elite_ratio = 0.2

    # Run GA
    ga.runGA(cities, gen_limit, gen_for_converged, pop_size, nStops, p_cross, p_mut, elite_ratio, optimal_route,
             crossover_method="pmx", mutation_method="rsm", plotting=True)
    ga.plot_best_route()
    ga.plot_best_generations()

    # Bigger GA test loop - Commented away
    # test_ga_methods(ga)

