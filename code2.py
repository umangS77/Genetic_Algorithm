import numpy as np
import matplotlib.pyplot as plt
import client as server
import random
import copy
# %matplotlib inline

# constants
TEAM_ID = 'F7U8r4Y2YN0EljgonlgClHUjOIQkHpbnwCcCRi2pTo2GK1m4EZ'
GENOME_LOWER_LIMIT = -10
GENOME_UPPER_LIMIT = 10

MUTATION_PROBABILITY = 0.5
MAX_MUTATE_DEC = 7000 # Used for controlling the effect of noise, lesser it is more it effects the population

GENERATION_COUNT = 3

INITIAL_WEIGHTS = [ 0.00000000e+00 , -1.45833746e-12 , -2.29005468e-13 , 4.61855199e-11 , -1.75253755e-10 , -1.83789231e-15 , 8.53222836e-16 , 2.29379808e-05 , -2.05033812e-06 , -1.59836679e-08 , 9.97340702e-10]

# len(INITIAL_WEIGHTS)
# POPULATION_SIZE = 8

class Darwin:
    '''
    This class encapsulates a genetic algorithm runner over population consisting of weight vectors
    '''
    CHROMOSOME_SIZE = 11
    POPULATION_SIZE = 4

    def __init__(self, val: list):
        if len(val) != self.CHROMOSOME_SIZE:
            raise ValueError

        self.population = self.generate_random_from_seed(val)
        self.population[0] = np.array(val)
        self.avg_fitness = [] # maintained across generations
        self.weight_fitness_map = [] # consists of tuples (vector,fitness)
        self.max_fitness_across_generations = None
        self.best_chromosome_across_generations = None
        self.train_for_best_chromosome = None
        self.valid_for_best_chromosome = None
        

    def generate_random_from_seed(self, val: list) -> np.ndarray:
        '''
        Generates a population from a single seed chromosome
        '''
        if len(val) != self.CHROMOSOME_SIZE:
            raise ValueError
        temp = [list(val) for i in range(self.POPULATION_SIZE)]
        temp = np.array(temp, dtype=np.double)
        temp =  self.mutate(temp)
        temp[0] = val
        return temp

    def Sort_Tuple(self, tup):
        tup.sort(key = lambda x: x[1],reverse = True)  
        return tup  
        
    def get_fitness(self):
        
        def error_to_fitness(train_err, valid_err):
            return -(1.2*train_err + valid_err)
        
        fitness = []
        train_errors = []
        valid_errors = []
        weight_fitness = []
        for chrom in self.population:
            
            train_err, valid_err = server.get_errors(TEAM_ID, list(chrom))
            fit = error_to_fitness(train_err, valid_err)
                
            fitness.append(fit)
            print(chrom)
            print(fit)
            train_errors.append(train_err)
            valid_errors.append(valid_err)
            weight_fitness.append((chrom,fit))

        fitness = np.array(fitness, dtype=np.double)
        self.weight_fitness_map = weight_fitness
        return fitness, train_errors, valid_errors

    @classmethod
    def crossover(self, mom: np.ndarray, dad: np.ndarray):
        '''
        Generates offsprings out of the two parents
        '''
            
        def random_prefix(mom: np.ndarray, dad: np.ndarray):
            '''
            Random prefixes and suffixes
            '''
            thresh = np.random.randint(self.CHROMOSOME_SIZE) 
            alice = np.copy(dad)
            bob = np.copy(mom)
            alice[0:thresh] = mom[0:thresh]
            bob[0:thresh] = dad[0:thresh]
            return alice,bob

        return random_prefix(mom, dad)
    
    def normal_crossover(self, mom: np.ndarray, dad:np.ndarray):
        thresh = np.random.randint(CHROMOSOME_SIZE)
        child = np.copy(dad)
        child[0:thresh] = mom[0:thresh]
        return child

    @classmethod
    def mutate(self, population: np.ndarray):
        '''
        Mutates the population randomly
        '''
        temp_pop = copy.deepcopy(population)
        def add_gauss_noise(population: np.ndarray):
            means = np.mean(population, axis=0) # mean of each gene across the population - to keep mutations of level
            
            for idx, val in np.ndenumerate(population):
                if np.random.random() < MUTATION_PROBABILITY:
                    noise = np.random.normal(loc=means[idx[1]], scale=abs(means[idx[1]]/1000))
                    population[idx] += noise

            return np.clip(population, GENOME_LOWER_LIMIT, GENOME_UPPER_LIMIT)
        
        def add_uniform_noise(population: np.ndarray):
            for idx, val in np.ndenumerate(population):
                if np.random.random() < MUTATION_PROBABILITY:
                    range_lim = val/MAX_MUTATE_DEC
                    noise = np.random.uniform(low=-range_lim, high=range_lim)
                    population[idx] = population[idx] + noise
            return np.clip(population, GENOME_LOWER_LIMIT, GENOME_UPPER_LIMIT)
            
        temp = add_uniform_noise(population)
            
        return temp

    def breed(self):
        '''
        Creates the next generation
        '''
        def russian_roulette():
            def get_parent_index(thresholds):
                draw = np.random.random() # in [0, 1)

                for i in range(len(thresholds)):
                    if draw < thresholds[i]:
                        return i
                return len(thresholds) - 1    
            
            fitness, train_errors, valid_errors = self.get_fitness()
            normalized_fitness = (fitness - np.min(fitness)) / np.ptp(fitness) # in [0,1]
            
            self.avg_fitness.append(np.mean(fitness))
            self.update_best(fitness, train_errors, valid_errors)

            thresholds = []
            thresh = 0.0
            fitness_sum = np.sum(normalized_fitness)
            for val in normalized_fitness:
                thresh = thresh + (val/fitness_sum)
                thresholds.append(thresh)

            offsprings = []
            for i in range(int(self.POPULATION_SIZE/2)):
                mom = self.population[get_parent_index(thresholds)]
                dad = self.population[get_parent_index(thresholds)]

                alice, bob = self.crossover(mom, dad)
                offsprings.append(alice)
                offsprings.append(bob)
                
            return np.array(offsprings, dtype=np.double)
        
        def Sort_Tuple(tup):
            tup.sort(key = lambda x: x[1])  
            return tup  
        
        
        def normal_breed():
            fitness,train,valid = self.get_fitness()
            self.weight_fitness_map = Sort_Tuple(self.weight_fitness_map)
            self.weight_fitness_map.reverse()
            normalized_fitness = (fitness - np.min(fitness)) / np.ptp(fitness) # in [0,1]
            
            self.avg_fitness.append(np.mean(fitness))
            self.update_best(fitness,train,valid)
            
            offsprings = []
            for i in range(4):
                for j in range(i+1,4):
                    mom = self.weight_fitness_map[i][0]
                    dad = self.weight_fitness_map[j][0]
                    
                    alice = self.normal_crossover(mom,dad)
                    offsprings.append(alice)
            
            offsprings.append(self.weight_fitness_map[0][0])
            offsprings.append(self.weight_fitness_map[1][0])
            
            return np.array(offsprings, dtype=np.double)
        
        offsprings = russian_roulette()
        self.population = self.mutate(offsprings)
    
    def update_best(self, fitness: np.ndarray, train_errors: list, valid_errors: list):
        '''
        Updates the best chromosome across generations parameter from self.population
        '''
        best_idx = np.argmax(fitness)
        if (not self.max_fitness_across_generations) or fitness[best_idx] > self.max_fitness_across_generations:
            self.max_fitness_across_generations = fitness[best_idx]
            self.best_chromosome_across_generations = self.population[best_idx]
            self.train_for_best_chromosome = train_errors[best_idx]
            self.valid_for_best_chromosome = valid_errors[best_idx]

    def train(self):
        for i in range(GENERATION_COUNT):
            self.breed()
        
        # plt.plot(self.avg_fitness)
        # plt.xlabel('Generations', fontsize=12)
        # plt.ylabel('Best Fitness', fontsize=12)
        # plt.title('Best Fitness across Generations', fontsize=14)
        # plt.show()
        
        return self.best_chromosome_across_generations, self.max_fitness_across_generations, self.train_for_best_chromosome, self.valid_for_best_chromosome

darwin = Darwin(INITIAL_WEIGHTS)

best_chromosome, final_fitness, train_err, valid_err = darwin.train()

print('fitness:')
print(final_fitness,train_err,valid_err)

# best_chromosome = best_pair[0]
print('best chromosome')
print(best_chromosome)

# for parent in next_parents:
#     print('Vector')
#     print(parent[0])
#     print('Fitness:', parent[1])
#     print('Train:', parent[2], 'Val:', parent[3])
#     print('')

input()
# # to prevent submissions on running all cells

print(darwin.max_fitness_across_generations)

# status = server.submit(TEAM_ID, list(best_chromosome))
# print(status)

# plt.plot(darwin.avg_fitness)
# plt.xlabel('Generations', fontsize=12)
# plt.ylabel('Average Fitness', fontsize=12)
# plt.title('Average Fitness across Generations', fontsize=14)
# plt.show()

# vec = [-939161.1269820328,-915725.3356427758,-16865011.685222685,-58544095.1510417,-1659263.4978541355,-3265094.570422368,-7180432.299878756,-7223540.998257766]
# vec = np.array(vec)
# vec = vec - np.min(vec)
# vec = vec/(np.ptp(vec))
# print(vec)

# print(darwin.max_fitness_accross_generations)

