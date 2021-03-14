# import numpy as np
# import client
# import random

# SECRET_KEY = 'F7U8r4Y2YN0EljgonlgClHUjOIQkHpbnwCcCRi2pTo2GK1m4EZ'

# overfit_vector =    [9.944698129405788e-19, -1.3673593963253289e-12, -2.3051301190083433e-13, 4.2888566827175184e-11, -1.7544324436394257e-10, -5.714925536936938e-16, 8.306633524512434e-16, 2.98301310801296e-05, -2.1675305838908403e-06, -1.4166414687926064e-08, 9.158557288045874e-10]


# FITNESS_FACTOR = 1
# VECTOR_SIZE = 11
# POPULATION_COUNT = 8
# GENERATION_COUNT = 10
# MUTATION_PROBABILITY = 0.9
# GENOME_LOWER_LIMIT = -10
# GENOME_UPPER_LIMIT = 10


# class GA:

#     def __init__(self, val: list):

#         self.population = self.generate_population(val)
#         self.population[0] = np.array(val)
#         self.fitness = []
#         self.vector_fitness = []
#         self.max_fitness_across_generations = None
#         self.best_fit_vector_across_generations = None
#         self.train_for_best_fit_vector = None
#         self.valid_for_best_fit_vector = None
#         self.TOT_ERR = 1e16
        

#     def generate_population(self, val: list):
#         temp = [list(val) for i in range(POPULATION_COUNT)]
#         temp = np.array(temp, dtype=np.double)
#         temp =  self.mutation(temp)
#         temp[0] = val
#         return temp
#     def check_fitness(self):
#         fitness = []
#         train_errors = []
#         validation_errors = []
#         weight_fitness = []
#         for chrom in self.population:
            
#             train_error, validation_error = client.get_errors(SECRET_KEY, list(chrom))
#             fit = -(train_error * FITNESS_FACTOR + validation_error)
                
#             fitness.append(fit)
#             print(chrom)
#             print(train_error,validation_error)
#             train_errors.append(train_error)
#             validation_errors.append(validation_error)
#             weight_fitness.append((chrom,fit))

#         fitness = np.array(fitness, dtype=np.double)
#         self.vector_fitness = weight_fitness
#         return fitness, train_errors, validation_errors

#     @classmethod
#     def crossover(self, parent1: np.ndarray, parent2: np.ndarray):
#         def execute_crossover(parent1, parent2):
#             cutoff = np.random.randint(VECTOR_SIZE)
#             c1 = parent2
#             c1[0:cutoff] = parent1[0:cutoff]
#             c2 = parent1
#             c2[0:cutoff] = parent2[0:cutoff]
#             return c1,c2

#         return execute_crossover(parent1, parent2)
    
#     def single_crossover(self, parent1: np.ndarray, parent2:np.ndarray):
#         cutoff = np.random.randint(11)
#         child = parent2
#         child[0:cutoff] = parent1[0:cutoff]
#         return child

#     def mutation(self, population: np.ndarray):
#         for idx, val in np.ndenumerate(population):
#             if np.random.random() < MUTATION_PROBABILITY:
#                 range_lim = val/2000
#                 popabs = abs(population[idx])
#                 noise = np.random.uniform(low=-range_lim, high=range_lim)
#                 population[idx] = population[idx] + noise
#                 if popabs > 10:
#                     population[idx] = 10 * (popabs/population[idx])
#         return population

#     def breed(self):
#         def Sort_Tuple(tup):
#             tup.sort(key = lambda x: x[1])  
#             return tup  
        
        
#         def generate():
#             fitness,train,valid = self.check_fitness()
#             self.vector_fitness = Sort_Tuple(self.vector_fitness)
#             self.vector_fitness.reverse()            
#             self.fitness.append(np.mean(fitness))
#             self.get_max_fit_vector(fitness,train,valid)
            
#             next_gen = []
#             for i in range(4):
#                 for j in range(i+1,4):
#                     parent1 = self.vector_fitness[i][0]
#                     parent2 = self.vector_fitness[j][0]
#                     child = self.single_crossover(parent1,parent2)
#                     next_gen.append(child)
            
#             next_gen.append(self.vector_fitness[0][0])
#             next_gen.append(self.vector_fitness[1][0])
            
#             return np.array(next_gen, dtype=np.double)
        
#         next_gen = generate()
#         self.population = self.mutation(next_gen)
    
#     def get_max_fit_vector(self, fitness , train_errors: list, validation_errors: list):
#         best_idx = np.argmax(fitness)
#         if (not self.max_fitness_across_generations) or fitness[best_idx] > self.max_fitness_across_generations:
#             self.max_fitness_across_generations = fitness[best_idx]
#             self.best_fit_vector_across_generations = self.population[best_idx]
#             self.train_for_best_fit_vector = train_errors[best_idx]
#             self.valid_for_best_fit_vector = validation_errors[best_idx]
#             self.TOT_ERR = self.train_for_best_fit_vector + self.valid_for_best_fit_vector

#     def run_GA(self):
#         for i in range(GENERATION_COUNT):
#             self.breed()
#         return self.best_fit_vector_across_generations, self.max_fitness_across_generations, self.train_for_best_fit_vector, self.valid_for_best_fit_vector

# ga = GA(overfit_vector)
# best_fit_vector, final_fitness, train_error, validation_error = ga.run_GA()
# # print('fitness:')
# # print(final_fitness,train_error,validation_error)
# print('best fit vector:\n', list(best_fit_vector))
# print("{:e}".format(ga.TOT_ERR), "{:e}".format(ga.train_for_best_fit_vector), "{:e}".format(ga.valid_for_best_fit_vector))


import client
import random
import numpy as np

SECRET_KEY = 'F7U8r4Y2YN0EljgonlgClHUjOIQkHpbnwCcCRi2pTo2GK1m4EZ'

# overfit_vector = [9.94844008537157e-19, -1.3673449094175483e-12, -2.307774976787406e-13, 4.2854401512830564e-11, -1.7584139294715718e-10, -5.721186451922296e-16, 8.304875840120217e-16, 2.959497956287452e-05, -2.1821061915897267e-06, -1.4170069270276784e-08, 9.176646493561812e-10]

FITNESS_FACTOR = 1
VECTOR_SIZE = 11
POPULATION_COUNT = 4
GENERATION_COUNT = 6
MUTATION_PROBABILITY = 0.9
GENOME_LOWER_LIMIT = -10
GENOME_UPPER_LIMIT = 10


class GA:

    def __init__(self, val: list):

        self.population = self.generate_population(val)
        self.population[0] = np.array(val)
        self.max_fitness = None
        self.best_fit_vector = None
        self.train_error_for_best_fit_vector = None
        self.validation_error_for_best_fit_vector = None
        self.fitness = []
        self.vector_fitness = []
        self.TOT_ERR = 1e18
        

    def generate_population(self, val: list):
        temp = [list(val) for i in range(POPULATION_COUNT)]
        temp = np.array(temp, dtype=np.double)
        temp =  self.mutation(temp)
        MUTATION_FLAG = 1
        temp[0] = val
        return temp

    def run_GA(self):
        for i in range(GENERATION_COUNT):
            self.reproduce()
        return self.best_fit_vector, self.max_fitness, self.train_error_for_best_fit_vector, self.validation_error_for_best_fit_vector

    def check_fitness(self):
        fitness = []
        train_errors = []
        validation_errors = []
        feature_fitness = []
        for vector in self.population:
            print(vector)
            train_error, validation_error = client.get_errors(SECRET_KEY, list(vector))
            print(train_error,validation_error)
            fit = -(train_error * FITNESS_FACTOR + validation_error)
            feature_fitness.append((vector,fit))
            fitness.append(fit)
            train_errors.append(train_error)
            validation_errors.append(validation_error)
            
        fitness = np.array(fitness, dtype=np.double)
        self.vector_fitness = feature_fitness
        return fitness, train_errors, validation_errors

    

    def crossover(self, parent1, parent2):
        cutoff = np.random.randint(VECTOR_SIZE)
        c1 = parent2
        c1[0:cutoff] = parent1[0:cutoff]
        c2 = parent1
        c2[0:cutoff] = parent2[0:cutoff]
        return c1,c2
    
    def single_crossover(self, parent1: np.ndarray, parent2:np.ndarray):
        cutoff = np.random.randint(11)
        child = parent2
        child[0:cutoff] = parent1[0:cutoff]
        return child

    def mutation(self, feature: np.ndarray):
        for idx, val in np.ndenumerate(feature):
            if np.random.random() < MUTATION_PROBABILITY:
                range_lim = val/2000
                # popabs = abs(population[idx])
                noise = np.random.uniform(low=-range_lim, high=range_lim)
                feature[idx] = feature[idx] + noise
                if feature[idx] > 10.0:
                    feature[idx] = 10.0
                elif feature[idx] < -10.0:
                    feature[idx] = -10
        return feature

    def reproduce(self):
        # def Sort_Tuple(tup):
        #     tup.sort(key = lambda x: x[1])  
        #     return tup  
        
        
        # def generate():
        x=4
        fitness,train,valid = self.check_fitness()
        temp_vec = self.vector_fitness
        temp_vec.sort(key = lambda x: x[1])
        self.vector_fitness = temp_vec
        self.vector_fitness.reverse()            
        self.fitness.append(np.mean(fitness))
        self.get_max_fit_vector(fitness,train,valid)
        
        next_gen = []

        for i in range(x):
            for j in range(i+1,x):
                parent1 = self.vector_fitness[i][0]
                parent2 = self.vector_fitness[j][0]
                # child = self.single_crossover(parent1,parent2)
                next_gen.append((self.single_crossover(parent1,parent2)))
        
        next_gen.append(self.vector_fitness[0][0])
        next_gen.append(self.vector_fitness[1][0])
        next_gen = np.array(next_gen, dtype=np.double)
            # return next_gen
        
        # next_gen = generate()
        self.population = self.mutation(next_gen)

    def get_max_fit_vector(self, fitness , train_errors, validation_errors):
        idx = np.argmax(fitness)
        if (not self.max_fitness):
            self.max_fitness = fitness[idx]
            self.validation_error_for_best_fit_vector = validation_errors[idx]
            MAX_FIT_FLAG = 1
            self.best_fit_vector = self.population[idx]
            self.train_error_for_best_fit_vector = train_errors[idx]
            self.TOT_ERR = self.train_error_for_best_fit_vector + self.validation_error_for_best_fit_vector

        elif fitness[idx] > self.max_fitness:
            self.max_fitness = fitness[idx]
            # self.TOT_ERR = self.train_error_for_best_fit_vector + self.validation_error_for_best_fit_vector
            self.validation_error_for_best_fit_vector = validation_errors[idx]
            self.best_fit_vector = self.population[idx]
            self.train_error_for_best_fit_vector = train_errors[idx]
            MAX_FIT_FLAG = 1
            self.TOT_ERR = self.train_error_for_best_fit_vector + self.validation_error_for_best_fit_vector
            


MAX_FIT_FLAG = 0
ga = GA(overfit_vector)
MUTATION_FLAG = 0
best_fit_vector, final_fitness, train_error, validation_error = ga.run_GA()
# print('fitness:')
# print(final_fitness,train_error,validation_error)
print('best fit vector:\n', list(best_fit_vector))
print("{:e}".format(ga.TOT_ERR), "{:e}".format(ga.train_error_for_best_fit_vector), "{:e}".format(ga.validation_error_for_best_fit_vector))

