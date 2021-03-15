
import client
import random
import numpy as np

SECRET_KEY = 'F7U8r4Y2YN0EljgonlgClHUjOIQkHpbnwCcCRi2pTo2GK1m4EZ'

overfit_vector = [9.94844008537157e-19, -1.3673449094175483e-12, -2.307774976787406e-13, 4.2854401512830564e-11, -1.7584139294715718e-10, -5.721186451922296e-16, 8.304875840120217e-16, 2.959497956287452e-05, -2.1821061915897267e-06, -1.4170069270276784e-08, 9.176646493561812e-10]

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
        print("\n")
        print("\n")
        print("POPULATION: ")
        print(temp)
        print("\n")
        print("\n")

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

        print("parent1 = \n")
        print(parent1)
        print("parent2 = \n")
        print(parent2)
        cutoff = np.random.randint(VECTOR_SIZE)
        print("cutoff index = ", str(cutoff))
        c1 = parent2
        c1[0:cutoff] = parent1[0:cutoff]
        c2 = parent1
        c2[0:cutoff] = parent2[0:cutoff]
        print("child 1 = \n")
        print(c1)
        print("child 2 = \n")
        print(c2)
        return c1,c2
    
    def single_crossover(self, parent1: np.ndarray, parent2:np.ndarray):
        cutoff = np.random.randint(11)
        child = parent2
        child[0:cutoff] = parent1[0:cutoff]
        return child

    def mutation(self, feature: np.ndarray):
        print("\n")
        print("\n")
        print("RECEIVED FOR MUTATTION\n")
        print(feature)
        print("\n")
        print("\n")

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
        print("\n")
        print("\n")
        print("Mutated feature: \n")
        print(feature)
        print("\n")
        print("\n")

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

