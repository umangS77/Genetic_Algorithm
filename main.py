
import client
import random
import numpy as np

SECRET_KEY = 'F7U8r4Y2YN0EljgonlgClHUjOIQkHpbnwCcCRi2pTo2GK1m4EZ'


file = open("generation_03.txt", "w")

overfit_vector = [9.917369362870062e-19, -1.3663345774890496e-12, -2.2987760470899746e-13, 5.0166628893336836e-11, -1.9137079609875392e-10, -5.707209963841367e-16, 9.219501151538645e-16, 3.2090844450778915e-05, -2.103950873147733e-06, -1.3825652924351045e-08, 8.643552922860863e-10]
# gen1 and gen2

overfit_vector = [9.920228219160712e-19, -1.364202711273918e-12, -2.3006364489300257e-13, 5.0111927990795596e-11, -1.9140452210688034e-10, -5.702463536373112e-16, 9.243845522203362e-16, 3.199653795367389e-05, -2.0985817255189582e-06, -1.3853093822021742e-08, 8.635160890129394e-10]

FITNESS_FACTOR = 2
VECTOR_SIZE = 11
POPULATION_COUNT = 8
GENERATION_COUNT = 10
MUTATION_PROBABILITY = 0.8
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
        file.write(str("\n"))
        file.write(str("\n"))
        file.write(str("POPULATION: "))
        file.write(str(str(temp)))
        file.write(str("\n"))
        file.write(str("\n"))

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
            file.write("\nsending vector for error: ")
            file.write(str(vector))
            train_error, validation_error = client.get_errors(SECRET_KEY, list(vector))
            file.write("\nreceived errors: ")
            outstr = str(train_error) + "    " + str(validation_error) + "\n"
            file.write(outstr)
            fit = -(train_error * FITNESS_FACTOR + validation_error )
            feature_fitness.append((vector,fit))
            fitness.append(fit)
            train_errors.append(train_error)
            validation_errors.append(validation_error)
            
        fitness = np.array(fitness, dtype=np.double)
        self.vector_fitness = feature_fitness
        return fitness, train_errors, validation_errors

    # def simple_cross(self, parent1: np.ndarray, parent2:np.ndarray):
    #     cutoff = np.random.randint(11)
    #     child = parent2
    #     child[0:cutoff] = parent1[0:cutoff]
    #     return child

    def crossover(self, parent1, parent2):
        file.write("\n\nCROSSOVER:\n\n")

        file.write(str("\n\tparent1 = " + str(parent1)))
        # file.write(str(parent1)
        file.write(str("\n\tparent2 = " + str(parent2)))
        # file.write(str(parent2)
        cutoff = np.random.randint(VECTOR_SIZE)
        file.write(str("\n\tcutoff index = "+ str(cutoff)))
        c1 = parent2
        c1[0:cutoff] = parent1[0:cutoff]
        c2 = parent1
        c2[0:cutoff] = parent2[0:cutoff]
        file.write(str("\n\tchild 1 = "+ str(c1)))
        # file.write(str(c1)
        file.write(str("\n\tchild 2 = "+ str(c2)))
        file.write(str("\n"))
        return c1,c2
    
    

    def mutation(self, feature: np.ndarray):
        file.write(str("\n\nMUTATION: "))
        file.write(str("\n"))
        file.write(str("\n"))
        file.write(str("RECEIVED FOR MUTATTION: "))
        file.write(str(str(feature)))
        file.write(str("\n"))
        for idx, val in np.ndenumerate(feature):
            if np.random.random() < MUTATION_PROBABILITY:
                range_lim = val/1000
                # popabs = abs(population[idx])
                noise = np.random.uniform(low=-range_lim, high=range_lim)
                feature[idx] = feature[idx] + noise
                if feature[idx] > 10.0:
                    feature[idx] = 10.0
                elif feature[idx] < -10.0:
                    feature[idx] = -10
        file.write(str("\nMutated feature:"))
        file.write(str(str(feature)))
        file.write(str("\n\n"))
        # file.write(str("\n")

        return feature

    def reproduce(self):
        
        limit_point = []
        next_gen = []
        thresh = 0.0    
        z = int(POPULATION_COUNT/2)

        
        fitness, train_errors, valid_errors = self.check_fitness()
        normalized_fitness = (fitness - np.min(fitness)) / np.ptp(fitness)
        tot_fit = np.sum(normalized_fitness)
        
        self.fitness.append(np.mean(fitness))
        self.get_max_fit_vector(fitness, train_errors, valid_errors)

        for v in normalized_fitness:
            thresh = thresh + (v/tot_fit)
            limit_point.append(thresh)

        for i in range(z):
            p1 = self.population[self.get_parent(limit_point)]
            p2 = self.population[self.get_parent(limit_point)]

            parent1, parent2 = self.crossover(p1, p2)
            next_gen.append(parent1)
            next_gen.append(parent2)
            
        next_gen =  np.array(next_gen, dtype=np.double)
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
            
    def get_parent(self,limit_point):
        v=-1
        random_num = np.random.random() 
        for i in range(len(limit_point)):
            if random_num < limit_point[i]:
                v=i
                break
        if v == -1:
            return len(limit_point) - 1
        else:
            return v

MAX_FIT_FLAG = 0
ga = GA(overfit_vector)
MUTATION_FLAG = 0
best_fit_vector, final_fitness, train_error, validation_error = ga.run_GA()
# file.write(str('fitness:')
# file.write(str(final_fitness,train_error,validation_error)
file.write(str('best fit vector:\n'+ str(list(best_fit_vector))))
# file.write(str("error:")
file.write(("\nerror:" +"    "+ str("{:e}".format(ga.train_error_for_best_fit_vector)) +"    "+ str("{:e}".format(ga.validation_error_for_best_fit_vector))))

file.close()