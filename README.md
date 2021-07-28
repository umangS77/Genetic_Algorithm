# Genetic Algorithm Project

Genetic Algorithm (GA) is a search-based optimization technique based on the principles of Genetics and Natural Selection. It is frequently used to find optimal or near-optimalsolutions to difficult problems which otherwise wouldtake a lifetime to solve. It is frequently used to solve optimization problems, in research, and in machine learning. We have a pool or population of potential solutions to a problem in GAs. These solutions are then subjected to recombination and mutation (as in natural genetics), resulting in the birth of new offspring, and the process is replicated over centuries. Each individual (or candidate solution) is given a fitness value (based on its objective function value), and the fitter ones have a better chance of mating and producing more “fitter” individuals. This is consistent with Darwin's "Survival of the Fittest" principle.

## Summary of our Project

### Step 1: Import the modules
We import the modules necessary in this project. The client module helps us to connect to the server and get results (train error and validation error) for a submitted vector. It also helps to submit the final best fit vector.

### Step 2: Initialize variables
```
SECRET_KEY ='F7U8r4Y2YN0EljgonlgClHUjOIQkHpbnwCcCRi2pTo2GK1m4EZ'# the secret key provided to us (Team 44)
overfit_vector = [9.94844008537157e-19,-1.3673449094175483e-12,-2.307774976787406e-13,4.2854401512830564e-11,-1.7584139294715718e-10,-5.721186451922296e-16,8.304875840120217e-16,2.959497956287452e-05,-2.1821061915897267e-06,-1.4170069270276784e-08,9.176646493561812e-10]
FITNESS_FACTOR =1.2 # the multiplication factor usedin determining the fitness by adding both errors
FRACTION =100 # the fraction of value to be added as noise for mutation
ABS_FACTOR =0.5 # the level of contribution of absdifference of train and validation error in fitness
VECTOR_SIZE =11 # the size of the feature vector
POPULATION_COUNT =8 # the number of populations generated
GENERATION_COUNT =15 # the number of generations generated for each population
MUTATION_PROBABILITY =0.8 # the probability of mutation
GENOME_LOWER_LIMIT =-10 # the upper value limit for a genome
GENOME_UPPER_LIMIT =10 # the lower value limit for a genome
```

### Step 3: Defining the Genetic Algorithm functioning code
The classGAcontains the functioning code of theGenetic Algorithm.
### Step 4: Initialize Population
```
def generate_population(self, val: list):
temp = [list(val)foriinrange(POPULATION_COUNT)]
temp = np.array(temp, dtype=np.double)
temp =  self.mutation(temp)
temp[0] = valreturntemp
```
We define a list temp which contains the population generated and fill it with the original population and its mutation.

### Step 5: Run the Genetic Algorithm
We reproduce the given overfit vector, using crossover and mutation processes and return the best fit vector, maximum fitness, trainerror and validation error for best fit vector. We use the Fitness proportionate selection for child generation. Fitness proportionate selection also known as roulette wheel selection, is a genetic operator used in genetic algorithms for selecting potentially useful solutions for recombination.

### Step 6: Check the fitness of generated children vectors
This is one the most important parts of the project. This is where the fitness of a particular vector is checked. We changed the main fitness criteria(fit = -(train_error *FITNESS_FACTOR + validation_error)) several times during the project.

### Step 7: Crossover of the parent population
After the fitnesses are obtained, we sort the parent vector according to the fitness and select the two fittest parents for crossover. We cross them with each other to generate new offspring. The cutoff (random) recognises the point where we cut the two parent chromosomes.

### Step 8: Obtain the fitness for the child population and create next best generation.
We find the fitness of the child vectors, which is then sorted. Then we get the max fit vector to use it to create further generations. From the best fit children, we create the next generation of population.

### Step 9: Mutation of a new generation
For mutation, we add some random noise to the obtained values and check if the value is between 10 to -10.

### Step 10: Print the best fit vector
Finally we get the best fit vector and print it to check the results.

## Fitness Function
We use a linear fitness function:
```
fit = -(train_error * FITNESS_FACTOR + validation_error+ABS_FACTOR*(abs(train_error - validation_error)))
```
Initially, we started with the function as fit = -(train_error+ validation_error). Negative sign here for sorting purpose. We achieved some decrease in error but after some point it became somewhat stagnant with a large difference in train and validation error. Then to reduce the difference in the train and validation errors, we tried to reduce the weightage of train error in total fitness. Therefore we multiply it by a FITNESS_FACTOR.The fitness factor ranged from 0.4 to 0.9. We tried different values,each time getting strange results. For 0.4,initially we saw a decrease in difference but then the train error started increasing. This was due to less contribution of train error in final fitness.Then after trying several values, we found 0.75 a better fitness factor than rest. In later stages, we found our train error to be more than the fitness error by a large margin.Then we had to set FITNESS_FACTOR ranging from 1.5 to 3.5 to decrease the difference in errors. We often reached stagnant error values with no further change. So, we also tried some other fitness functions. We tried the Sphere function(summation of square of values) but it didn't work out. We also tried Rosenbrock function for a change but did not obtain any productive result. Another change we did was to add the absolute difference of train and validation error to reduce the difference between them. It helped us to some extent, along with general fitness function.


## Crossover Function
For crossover, our function is simple. We choose a random cutoff point out of the given VECTOR_SIZE and form child by taking parent1[0:cutoff] and rest from parent2 for child1 and similar for child2.

## Mutation Function
To mutate a value , first we check its mutation probability. We define a variable as MUTATION_PROBABILITY through which we can control the probability of mutation. If a random value is less than mutation probability, we add a noise in the value to be mutated. We decide a range limit range_lim as a fraction of the value. Then the noise is randomly chosen from the range of -range_lim to range_lim. We also cap the values as between -10 to 10 as asked in the question pdf. We were initially following the provided instructions: mutate with a very small probability, and to mutate, choose a random index and replace the value of the index's coefficient with a totally random value. However, we quickly realised that this was not a good method. Following that, we attempted several mutation probabilities, each time getting a pattern of results before the obtained results diverted from the required result. We also ranged the fraction of the value to be used as noise, ranging from 0.1%  to 10%.

## Heuristics Applied
- Using a FITNESS_FACTOR in fitness function: As explained earlier, we use a fitnessfactor to multiply it and handle the contribution of train and validation error in total fitness.
- Adding absolute difference of train error and validation error to fitness: As explained earlier, to reduce the difference in train and absolute error , we added the absolute difference between them to the fitness function.
- Simulated Annealing: We tried to determine the mutation probability for each gene randomly but it did not work out for us.
- Taking Mean of generated vectors: Often when the errors got stagnant to us, we tried to take the mean of several past generations as well as poor fitness generations. This was a boon for us as it greatly decreased the errors we were receiving. We then thought of a strange method. When stuck, just make a new vector by adding some past and/or poor fitness vectors in some ratios and process the resultant vectors through the Generic Algorithm. It worked tremendously well for us.
