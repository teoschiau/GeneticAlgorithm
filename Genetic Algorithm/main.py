import random
import numpy as np
import matplotlib.pyplot as plt

INDIVIDUALS_LEN = 25
POPULATION_SIZE = 50
MUTATION_PROBABILITY = 0.01

def create_individual():
    return [random.choice([0, 1]) for _ in range(INDIVIDUALS_LEN)]


def create_population(size):
    return [create_individual() for _ in range(size)]


def oneMax(individual):
    fit = 0
    for i in individual:
        if i == 1:
            fit += 1
    return fit


def oneAlternating(individual):
    fit = 1
    for i in range(len(individual) - 1):
        if (individual[i] == 0 and individual[i + 1] == 1) or (individual[i] == 1 and individual[i] == 0):
            fit += 1
    return fit


def calculate_fitness(population):
    fit = []
    for ind in range(POPULATION_SIZE):
        fit.append(oneMax(population[ind]))
    return fit


def selection(pop, fits):
    return random.choices(pop, fits, k=POPULATION_SIZE)


def crossover(ind1, ind2):
    point = random.randint(0, INDIVIDUALS_LEN - 1)
    o1 = ind1[:point] + ind2[point:]
    o2 = ind1[point:] + ind2[:point]
    return o1, o2


def population_crossover(pop, prob):
    offsprings = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        o1 = p1
        o2 = p2
        if random.random() < prob:
            o1, o2 = crossover(p1, p2)
        offsprings.append(o1)
        offsprings.append(o2)
    return offsprings


def mutation(individual):
    return [1 - i if random.random() < MUTATION_PROBABILITY else i for i in individual]


def operators(pop, cprob, mprob):
    new_pop = []
    new_pop = population_crossover(pop, cprob)

    if random.random() < mprob:
        new_pop = list(map(mutation, new_pop))

    return new_pop


def run_algorithm(ccrossover, cmutation):
    population = create_population(POPULATION_SIZE)
    log = []
    for g in range(50):
        fitness = calculate_fitness(population)
        log.append((g, max(fitness), sum(fitness) / 100, g* POPULATION_SIZE))
        mating_pool = selection(population, fitness)
        offsprings = operators(mating_pool, ccrossover, cmutation)
        population = offsprings
    return population, log


if __name__ == '__main__':
    logs = []
    logs1 = []
    for i in range(10):
        population, log = run_algorithm(0.02, 0.6)
        population1, log1 = run_algorithm(0.3, 0.3)
        logs.append(log)
        logs1.append(log1)

    fits = calculate_fitness(population)
    fits1 = calculate_fitness(population1)

    evals = []
    best_fit = []
    for log in logs:
        evals.append([l[3] for l in log])
        best_fit.append([l[1] for l in log])

    evals1 = []
    best_fit1 = []
    for log in logs1:
        evals1.append([l[3] for l in log])
        best_fit1.append([l[1] for l in log])

    evals = np.array(evals)
    best_fit = np.array(best_fit)

    evals1 = np.array(evals1)
    best_fit1 = np.array(best_fit1)

    plt.title("Fitness Progression of Two Populations Over Generations")

    plt.plot(evals[0, :], np.median(best_fit, axis=0), color='red')
    plt.plot(evals1[0, :], np.median(best_fit1, axis=0), color='blue')

    plt.fill_between(evals[0, :], np.percentile(best_fit, q=25, axis=0),
                     np.percentile(best_fit, q=75, axis=0), alpha=0.2, color='red')
    plt.fill_between(evals1[0, :], np.percentile(best_fit1, q=25, axis=0),
                     np.percentile(best_fit1, q=75, axis=0), alpha=0.2, color='blue')

    plt.show()


"""
The algorithm works by choosing a random population, then calculates the fitness for the population  choosing a fitness function, either the oneMax or
oneAlternating. Then it is created the mating pool by doing the operations of selecting 2 random individuals based on their fitness, then the crossover
and mutation. The initial population is then replaced by a better one.

In parallel I have run the algorithm with different probabilities for the mutation and crossover and the plots shows an improve in the generations. When
making the mutation probability higher and the cross probability lower, I have remarked an improvement in the new generation s, I am plotting two different generations
based on the probability of crossover and mutation.
"""