
import numpy as np
import time
import matplotlib.pyplot as plt
from ann_criterion import optimality_criterion

class GeneticAlgorithm:

    '''
    Implementation of a genetic algorithm for solving "black-box" optimization
    problems with continuous variables.
    
    This class encapsulates all necessary components, including population
    initialization, selection, crossover, and mutation, as well as mechanisms
    for performance tracking and stopping criteria.
    
    Notes: fitness <=> fitness
           crossover <=> crossover
           generation <=> iteration

    Attributes
    ----------
    fitness_func : function
        Objective function to be minimized.
    dim : int
        Number of genes in the chromosome (problem dimensionality).
    bounds : list of tuples
        List of bounds (min, max) for each gene.
    pop_size : int, optional
        Population size (default 250).
    generations : int, optional
        Maximum number of generations (default 300).
    crossover_rate : float, optional
        Crossover probability, value between 0 and 1 (default 0.9).
    mutation_rate : float, optional
        Mutation probability per gene, value between 0 and 1 (default 0.04).
    selection_k : int, optional
        Tournament size for tournament selection (default 3).
    crossover_eta : float, optional
        Distribution index for SBX crossover (default 5).
    mutation_eta_schedule : tuple, optional
        Schedule (initial, final) values for the eta parameter of Polynomial
        mutation (default (5, 50)).
    elitism_count : int, optional
        Number of best individuals carried over to the next generation
        (default 3).
    stagnation_limit : int, optional
        Number of generations without improvement after which the algorithm stops
        (default 15).
           
    '''

    def __init__(self, fitness_func, dim, bounds, pop_size=250, generations=300, crossover_rate=0.9, mutation_rate=0.04, selection_k=3,
                 crossover_eta=5, mutation_eta_schedule=(5, 50), elitism_count=3,
                 stagnation_limit=15):
        
        self.fitness_func = fitness_func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_k = selection_k
        self.crossover_eta = crossover_eta
        self.mutation_eta_schedule = mutation_eta_schedule
        self.elitism_count = elitism_count
        self.stagnation_limit = stagnation_limit

        # Initialize population
        self.population = self._initialize_population()

        # For tracking and reporting
        self.best_individual = None
        self.best_fitness = float('inf')
        self.history = {'best_fitness': [], 'avg_fitness': [], 'diversity': []}

    def _initialize_population(self):
        """Creates initial population of random individuals within defined bounds"""
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.dim):
            low, high = self.bounds[i]
            population[:, i] = np.random.uniform(low, high, self.pop_size)
        return population

    def _evaluate_population(self, population):
        """
        Evaluates fitness of each individual in the population
        Problem is minimization, so fitness is transformed: f(x) = -f(x)
        """
        objective_values = np.array([self.fitness_func(ind) for ind in population])
        # Transformation for fitness maximization
        fitness_scores = (-1.0) * objective_values
        return fitness_scores, objective_values

    def _selection(self, fitness_scores):
        """
        Performs parent selection using tournament selection and elitism
        """
        # Elitism - save the best individuals
        elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
        elites = self.population[elite_indices]

        # Tournament selection for the rest of population
        parents = []
        num_parents_to_select = self.pop_size - self.elitism_count
        for _ in range(num_parents_to_select):
            tournament_indices = np.random.randint(0, self.pop_size, self.selection_k)
            tournament_fitness = fitness_scores[tournament_indices]
            winner_index_in_tournament = np.argmax(tournament_fitness)
            winner_index_in_population = tournament_indices[winner_index_in_tournament]
            parents.append(self.population[winner_index_in_population])

        return elites, np.array(parents)
    
    def _crossover(self, parents):
        """
        Performs parent crossover using Simulated Binary Crossover (SBX) method
        """
        offspring = []
        # Ensure even number of parents for pairing
        if len(parents) % 2 != 0:
            parents = np.vstack([parents, parents[np.random.randint(0, len(parents))]])

        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[i + 1]
            child1, child2 = p1.copy(), p2.copy()

            if np.random.rand() < self.crossover_rate:
                for j in range(self.dim):
                    # Continue if the genes are indentical
                    if abs(p1[j] - p2[j]) < 1e-14:
                        continue

                    # Calculate the beta (β) factor
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (self.crossover_eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.crossover_eta + 1.0))

                    # Use SBX for generating genes of childrens
                    c1_j = 0.5 * (((1 + beta) * p1[j]) + ((1 - beta) * p2[j]))
                    c2_j = 0.5 * (((1 - beta) * p1[j]) + ((1 + beta) * p2[j]))

                    # Make sure that the values are in the bounds
                    child1[j] = np.clip(c1_j, self.bounds[j][0], self.bounds[j][1])
                    child2[j] = np.clip(c2_j, self.bounds[j][0], self.bounds[j][1])

            offspring.extend([child1, child2])

        return np.array(offspring)

    def _mutation(self, offspring, current_eta):
        """
        Performs Polynomial mutation on offspring
        """
        for i in range(len(offspring)):
            for j in range(self.dim):
                if np.random.rand() < self.mutation_rate:
                    # Original gene value
                    y = offspring[i, j]
                    # Bound for gene
                    low, high = self.bounds[j]
                    
                    # Calculate perturbation (delta), degree of change
                    delta1 = (y - low) / (high - low)
                    delta2 = (high - y) / (high - low)
                    
                    rand_num = np.random.rand()
                    mut_pow = 1.0 / (current_eta + 1.0)
                    
                    if rand_num < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand_num + (1.0 - 2.0 * rand_num) * (xy ** (current_eta + 1.0))
                        delta_q = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand_num) + 2.0 * (rand_num - 0.5) * (xy ** (current_eta + 1.0))
                        delta_q = 1.0 - val ** mut_pow
                        
                    # Mutation
                    y = y + delta_q * (high - low)
                    
                    # Make sure it is in the bounds
                    offspring[i, j] = np.clip(y, low, high)
                    
        return offspring

    def run(self):
        """
        Main loop of Genetic Algorithm
        """
        print("Starting the Genetic Algorithm...")
        start_time = time.time()
        stagnation_counter = 0
        initial_eta, final_eta = self.mutation_eta_schedule

        for generation in range(self.generations):

            progress = generation / self.generations
            current_mutation_eta = initial_eta + (final_eta - initial_eta) * progress

            # 1. Evaluation
            fitness_scores, objective_values = self._evaluate_population(self.population)

            # 2. Track results
            current_best_idx = np.argmin(objective_values)
            current_best_objective = objective_values[current_best_idx]

            if current_best_objective < self.best_fitness:
                self.best_fitness = current_best_objective
                self.best_individual = self.population[current_best_idx]
                stagnation_counter = 0  # Reset the stagnation counter
            else:
                stagnation_counter += 1

            avg_objective = np.mean(objective_values)
            diversity = np.std(objective_values)
            self.history['best_fitness'].append(self.best_fitness)
            self.history['avg_fitness'].append(avg_objective)
            self.history['diversity'].append(diversity)

            if (generation + 1) % 10 == 0:
                print(f"Generation: {generation + 1}/{self.generations} | "
                      f"Best fitness: {self.best_fitness:.6f} | "
                      f"Average fitness: {avg_objective:.6f} | "
                      f"Diversity: {diversity:.6f}")

            # 3. Check the criterium for stoping
            if stagnation_counter >= self.stagnation_limit:
                print(f"\nTermination due to stagnation after {generation + 1} generations.")
                break

            # 4. Selection
            elites, parents_for_crossover = self._selection(fitness_scores)

            # 5. Crossover
            offspring = self._crossover(parents_for_crossover)

            # 6. Mutation
            mutated_offspring = self._mutation(offspring, current_mutation_eta)

            # 7. Create new population
            # Ensure that new population has exactly pop_size individuals
            num_offspring = len(mutated_offspring)
            num_elites = len(elites)
            if num_offspring + num_elites > self.pop_size:
                mutated_offspring = mutated_offspring[:self.pop_size - num_elites]

            self.population = np.vstack([elites, mutated_offspring])

        end_time = time.time()
        print(f"\nAlgorithm completed in {end_time - start_time:.2f} seconds.")
        print(f"Best found solution (function value): {self.best_fitness}")
        print(f"Optimal vector w: {self.best_individual}")

    def plot_history(self):
        """Displays convergence plot of the algorithm"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['best_fitness'], label='Best fitness per generation')
        plt.plot(self.history['avg_fitness'], label='Average fitness per generation', linestyle='--')
        plt.title('Genetic Algorithm Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Objective function value (minimization)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_diversity(self):
        """Displays the population diversity plot."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['diversity'], label='Population Diversity (StdDev)')
        plt.title('Population Diversity over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Standard Deviation of Objective Values')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # Problem definition
    DIMENSION = 60
    # Assumed bounds for neural network weights
    BOUNDS = [(-10, 10)] * DIMENSION

    # GA hyperparameters setup
    # These parameters are result of GA tuning and testing process
    ga_params = {
        'fitness_func': optimality_criterion,
        'dim': DIMENSION,
        'bounds': BOUNDS,
        'pop_size': 250,
        'generations': 300,
        'crossover_rate': 0.9,
        'mutation_rate': 0.04,
        'selection_k': 3,
        'crossover_eta': 5,
        'mutation_eta_schedule': (5, 50),
        'elitism_count': 3,
        'stagnation_limit': 15
    }

    # Creating and starting the algorithm
    ga = GeneticAlgorithm(**ga_params)
    ga.run()

    # Show results
    ga.plot_history()
    ga.plot_diversity()
