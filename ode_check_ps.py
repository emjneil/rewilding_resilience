
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import timeit
import seaborn as sns
import numpy.matlib
from scipy import stats
import csv
from scipy.optimize import root
import random

species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']

# we want to check if our original PS passes the requirements or not
r = [0, 0, 0.91, 0, 0, 0, 0, 0.35, 0.11]


A = [
    # exmoor pony - special case, no growth
    [0, 0, 2.7, 0, 0, 0, 0, 0.17, 0.4],
    # fallow deer 
    [0, 0, 3.81, 0, 0, 0, 0, 0.37, 0.49],
    # grassland parkland
    [-0.0024, -0.0032, -0.83, -0.015, -0.003, -0.00092, -0.0082, -0.046, -0.049],
    # longhorn cattle  
    [0, 0, 4.99, 0, 0, 0, 0, 0.21, 0.4],
    # red deer  
    [0, 0, 2.8, 0, 0, 0, 0, 0.29, 0.42],
    # roe deer 
    [0, 0, 4.9, 0, 0, 0, 0, 0.25, 0.38],
    # tamworth pig 
    [0, 0, 3.7, 0, 0,0, 0, 0.23, 0.41],  
    # thorny scrub
    [-0.0014, -0.005, 0, -0.0051, -0.0039, -0.0023, -0.0018, -0.015, -0.022],
    # woodland
    [-0.0041, -0.0058, 0, -0.0083, -0.004, -0.0032, -0.0037, 0.0079, -0.0062]
    ]



# generate starting conditions
def generate_rescaled_list(j, i):
    total_sum = sum(j)
    random_numbers = [random.random() for _ in j]
    scaled_numbers = [num / sum(random_numbers) for num in random_numbers]
    rescaled_numbers = [scaled_num * total_sum for scaled_num in scaled_numbers]
    rescaled_i = [num / sum(rescaled_numbers) for num in rescaled_numbers]
    return rescaled_i

# actual starting values for vegetation in 2005
j = [89.9, 4.3, 5.8]
i = [1, 1, 1]




def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
    # consumers with PS2 have negative growth rate 
    r[0] = np.log(1/(100*X[0])) if X[0] != 0 else 0
    r[1] = np.log(1/(100*X[1])) if X[1] != 0 else 0
    r[3] = np.log(1/(100*X[3])) if X[3] != 0 else 0
    r[4] = np.log(1/(100*X[4])) if X[4] != 0 else 0
    r[5] = np.log(1/(100*X[5])) if X[5] != 0 else 0
    r[6] = np.log(1/(100*X[6])) if X[6] != 0 else 0

    return X * (r + np.matmul(A, X))


# here is the lotka-volterra equation
def lotka_volterra_equation(X, A, r):
    # put things to zero if lower than a threshold
    X[X<1e-8] = 0
    # consumers have negative growth rate 
    r[0] = np.log(1/(100*X[0])) if X[0] != 0 else 0
    r[1] = np.log(1/(100*X[1])) if X[1] != 0 else 0
    r[3] = np.log(1/(100*X[3])) if X[3] != 0 else 0
    r[4] = np.log(1/(100*X[4])) if X[4] != 0 else 0
    r[5] = np.log(1/(100*X[5])) if X[5] != 0 else 0
    r[6] = np.log(1/(100*X[6])) if X[6] != 0 else 0
    # return values
    return X * (r + np.matmul(A, X))



# get the root of the equation (steady state) - keeping track of timesteps
def solve_lotka_volterra(A, r, X0, time_points, forced_species_pony, forced_species_fallow, forced_species_cow, forced_species_red, forced_species_pig, stocking_changes, tolerance=1e-6, max_iterations=500):
    
    def update_population(t, X):
        if t in time_points:
            X[forced_species_pony] = 0.65 * stocking_changes
            X[forced_species_fallow] = 5.9 * stocking_changes
            X[forced_species_cow] = 1.5 * stocking_changes
            X[forced_species_red] = 2.7 * stocking_changes
            X[forced_species_pig] = 0.55 * stocking_changes
        return X


    def equation_with_forcing(X, t):
        X = update_population(t, X)
        return lotka_volterra_equation(X, A, r)

    iteration_count = 0
    prev_X = np.array(X0)
    population_history = [prev_X]


    while iteration_count < max_iterations:
        t = iteration_count + 1
        result = root(equation_with_forcing, prev_X, args=(t,))
        X = result.x

        if np.linalg.norm(X - prev_X) < tolerance:
            break

        prev_X = X

        # but we also want the actual population (not the root)
        population_history.append(X)
        iteration_count += 1
    
    return X, iteration_count, population_history



# set the parameters
time_points = list(range(1, 500))  # Time points to force X[0] to be 1 at the beginning of each year
forced_species_pony = 0             # Index of the forced species (0-based)
forced_species_fallow = 1
forced_species_cow = 3
forced_species_red = 4
forced_species_pig = 6
stocking_changes = random.choices([1, 0, 0.5, 2, 5], k=1000)
run_number = 0

final_df = {}
all_timeseries = []


# calculate the steady state (where X0 = 2020 conditions)
for i in stocking_changes:

    # generate starting conditions
    random_list = generate_rescaled_list(j, i)
    # make the starting conditions based on this
    X0_init = [0, 0, random_list[0], 0, 0, 1, 0, random_list[1], random_list[2]]

    # run it from 2005-2020 
    all_times = []
    t_init = np.linspace(0, 3.75, 12)
    results = solve_ivp(ecoNetwork, (0, 3.75), X0_init,  t_eval = t_init, args=(A, r), method = 'RK23')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 12).transpose(),1)))
    y = pd.DataFrame(data=y, columns=species)
    all_times = np.append(all_times, results.t)
    y['time'] = all_times
    last_results = y.loc[y['time'] == 3.75]
    last_results = last_results.drop('time', axis=1)
    last_results = last_results.values.flatten()
    # ponies, longhorn cattle, and tamworth pigs reintroduced in 2009
    starting_2009 = last_results.copy()
    starting_2009[0] = 1
    starting_2009[3] = 1
    starting_2009[6] = 1
    t_01 = np.linspace(4, 4.75, 3)
    second_ABC = solve_ivp(ecoNetwork, (4,4.75), starting_2009,  t_eval = t_01, args=(A, r), method = 'RK23')
    # 2010
    last_values_2009 = second_ABC.y[0:11, 2:3].flatten()
    starting_values_2010 = last_values_2009.copy()
    starting_values_2010[0] = 0.57
    # fallow deer reintroduced
    starting_values_2010[1] = 1
    starting_values_2010[3] = 1.5
    starting_values_2010[6] = 0.85
    t_1 = np.linspace(5, 5.75, 3)
    third_ABC = solve_ivp(ecoNetwork, (5,5.75), starting_values_2010,  t_eval = t_1, args=(A, r), method = 'RK23')
    # 2011
    last_values_2010 = third_ABC.y[0:11, 2:3].flatten()
    starting_values_2011 = last_values_2010.copy()
    starting_values_2011[0] = 0.65
    starting_values_2011[1] = 1.9
    starting_values_2011[3] = 1.7
    starting_values_2011[6] = 1.1
    t_2 = np.linspace(6, 6.75, 3)
    fourth_ABC = solve_ivp(ecoNetwork, (6,6.75), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
    # 2012
    last_values_2011 = fourth_ABC.y[0:11, 2:3].flatten()
    starting_values_2012 = last_values_2011.copy()
    starting_values_2012[0] = 0.74
    starting_values_2012[1] = 2.4
    starting_values_2012[3] = 2.2
    starting_values_2012[6] = 1.7
    t_3 = np.linspace(7, 7.75, 3)
    fifth_ABC = solve_ivp(ecoNetwork, (7,7.75), starting_values_2012,  t_eval = t_3, args=(A, r), method = 'RK23')
    # 2013
    last_values_2012 = fifth_ABC.y[0:11, 2:3].flatten()
    starting_values_2013 = last_values_2012.copy()
    starting_values_2013[0] = 0.43
    starting_values_2013[1] = 2.4
    starting_values_2013[3] = 2.4
    # red deer reintroduced
    starting_values_2013[4] = 1
    starting_values_2013[6] = 0.3
    t_4 = np.linspace(8, 8.75, 3)
    sixth_ABC = solve_ivp(ecoNetwork, (8,8.75), starting_values_2013,  t_eval = t_4, args=(A, r), method = 'RK23')
    # 2014
    last_values_2013 = sixth_ABC.y[0:11, 2:3].flatten()
    starting_values_2014 = last_values_2013.copy()
    starting_values_2014[0] = 0.44
    starting_values_2014[1] = 2.4
    starting_values_2014[3] = 5
    starting_values_2014[4] = 1
    starting_values_2014[6] = 0.9
    t_5 = np.linspace(9, 9.75, 3)
    seventh_ABC = solve_ivp(ecoNetwork, (9,9.75), starting_values_2014,  t_eval = t_5, args=(A, r), method = 'RK23') 
    # 2015
    last_values_2014 = seventh_ABC.y[0:11, 2:3].flatten()
    starting_values_2015 = last_values_2014
    starting_values_2015[0] = 0.43
    starting_values_2015[1] = 2.4
    starting_values_2015[3] = 2
    starting_values_2015[4] = 1
    starting_values_2015[6] = 0.71
    t_2015 = np.linspace(10, 10.75, 3)
    ABC_2015 = solve_ivp(ecoNetwork, (10,10.75), starting_values_2015,  t_eval = t_2015, args=(A, r), method = 'RK23')
    # 2016
    last_values_2015 = ABC_2015.y[0:11, 2:3].flatten()
    starting_values_2016 = last_values_2015.copy()
    starting_values_2016[0] = 0.48
    starting_values_2016[1] = 3.3
    starting_values_2016[3] = 1.7
    starting_values_2016[4] = 2
    starting_values_2016[6] = 0.7
    t_2016 = np.linspace(11, 11.75, 3)
    ABC_2016 = solve_ivp(ecoNetwork, (11,11.75), starting_values_2016,  t_eval = t_2016, args=(A, r), method = 'RK23')       
    # 2017
    last_values_2016 = ABC_2016.y[0:11, 2:3].flatten()
    starting_values_2017 = last_values_2016.copy()
    starting_values_2017[0] = 0.43
    starting_values_2017[1] = 3.9
    starting_values_2017[3] = 1.7
    starting_values_2017[4] = 1.1
    starting_values_2017[6] = 0.95
    t_2017 = np.linspace(12, 12.75, 3)
    ABC_2017 = solve_ivp(ecoNetwork, (12,12.75), starting_values_2017,  t_eval = t_2017, args=(A, r), method = 'RK23')     
    # 2018
    last_values_2017 = ABC_2017.y[0:11, 2:3].flatten()
    starting_values_2018 = last_values_2017.copy()
    # pretend bison were reintroduced (to estimate growth rate / interaction values)
    starting_values_2018[0] = 0
    starting_values_2018[1] = 6.0
    starting_values_2018[3] = 1.9
    starting_values_2018[4] = 1.9
    starting_values_2018[6] = 0.84
    t_2018 = np.linspace(13, 13.75, 3)
    ABC_2018 = solve_ivp(ecoNetwork, (13,13.75), starting_values_2018,  t_eval = t_2018, args=(A, r), method = 'RK23')     
    # 2019
    last_values_2018 = ABC_2018.y[0:11, 2:3].flatten()
    starting_values_2019 = last_values_2018.copy()
    starting_values_2019[0] = 0
    starting_values_2019[1] = 6.6
    starting_values_2019[3] = 1.7
    starting_values_2019[4] = 2.9
    starting_values_2019[6] = 0.44
    t_2019 = np.linspace(14, 14.75, 3)
    ABC_2019 = solve_ivp(ecoNetwork, (14,14.75), starting_values_2019,  t_eval = t_2019, args=(A, r), method = 'RK23')
    # 2020
    last_values_2019 = ABC_2019.y[0:11, 2:3].flatten()
    starting_values_2020 = last_values_2019.copy()
    starting_values_2020[0] = 0.65
    starting_values_2020[1] = 5.9
    starting_values_2020[3] = 1.5
    starting_values_2020[4] = 2.7
    starting_values_2020[6] = 0.55
    t_2020 = np.linspace(15, 16, 3)
    ABC_2020 = solve_ivp(ecoNetwork, (15,16), starting_values_2020,  t_eval = t_2020, args=(A, r), method = 'RK23')     
    # get those last values
    last_values_2020 = ABC_2020.y[0:11, 2:3].flatten()

    # gather final conditions and run to equilbrium
    X0 = [0.65, 5.9, last_values_2020[2], 1.5, 2.7, last_values_2020[5], 0.55, last_values_2020[7], last_values_2020[8]]

    run_number += 1
    steady_state, time_step, population_history = solve_lotka_volterra(A, r, X0, time_points, forced_species_pony, forced_species_fallow, forced_species_cow, forced_species_red, forced_species_pig, i)
    
    print("Steady state:", steady_state)
    print("Reached steady state at time step:", time_step)


    # keep track of time-series
    df_population = pd.DataFrame(population_history, columns=species)
    df_population.insert(0, 'time', range(17, 17 + len(df_population))) # start from 16, where the previous one ends
    df_population["run_number"] = run_number

    # get 2005-2020 time-series
    combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
    combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
    # reshape the outputs
    y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
    y_2 = pd.DataFrame(data=y_2, columns=species)
    y_2['time'] = combined_times
    y_2["run_number"] = run_number
    # concat all time-series
    timeseries = pd.concat([y_2, df_population])


    # only save those that reach equilibrium within 500 years
    if time_step < 500:

        # gather that stuff
        final_df[run_number] = {"Grassland Initial": X0[2]*89.9, "Grassland End": steady_state[2]*89.9, 
                                "Woodland Initial": X0[8]*5.8, "Woodland End":  steady_state[8]*5.8, 
                                "Thorny Scrub Initial": X0[7]*4.3, "Thorny Scrub End":  steady_state[7]*4.3, 
                                "Stocking Density": i,
                                "Run Number": run_number
        }

        all_timeseries.append(timeseries)


forecasting = pd.DataFrame.from_dict(final_df, "index")
all_timeseries = pd.concat(all_timeseries)

# save to csv
forecasting.to_csv("resilience_landscape_eem.csv")
all_timeseries.to_csv("timeseries_eem.csv")