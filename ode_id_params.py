# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

# download packages
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

# define the number of simulations to try
totalSimulations = 100_000

# store species in a list
species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']

# define the Lotka-Volterra equation
def ecoNetwork(t, X, A, r):
    X[X<1e-8] = 0
    # consumers with PS2 have negative growth rate 
    r[0] = np.log(1/(100*X[0])) if X[0] != 0 else 0
    r[1] = np.log(1/(100*X[1])) if X[1] != 0 else 0
    r[3] = np.log(1/(100*X[3])) if X[3] != 0 else 0
    r[4] = np.log(1/(100*X[4])) if X[4] != 0 else 0
    r[5] = np.log(1/(100*X[5])) if X[5] != 0 else 0
    r[6] = np.log(1/(100*X[6])) if X[6] != 0 else 0
    return X * (r + np.matmul(A, X))



# # # -------- GENERATE PARAMETERS ------ 

def generateInteractionMatrix():

    interaction_matrix = [
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

    # generate random uniform numbersF
    variation = np.random.uniform(low = 0.9, high=1.1, size = (len(species),len((species))))
    interaction_matrix = interaction_matrix * variation
    # return array
    return interaction_matrix



def generateGrowth():
    growthRates = [0, 0, 0.91, 0, 0, 0, 0, 0.35, 0.11]
    # multiply by a range
    variation = np.random.uniform(low = 0.9, high=1.1, size = (len(species),))
    growth = growthRates * variation
    return growth
    


def generateX0():
    # scale everything to abundance of one (except species to be reintroduced)
    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]
    return X0








# # # # --------- Does it pass ecological criteria? -------


def passed_ecological_criteria():  # want it to have < 500 roe deer and reach equilibrium in < 500 yrs
    run_number = 0

    final_df = {}
    all_timeseries = []
    all_parameters = []

    # generate the parameters
    for _ in range(totalSimulations):
        run_number += 1
        print(run_number)
        A = generateInteractionMatrix()
        r = generateGrowth()
        X0 = generateX0()

        # remember the parameters used
        X0_growth = pd.concat([pd.DataFrame(X0), pd.DataFrame(r)], axis = 1)
        X0_growth.columns = ['X0','growth']
        parameters_used = pd.concat([X0_growth, pd.DataFrame(A, index = species, columns = species)])

        # run it from 2005-2020 
        all_times = []
        t_init = np.linspace(0, 3.75, 12)
        results = solve_ivp(ecoNetwork, (0, 3.75), X0,  t_eval = t_init, args=(A, r), method = 'RK23')
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
        t_2020 = np.linspace(15, 15.75, 3)
        ABC_2020 = solve_ivp(ecoNetwork, (15,15.75), starting_values_2020,  t_eval = t_2020, args=(A, r), method = 'RK23')     
        # get those last values
        last_values_2020 = ABC_2020.y[0:11, 2:3].flatten()

        # gather final conditions and run to equilbrium
        X0 = [0.65, 5.9, last_values_2020[2], 1.5, 2.7, last_values_2020[5], 0.55, last_values_2020[7], last_values_2020[8]]

        # now run it for 500 years - does it reach this steady state?
        years_covered = 16
        combined_future_runs = []
        combined_future_times = []

        last_values = [ABC_2020.y[0:11, 2:3].flatten()]
        
        for y in range(500): 
            years_covered += 1
            # run the model for that year 
            my_last_values = last_values[-1]
            starting_values = my_last_values.copy()
            starting_values[0] = 0.65
            starting_values[1] = 5.9
            starting_values[3] = 1.5
            starting_values[4] = 2.7
            starting_values[6] = 0.55
            # keep track of time
            t_steady = np.linspace(years_covered, years_covered+0.75, 3)
            # run model
            ABC_future = solve_ivp(ecoNetwork, (years_covered, years_covered+0.75), starting_values,  t_eval = t_steady, args=(A, r), method = 'RK23')   
            # append last values
            next_last_values = last_values.append(ABC_future.y[0:11, 2:3].flatten())

            # and save the outputs
            combined_future_runs.append(ABC_future.y)
            combined_future_times.append(ABC_future.t)

        combined_future = np.hstack(combined_future_runs)
        combined_future_times = np.hstack(combined_future_times)
        y_future = (np.vstack(np.hsplit(combined_future.reshape(len(species), 1500).transpose(),1)))
        y_future = pd.DataFrame(data=y_future, columns=species)
        y_future['time'] = combined_future_times
        y_future["run_number"] = run_number


        # get 2005-2020 time-series
        combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # reshape the outputs
        y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
        y_2 = pd.DataFrame(data=y_2, columns=species)
        y_2['time'] = combined_times
        y_2["run_number"] = run_number
        # concat all time-series
        timeseries = pd.concat([y_2, y_future])

        end = timeseries.iloc[-1]

        # if roe deer is < 500
        if end["roeDeer"] < 41.6: 
            # if equilibrium is reached by 500yrs 
            last_hundred = timeseries.loc[timeseries["time"] >= 400]
            # only species I care about - ones that aren't controlled
            uncontrolled_only = last_hundred[['grasslandParkland', 'woodland', 'thornyScrub', 'roeDeer']]

            gradients = {}

            for column in uncontrolled_only.columns:
                
                # Get the population values for the current species
                population_values = uncontrolled_only[column].values
                # Create an array representing the years (assuming the index represents the years)
                years = uncontrolled_only.index.values
                # Fit a first-degree polynomial (line) to the population values
                coefficients = np.polyfit(years, population_values, deg=1)
                # Extract the gradient (slope) from the coefficients
                gradient = coefficients[0]
                # Store the gradient in the dictionary
                gradients[column] = gradient

            # is it at equilibrium?
            if abs(gradients["roeDeer"]) < 0.01 and abs(gradients["thornyScrub"]) < 0.01 and abs(gradients["grasslandParkland"]) < 0.01 and abs(gradients["woodland"]) < 0.01:
                # remember only those params that pass
                all_parameters.append(parameters_used)
                all_timeseries.append(timeseries)


    parameters = pd.concat(all_parameters)
    all_timeseries = pd.concat(all_timeseries)

    num_passed = int(all_timeseries.shape[0]/1548)

    print("passed forecasting", all_timeseries.shape[0]/1548) # divide the shape by the total number of time-steps (516 years x 3 steps)

    with open("eem_passed_ecological.txt", "w") as text_file:
        print("number of simulations: {}".format(all_timeseries.shape[0]/1548), file=text_file)

    return parameters, num_passed





# # # # --------- SOLVE ODE #1: Pre-reintroductions (2000-2009) -------



def runODE_1():
    parameters, num_passed = passed_ecological_criteria()

    t = np.linspace(2005, 2008.75, 8)
    t_eco = np.linspace(0, 1, 2)
    all_runs = []
    all_times = []
    all_parameters = []
    NUMBER_OF_SIMULATIONS = 0
    run_number = 0

    # grab the accepted X0 and growths
    growthRates_2 = parameters.loc[parameters['growth'].notnull(), ['growth']]
    growthRates_2 = pd.DataFrame(growthRates_2.values.reshape((num_passed), len(species)), columns = species)
    r_all = growthRates_2.to_numpy()

    # and interaction matrix
    interaction_strength_2 = parameters.drop(['X0', 'growth'], axis=1)
    interaction_strength_2 = interaction_strength_2.dropna()
    A_all = interaction_strength_2.to_numpy()

    for r, A in zip(r_all, np.array_split(A_all,num_passed)):
        run_number += 1
        print(run_number)

        # generate X0 (it's the same every time)
        X0 = generateX0()

        # remember parameters 
        X0_growth_2 = pd.concat([pd.DataFrame(X0), pd.DataFrame(r)], axis = 1)
        X0_growth_2.columns = ['X0','growth']
        parameters_used = pd.concat([X0_growth_2, pd.DataFrame(A, index = species, columns = species)])

        # run the ODE
        first_ABC = solve_ivp(ecoNetwork, (2005, 2008.75), X0,  t_eval = t, args=(A, r), method = 'RK23')
        # append all the runs
        all_runs = np.append(all_runs, first_ABC.y)
        all_times = np.append(all_times, first_ABC.t)
        # add one to the counter (so we know how many simulations were run in the end)
        NUMBER_OF_SIMULATIONS += 1

        # concat parameters
        all_parameters.append(parameters_used)   

    # combine the final runs into one dataframe
    final_runs = (np.vstack(np.hsplit(all_runs.reshape(len(species)*NUMBER_OF_SIMULATIONS, 8).transpose(),NUMBER_OF_SIMULATIONS)))
    final_runs = pd.DataFrame(data=final_runs, columns=species)
    final_runs['time'] = all_times
    # put all the parameters tried into a dataframe, and give them IDs
    all_parameters = pd.concat(all_parameters)
    all_parameters['ID'] = ([(x+1) for x in range(NUMBER_OF_SIMULATIONS) for _ in range(len(parameters_used))])
    return final_runs, all_parameters, NUMBER_OF_SIMULATIONS




# --------- FILTER OUT UNREALISTIC RUNS -----------

def filterRuns_1():
    final_runs, all_parameters, NUMBER_OF_SIMULATIONS = runODE_1()
    # also ID the runs 
    IDs = np.arange(1,NUMBER_OF_SIMULATIONS+1)
    final_runs['ID'] = np.repeat(IDs,8)
    # select only the last year
    accepted_year = final_runs.loc[final_runs['time'] == 2008.75]

    # filter the runs through conditions
    accepted_simulations = accepted_year[
    (accepted_year['roeDeer'] <= 3.3) & (accepted_year['roeDeer'] >= 1) &
    (accepted_year['grasslandParkland'] <= 1) & (accepted_year['grasslandParkland'] >= 0.18) &
    (accepted_year['woodland'] <=2.9) & (accepted_year['woodland'] >= 1) &
    (accepted_year['thornyScrub'] <= 4.9) & (accepted_year['thornyScrub'] >= 1)
    ]
    print("number accepted, first ODE:", accepted_simulations.shape)

    with open("eem_passed_2009_filters.txt", "w") as text_file:
        print("number of simulations: {}".format(accepted_simulations.shape), file=text_file)

    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters = all_parameters[all_parameters['ID'].isin(accepted_simulations['ID'])]
    # show which runs were accepted and which were rejected
    final_runs['accepted?'] = np.where(final_runs['ID'].isin(accepted_parameters['ID']), 'Accepted', 'Rejected')
    return accepted_parameters, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS




# # # # ---------------------- ODE #2: Years 2009-2018 -------------------------


def generateParameters2():
    accepted_parameters, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS = filterRuns_1()
    # take the accepted growth rates
    growthRates_2 = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    growthRates_2 = pd.DataFrame(growthRates_2.values.reshape(len(accepted_simulations), len(species)), columns = species)
    r_secondRun = growthRates_2.to_numpy()
    # make the initial conditions for this ODE = the final conditions of the first ODE
    shortenedAcceptedsimulations = accepted_simulations.drop(['ID', 'time'], axis=1).copy()
    accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']] = shortenedAcceptedsimulations.values.flatten()
    X0_2 = accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']]
    X0_2 = pd.DataFrame(X0_2.values.reshape(len(accepted_simulations), len(species)), columns = species)
    # add reintroduced species; ponies, longhorn cattle, and tamworth were reintroduced in 2009
    X0_2.loc[:, 'exmoorPony'] = [1 for i in X0_2.index]
    X0_2.loc[:, 'longhornCattle'] = [1 for i in X0_2.index]
    X0_2.loc[:,'tamworthPig'] = [1 for i in X0_2.index]
    X0_secondRun = X0_2.to_numpy()
    # select accepted interaction strengths
    interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_2 = interaction_strength_2.dropna()
    A_secondRun = interaction_strength_2.to_numpy()
    return r_secondRun, X0_secondRun, A_secondRun, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS



# # # # # ------ SOLVE ODE #2: Post-reintroductions (2009-2018) -------

def runODE_2():
    r_secondRun, X0_secondRun, A_secondRun, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS  = generateParameters2()
    all_runs_2 = []
    all_times_2 = []
    all_parameters_2 = []
    run_number = 0
    # loop through each row of accepted parameters
    for X0_3, r_2, A_2 in zip(X0_secondRun,r_secondRun, np.array_split(A_secondRun,len(accepted_simulations))):
        run_number += 1
        print(run_number)
        # concantenate the parameters
        X0_growth_2 = pd.concat([pd.DataFrame(X0_3), pd.DataFrame(r_2)], axis = 1)
        X0_growth_2.columns = ['X0','growth']
        parameters_used_2 = pd.concat([X0_growth_2, pd.DataFrame(A_2, index = species, columns = species)])
        # 2009
        t = np.linspace(2009, 2009.75, 2)
        second_ABC = solve_ivp(ecoNetwork, (2009, 2009.75), X0_3,  t_eval = t, args=(A_2, r_2), method = 'RK23')
        # 2010: fallow deer reintroduced
        starting_values_2010 = second_ABC.y[0:9, 1:2].flatten()
        starting_values_2010[0] = 0.57
        starting_values_2010[1] = 1
        starting_values_2010[3] = 1.5
        starting_values_2010[6] = 0.85
        t_1 = np.linspace(2010, 2010.75, 2)
        third_ABC = solve_ivp(ecoNetwork, (2010, 2010.75), starting_values_2010,  t_eval = t_1, args=(A_2, r_2), method = 'RK23')
        # 2011
        starting_values_2011 = third_ABC.y[0:9, 1:2].flatten()
        starting_values_2011[0] = 0.65
        starting_values_2011[1] = 1.9
        starting_values_2011[3] = 1.7
        starting_values_2011[6] = 1.1
        t_2 = np.linspace(2011, 2011.75, 2)
        fourth_ABC = solve_ivp(ecoNetwork, (2011, 2011.75), starting_values_2011,  t_eval = t_2, args=(A_2, r_2), method = 'RK23')
        # 2012
        starting_2012 = fourth_ABC.y[0:9, 1:2].flatten()
        starting_values_2012 = starting_2012.copy()
        starting_values_2012[0] = 0.74
        starting_values_2012[1] = 2.4
        starting_values_2012[3] = 2.2
        starting_values_2012[6] = 1.7
        t_3 = np.linspace(2012, 2012.75, 2)
        fifth_ABC = solve_ivp(ecoNetwork, (2012, 2012.75), starting_values_2012,  t_eval = t_3, args=(A_2, r_2), method = 'RK23')
        # 2013: red deer reintroduced
        starting_2013 = fifth_ABC.y[0:11, 1:2].flatten()
        starting_values_2013 = starting_2013.copy()
        starting_values_2013[0] = 0.43
        starting_values_2013[1] = 2.4
        starting_values_2013[3] = 2.4
        starting_values_2013[4] = 1
        starting_values_2013[6] = 0.3
        t_4 = np.linspace(2013, 2013.75, 2)
        sixth_ABC = solve_ivp(ecoNetwork, (2013, 2013.75), starting_values_2013,  t_eval = t_4, args=(A_2, r_2), method = 'RK23')
        # 2014
        starting_2014 = sixth_ABC.y[0:11, 1:2].flatten()
        starting_values_2014 = starting_2014.copy()
        starting_values_2014[0] = 0.44
        starting_values_2014[1] = 2.4
        starting_values_2014[3] = 5
        starting_values_2014[4] = 1
        starting_values_2014[6] = 0.9
        t_5 = np.linspace(2014, 2014.75, 2)
        seventh_ABC = solve_ivp(ecoNetwork, (2014, 2014.75), starting_values_2014,  t_eval = t_5, args=(A_2, r_2), method = 'RK23')
        # 2015
        starting_values_2015 = seventh_ABC.y[0:11, 1:2].flatten()
        starting_values_2015[0] = 0.43
        starting_values_2015[1] = 2.4
        starting_values_2015[3] = 2
        starting_values_2015[4] = 1
        starting_values_2015[6] = 0.71
        t_2015 = np.linspace(2015, 2015.75, 2)
        ABC_2015 = solve_ivp(ecoNetwork, (2015, 2015.75), starting_values_2015,  t_eval = t_2015, args=(A_2, r_2), method = 'RK23')
        last_values_2015 = ABC_2015.y[0:11, 1:2].flatten()
        # 2016
        starting_values_2016 = last_values_2015.copy()
        starting_values_2016[0] = 0.48
        starting_values_2016[1] = 3.3
        starting_values_2016[3] = 1.7
        starting_values_2016[4] = 2
        starting_values_2016[6] = 0.69
        t_2016 = np.linspace(2016, 2016.75, 2)
        ABC_2016 = solve_ivp(ecoNetwork, (2016, 2016.75), starting_values_2016,  t_eval = t_2016, args=(A_2, r_2), method = 'RK23')
        last_values_2016 = ABC_2016.y[0:11, 1:2].flatten()
        # 2017
        starting_values_2017 = last_values_2016.copy()
        starting_values_2017[0] = 0.43
        starting_values_2017[1] = 3.9
        starting_values_2017[3] = 1.7
        starting_values_2017[4] = 1.1
        starting_values_2017[6] = 0.95
        t_2017 = np.linspace(2017, 2017.75, 2)
        ABC_2017 = solve_ivp(ecoNetwork, (2017, 2017.75), starting_values_2017,  t_eval = t_2017, args=(A_2, r_2), method = 'RK23')
        last_values_2017 = ABC_2017.y[0:11, 1:2].flatten()
        # 2018
        starting_values_2018 = last_values_2017.copy()
        starting_values_2018[0] = 0
        starting_values_2018[1] = 6
        starting_values_2018[3] = 1.9
        starting_values_2018[4] = 1.9
        starting_values_2018[6] = 0.84
        t_2018 = np.linspace(2018, 2018.75, 2)
        ABC_2018 = solve_ivp(ecoNetwork, (2018, 2018.75), starting_values_2018,  t_eval = t_2018, args=(A_2, r_2), method = 'RK23')
        last_values_2018 = ABC_2018.y[0:11, 1:2].flatten()
        # 2019
        starting_values_2019 = last_values_2018.copy()
        starting_values_2019[0] = 0
        starting_values_2019[1] = 6.6
        starting_values_2019[3] = 1.7
        starting_values_2019[4] = 2.9
        starting_values_2019[6] = 0.44
        t_2019 = np.linspace(2019, 2019.75, 2)
        ABC_2019 = solve_ivp(ecoNetwork, (2019, 2019.75), starting_values_2019,  t_eval = t_2019, args=(A_2, r_2), method = 'RK23')
        last_values_2019 = ABC_2019.y[0:11, 1:2].flatten()
        # 2020
        starting_values_2020 = last_values_2019.copy()
        starting_values_2020[0] = 0.65
        starting_values_2020[1] = 5.9
        starting_values_2020[3] = 1.5
        starting_values_2020[4] = 2.7
        starting_values_2020[6] = 0.55
        t_2020 = np.linspace(2020, 2020.75, 2)
        ABC_2020 = solve_ivp(ecoNetwork, (2020,2020.75), starting_values_2020,  t_eval = t_2020, args=(A_2, r_2), method = 'RK23')
        # concatenate & append all the runs
        combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # append to dataframe
        all_runs_2 = np.append(all_runs_2, combined_runs)
        # append all the parameters
        all_parameters_2.append(parameters_used_2)   
        all_times_2 = np.append(all_times_2, combined_times)
    # check the final runs
    final_results = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations), 24).transpose(),len(accepted_simulations))))
    final_results = pd.DataFrame(data=final_results, columns=species)
    final_results['time'] = all_times_2
    # append all the parameters to a dataframe
    all_parameters = pd.concat(all_parameters_2)
    # add ID to the dataframe & parameters
    all_parameters['ID'] = ([(x+1) for x in range(len(accepted_simulations)) for _ in range(len(parameters_used_2))])
    IDs = np.arange(1,1 + len(accepted_simulations))
    final_results['ID'] = np.repeat(IDs,24)

    return final_results, all_parameters, final_runs




# --------- FILTER OUT UNREALISTIC RUNS: Post-reintroductions -----------
def filterRuns():
    final_results, all_parameters, final_runs = runODE_2()

    # they already passed the 2005 filters
    final_results["passed_filters"] = 4

    # find the difficult filters 
    difficult_filters = pd.DataFrame({
    'filter_number': np.arange(0,28),
    'times_passed': np.zeros(28)})

    my_time = final_results.loc[final_results['time'] == 2020.75]
    accepted_runs_exmoor = []
    accepted_runs_roe = []
    accepted_runs_grass = []
    accepted_runs_wood = []
    accepted_runs_scrub = []
    # exmoor pony
    for index, row in my_time.iterrows():
        if (row['exmoorPony'] <= 0.7) & (row['exmoorPony'] >= 0.61):
            accepted_runs_exmoor.append(row["ID"])
            difficult_filters.loc[0,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_exmoor),final_results['passed_filters']+1,final_results['passed_filters']) 
    # roe deer
    for index, row in my_time.iterrows():
        if (row['roeDeer'] <= 6.7) & (row['roeDeer'] >= 1.7):
            accepted_runs_roe.append(row["ID"])
            difficult_filters.loc[1,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_roe),final_results['passed_filters']+1,final_results['passed_filters']) 
    # grass
    for index, row in my_time.iterrows():
        if (row['grasslandParkland'] <= 0.41) & (row['grasslandParkland'] >= 0.18):
            accepted_runs_grass.append(row["ID"])
            difficult_filters.loc[2,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_grass),final_results['passed_filters']+1,final_results['passed_filters']) 
    # woodland
    for index, row in my_time.iterrows():
        if (row['woodland'] <= 4.6) & (row['woodland'] >= 2.8):
            accepted_runs_wood.append(row["ID"])
            difficult_filters.loc[3,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_wood),final_results['passed_filters']+1,final_results['passed_filters']) 
    # scrub
    for index, row in my_time.iterrows():
        if (row['thornyScrub'] <= 14.4) & (row['thornyScrub'] >= 9.7):
            accepted_runs_scrub.append(row["ID"])
            difficult_filters.loc[4,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_scrub),final_results['passed_filters']+1,final_results['passed_filters']) 

    # now take top 100 remaining filters
    # filter 2015 values
    my_time = final_results.loc[final_results['time'] == 2015.75]
    accepted_runs_exmoor = []
    accepted_runs_fallow = []
    accepted_runs_cattle = []
    accepted_runs_red = []
    accepted_runs_pig = []
    # exmoor pony
    for index, row in my_time.iterrows():
        if (row['exmoorPony'] <= 0.49) & (row['exmoorPony'] >= 0.39):
            accepted_runs_exmoor.append(row["ID"])
            difficult_filters.loc[5,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_exmoor),final_results['passed_filters']+1,final_results['passed_filters']) 
    # fallow deer
    for index, row in my_time.iterrows():
        if (row['fallowDeer'] <= 3.9) & (row['fallowDeer'] >= 2.7):
            accepted_runs_fallow.append(row["ID"])
            difficult_filters.loc[6,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_fallow),final_results['passed_filters']+1,final_results['passed_filters']) 
    # cattle
    for index, row in my_time.iterrows():
        if (row['longhornCattle'] <= 2.9) & (row['longhornCattle'] >= 2):  
            accepted_runs_cattle.append(row["ID"])
            difficult_filters.loc[7,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_cattle),final_results['passed_filters']+1,final_results['passed_filters']) 
    # red deer
    for index, row in my_time.iterrows():
        if (row['redDeer'] <= 1.4) & (row['redDeer'] >= 0.6):
            accepted_runs_red.append(row["ID"])
            difficult_filters.loc[8,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_red),final_results['passed_filters']+1,final_results['passed_filters']) 
    # pigs
    for index, row in my_time.iterrows():
        if (row['tamworthPig'] <= 1.6) & (row['tamworthPig'] >= 0.6):
            accepted_runs_pig.append(row["ID"])
            difficult_filters.loc[9,'times_passed'] += 1

    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_pig),final_results['passed_filters']+1,final_results['passed_filters']) 

    # filter 2016 values : ponies = the same, fallow deer = 3.9 but 25 were culled; longhorn got to maximum 2.03; pig got to maximum 0.85
    my_time = final_results.loc[final_results['time'] == 2016.75]
    accepted_runs_exmoor = []
    accepted_runs_fallow = []
    accepted_runs_cattle = []
    accepted_runs_red = []
    accepted_runs_pig = []    
    # exmoor pony
    for index, row in my_time.iterrows():
        if (row['exmoorPony'] <= 0.5) & (row['exmoorPony'] >= 0.43):
            accepted_runs_exmoor.append(row["ID"])
            difficult_filters.loc[10,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_exmoor),final_results['passed_filters']+1,final_results['passed_filters']) 
    # fallow deer
    for index, row in my_time.iterrows():
        if (row['fallowDeer'] <= 4.5) & (row['fallowDeer'] >= 3.3):
            accepted_runs_fallow.append(row["ID"])
            difficult_filters.loc[11,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_fallow),final_results['passed_filters']+1,final_results['passed_filters']) 
    # cattle
    for index, row in my_time.iterrows():
        if (row['longhornCattle'] <= 2.5) & (row['longhornCattle'] >= 1.6):
            accepted_runs_cattle.append(row["ID"])
            difficult_filters.loc[12,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_cattle),final_results['passed_filters']+1,final_results['passed_filters']) 
    # red deer
    for index, row in my_time.iterrows():
        if (row['redDeer'] <= 2.4) & (row['redDeer'] >= 1.6): 
            accepted_runs_red.append(row["ID"])
            difficult_filters.loc[13,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_red),final_results['passed_filters']+1,final_results['passed_filters']) 
    # pigs
    for index, row in my_time.iterrows():
        if (row['tamworthPig'] <= 1.4) & (row['tamworthPig'] >= 0.35):
            accepted_runs_pig.append(row["ID"])
            difficult_filters.loc[14,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_pig),final_results['passed_filters']+1,final_results['passed_filters']) 

    # filter 2017 values : ponies = the same; fallow = 7.34 + 1.36 culled (this was maybe supplemented so no filter), same with red; cows got to max 2.06; red deer got to 1.85 + 2 culled; pig got to 1.1
    my_time = final_results.loc[final_results['time'] == 2017.75]
    accepted_runs_exmoor = []
    accepted_runs_fallow = []
    accepted_runs_cattle = []
    accepted_runs_red = []
    accepted_runs_pig = []    
    # exmoor pony
    for index, row in my_time.iterrows():
        if (row['exmoorPony'] <= 0.48) & (row['exmoorPony'] >= 0.39):
            accepted_runs_exmoor.append(row["ID"])
            difficult_filters.loc[15,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_exmoor),final_results['passed_filters']+1,final_results['passed_filters']) 
    # fallow deer
    for index, row in my_time.iterrows():
         if (row['fallowDeer'] <= 6.6) & (row['fallowDeer'] >= 5.4):
            accepted_runs_fallow.append(row["ID"])
            difficult_filters.loc[16,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_fallow),final_results['passed_filters']+1,final_results['passed_filters']) 
    # cattle
    for index, row in my_time.iterrows():
         if (row['longhornCattle'] <= 2.5) & (row['longhornCattle'] >= 1.6):  
            accepted_runs_cattle.append(row["ID"])
            difficult_filters.loc[17,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_cattle),final_results['passed_filters']+1,final_results['passed_filters']) 
    # red deer
    for index, row in my_time.iterrows():
         if (row['redDeer'] <= 1.5) & (row['redDeer'] >= 0.7):  
            accepted_runs_red.append(row["ID"])
            difficult_filters.loc[18,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_red),final_results['passed_filters']+1,final_results['passed_filters']) 
    # pigs
    for index, row in my_time.iterrows():
         if (row['tamworthPig'] <= 1.6) & (row['tamworthPig'] >= 0.6):
            accepted_runs_pig.append(row["ID"])
            difficult_filters.loc[19,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_pig),final_results['passed_filters']+1,final_results['passed_filters']) 

    # filter 2018 values : p ponies = same, fallow = 6.62 + 57 culled; cows got to max 2.21; reds got to 2.85 + 3 culled; pigs got to max 1.15
    my_time = final_results.loc[final_results['time'] == 2018.75]
    accepted_runs_fallow = []
    accepted_runs_cattle = []
    accepted_runs_red = []
    accepted_runs_pig = []    
    # fallow deer
    for index, row in my_time.iterrows():
        if (row['fallowDeer'] <= 8.1) & (row['fallowDeer'] >= 6.9):
            accepted_runs_fallow.append(row["ID"])
            difficult_filters.loc[20,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_fallow),final_results['passed_filters']+1,final_results['passed_filters']) 
    # cattle
    for index, row in my_time.iterrows():
        if (row['longhornCattle'] <= 2.7) & (row['longhornCattle'] >= 1.7):
            accepted_runs_cattle.append(row["ID"])
            difficult_filters.loc[21,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_cattle),final_results['passed_filters']+1,final_results['passed_filters']) 
    # red deer
    for index, row in my_time.iterrows():
        if (row['redDeer'] <= 2.2) & (row['redDeer'] >= 1.5): 
            accepted_runs_red.append(row["ID"])
            difficult_filters.loc[22,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_red),final_results['passed_filters']+1,final_results['passed_filters']) 
    # pigs
    for index, row in my_time.iterrows():
        if (row['tamworthPig'] <= 1.7) & (row['tamworthPig'] >= 0.7):
            accepted_runs_pig.append(row["ID"])
            difficult_filters.loc[23,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_pig),final_results['passed_filters']+1,final_results['passed_filters']) 
     
    
    # filter 2019 values : ponies = 0, fallow = 6.62 + 1.36 culled; longhorn maximum 2
    my_time = final_results.loc[final_results['time'] == 2019.75]
    accepted_runs_fallow = []
    accepted_runs_cattle = []
    accepted_runs_red = []
    accepted_runs_pig = []    
    # fallow deer
    for index, row in my_time.iterrows():
        if (row['fallowDeer'] <= 8.9) & (row['fallowDeer'] >= 7.7):
            accepted_runs_fallow.append(row["ID"])
            difficult_filters.loc[24,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_fallow),final_results['passed_filters']+1,final_results['passed_filters']) 
    # cattle
    for index, row in my_time.iterrows():
        if (row['longhornCattle'] <= 2.5) & (row['longhornCattle'] >= 1.6):
            accepted_runs_cattle.append(row["ID"])
            difficult_filters.loc[25,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_cattle),final_results['passed_filters']+1,final_results['passed_filters']) 
    # red deer
    for index, row in my_time.iterrows():
        if (row['redDeer'] <= 3.2) & (row['redDeer'] >= 2.5):
            accepted_runs_red.append(row["ID"])
            difficult_filters.loc[26,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_red),final_results['passed_filters']+1,final_results['passed_filters']) 
    # pigs
    for index, row in my_time.iterrows():
        if (row['tamworthPig'] <= 1) & (row['tamworthPig'] >= 0):
            accepted_runs_pig.append(row["ID"])
            difficult_filters.loc[27,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['ID'].isin(accepted_runs_pig),final_results['passed_filters']+1,final_results['passed_filters']) 


    final_results.to_csv("eem_all_runs.csv")
    all_parameters.to_csv("eem_all_parameters.csv")


filterRuns()





# # # # # FIND THE TOP 100 RUNS # # # # # # 


def top_runs():

    # open results
    final_results = pd.read_csv('eem_all_runs.csv')
    all_parameters = pd.read_csv('eem_all_parameters.csv')


    # separate out the first and last runs (the 2005-2009 runs are already tagged accepted/rejected)
    first_ode = final_results.loc[final_results['time'] <= 2008.75]
    second_ode = final_results.loc[final_results['time'] > 2008.75]

    # make sure they pass habitat conditions
    final_results_filtered = second_ode[(second_ode['time'] == 2020.75) & (second_ode['roeDeer'] <= 6.7) & (second_ode['roeDeer'] >= 1.7) & 
    (second_ode['grasslandParkland'] <= 0.41) & (second_ode['grasslandParkland'] >= 0.18) &
    (second_ode['woodland'] <= 3.7) & (second_ode['woodland'] >= 3.3) &
    (second_ode['thornyScrub'] <= 14.4) & (second_ode['thornyScrub'] >= 9.7)]

    print("number accepted, second ODE",len(final_results_filtered) )

    with open("eem_passed_2020.txt", "w") as text_file:
        print("number of simulations: {}".format(len(final_results_filtered)), file=text_file)


    # take the top 100 of those that passed the habitats, and tag these as accepted - this one is already looking at just last year
    best_results = final_results_filtered.nlargest(100, 'passed_filters')

    # match ID number in accepted_simulations to its parameters in all_parameters
    second_ode['accepted?'] = np.where(second_ode['ID'].isin(best_results['ID']), 'Accepted', 'Rejected')
    # match ID number in accepted_simulations to its parameters in all_parameters
    all_parameters['accepted?'] = np.where(all_parameters['ID'].isin(best_results['ID']), 'Accepted', 'Rejected')    
    print("number accepted", len(best_results))


    # concat the first and second odes
    final_results = pd.concat([first_ode, second_ode])


    # save only the accepted ones 
    accepted_params = all_parameters.loc[all_parameters['accepted?'] == "Accepted"]
    accepted_runs = final_results.loc[final_results['accepted?'] == "Accepted"]
    
    accepted_runs.to_csv("eem_accepted_runs.csv")
    accepted_params.to_csv("eem_accepted_parameters.csv")



top_runs()





# # # # # ---------------------- ODE #3: projecting 10 years (2018-2028) -------------------------

# def histograms_corrMatrix():

#     # open the csv
#     all_parameters = pd.read_csv('all_parameters_ps2_unstable.csv').iloc[:,1:]
#     number_passed = 2045 #ps1 - stable

#     accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == 'Accepted')]
#     # look at growth rates
#     parameters = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
#     parameters = pd.DataFrame(parameters.values.reshape(number_passed, len(species)), columns = species)
#     growth_rates = parameters.loc[:,["grasslandParkland", "thornyScrub", "woodland"]]

#     # look at interaction matrices
#     interaction_strength_3 = accepted_parameters.drop(['X0', 'growth', 'ID', 'accepted?'], axis=1).dropna()
#     interaction_strength_3 = interaction_strength_3.set_index(list(interaction_strength_3)[0])
#     # reshape int matrix
#     exmoorInts = interaction_strength_3[interaction_strength_3.index=='exmoorPony']
#     exmoorInts.columns = ['pony_pony', 'pony_fallow','pony_grass','pony_cattle','pony_red','pony_roe','pony_pig','pony_scrub','pony_wood']
#     exmoorInts = exmoorInts.reset_index(drop=True)
#     fallowInts = interaction_strength_3[interaction_strength_3.index=='fallowDeer']
#     fallowInts.columns = ['fallow_pony', 'fallow_fallow', 'fallow_grass','fallow_cattle','fallow_red', 'fallow_roe', 'fallow_pig', 'fallow_scrub', 'fallow_wood']
#     fallowInts = fallowInts.reset_index(drop=True)
#     arableInts = interaction_strength_3[interaction_strength_3.index=='grasslandParkland']
#     arableInts.columns = ['grass_pony', 'grass_fallow', 'grass_grass','grass_cattle','grass_red', 'grass_roe', 'grass_pig', 'grass_scrub', 'grass_wood']
#     arableInts = arableInts.reset_index(drop=True)
#     longhornInts = interaction_strength_3[interaction_strength_3.index=='longhornCattle']
#     longhornInts.columns = ['cattle_pony', 'cattle_fallow', 'cattle_grass','cattle_cattle','cattle_red', 'cattle_roe', 'cattle_pig', 'cattle_scrub', 'cattle_wood']
#     longhornInts = longhornInts.reset_index(drop=True)
#     redDeerInts = interaction_strength_3[interaction_strength_3.index=='redDeer']
#     redDeerInts.columns = ['red_pony', 'red_fallow', 'red_grass','red_cattle','red_red', 'red_roe', 'red_pig', 'red_scrub', 'red_wood']
#     redDeerInts = redDeerInts.reset_index(drop=True)
#     roeDeerInts = interaction_strength_3[interaction_strength_3.index=='roeDeer']
#     roeDeerInts.columns = ['roe_pony', 'roe_fallow', 'roe_grass','roe_cattle','roe_red', 'roe_roe', 'roe_pig', 'roe_scrub', 'roe_wood']
#     roeDeerInts = roeDeerInts.reset_index(drop=True)
#     tamworthPigInts = interaction_strength_3[interaction_strength_3.index=='tamworthPig']
#     tamworthPigInts.columns = ['pig_pony', 'pig_fallow', 'pig_grass','pig_cattle','pig_red', 'pig_roe', 'pig_pig', 'pig_scrub', 'pig_wood']
#     tamworthPigInts = tamworthPigInts.reset_index(drop=True)
#     thornyScrubInts = interaction_strength_3[interaction_strength_3.index=='thornyScrub']
#     thornyScrubInts.columns = ['scrub_pony', 'scrub_fallow', 'scrub_grass','scrub_cattle','scrub_red', 'scrub_roe', 'scrub_pig', 'scrub_scrub', 'scrub_wood']
#     thornyScrubInts = thornyScrubInts.reset_index(drop=True)
#     woodlandInts = interaction_strength_3[interaction_strength_3.index=='woodland']
#     woodlandInts.columns = ['wood_pony', 'wood_fallow', 'wood_grass','wood_cattle','wood_red', 'wood_roe', 'wood_pig', 'wood_scrub', 'wood_wood']
#     woodlandInts = woodlandInts.reset_index(drop=True)
#     combined = pd.concat([growth_rates, exmoorInts, fallowInts, arableInts, longhornInts, redDeerInts, roeDeerInts, tamworthPigInts, thornyScrubInts, woodlandInts], axis=1)
#     parameters = combined.loc[:, (combined != 0).any(axis=0)]
#     correlationMatrix = combined.corr()

#     for column in parameters:
#         print(column,stats.kstest(parameters[column], stats.uniform(loc=min(parameters[column]), scale=(max(parameters[column])-min(parameters[column]))).cdf))
    
#     # corr matrix
#     corr = parameters.corr()
#     # mask the upper triangle; True = do NOT show
#     mask = np.zeros_like(corr, dtype=np.bool)
#     mask[np.triu_indices_from(mask)] = True
#     # graph it 
#     f, ax = plt.subplots(figsize=(11, 9))
#     cmap =sns.diverging_palette(220, 60, l=65, center="light", as_cmap=True)
#     # Draw the heatmap with the mask and correct aspect ratio
#     heatmap = sns.heatmap(
#         corr,          # The data to plot
#         mask=mask,     # Mask some cells
#         cmap=cmap,     # What colors to plot the heatmap as
#         annot=False,    # Should the values be plotted in the cells?
#         vmax=1,       # The maximum value of the legend. All higher vals will be same color
#         vmin=-1,      # The minimum value of the legend. All lower vals will be same color
#         center=0,      # The center value of the legend. With divergent cmap, where white is
#         square=True,   # Force cells to be square
#         linewidths=0.1, # Width of lines that divide cells
#         cbar_kws={"shrink": .5},  # Extra kwargs for the legend; in this case, shrink by 50%
#     )
#     heatmap.figure.tight_layout()
#     plt.savefig('corr_matrix.png')
#     plt.show()

#     # # # significant variables (21) - ps1-stable
#     significant_variables = ["grasslandParkland","thornyScrub","woodland",
#     "pony_grass","pony_scrub","pony_wood","fallow_scrub",
#     "grass_grass","cattle_grass","cattle_scrub","cattle_wood",
#     "red_scrub","roe_grass","roe_scrub","roe_wood",
#     "pig_grass","pig_scrub",
#     "pig_wood","scrub_fallow","scrub_scrub","wood_scrub"
#  ]


#     count=1
#     for i in significant_variables:
#         # histograms for the significant ones
#         parameters[i].hist()
#         plt.title(f"Histogram of the diagonal for parameter {i}")
#         plt.savefig(f'hist_parameter{count}.png')
#         plt.show()
#         count+=1



# # # # # # ------ Reality checks -------

# def reality_1(): # remove primary producers, consumers should decline to zero 
#     # open the csv
#     all_parameters = pd.read_csv('all_parameters_ps2_unstable.csv')
#     # how many simulations were accepted? 
#     number_accepted = (len(all_parameters.loc[(all_parameters['accepted?'] == "Accepted")]))/(len(species)*2)

#     all_runs_realityCheck = []
#     all_times_realityCheck = []
#     t = np.linspace(2015, 2020, 10)
#     X0 = [1, 1, 0, 1, 1, 1, 1, 0, 0] # no primary producers, only consumers

#     # get the accepted parameters
#     accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == "Accepted")].iloc[:,1:] 
#     accepted_growth = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
#     accepted_r = pd.DataFrame(accepted_growth.values.reshape(int(number_accepted), len(species)), columns = species).to_numpy()
#     # select accepted interaction strengths
#     interaction_strength_2 = accepted_parameters.drop(["Unnamed: 0.1",'X0', 'growth', 'ID', 'accepted?'], axis=1).dropna()
#     A_reality = interaction_strength_2.to_numpy()

#     for r, A in zip(accepted_r, np.array_split(A_reality,number_accepted)):
#         realityCheck_ABC = solve_ivp(ecoNetwork, (2015, 2020), X0,  t_eval = t, args=(A, r), method = 'RK23') 
#         # append results
#         all_runs_realityCheck = np.append(all_runs_realityCheck, realityCheck_ABC.y)
#         all_times_realityCheck = np.append(all_times_realityCheck, realityCheck_ABC.t)
#     realityCheck = (np.vstack(np.hsplit(all_runs_realityCheck.reshape(len(species)*int(number_accepted), 10).transpose(),number_accepted)))
#     realityCheck = pd.DataFrame(data=realityCheck, columns=species)
#     IDs_reality = np.arange(1, 1 + number_accepted)
#     realityCheck['ID'] = np.repeat(IDs_reality,10)
#     realityCheck['time'] = all_times_realityCheck

#     # plot reality check
#     grouping1 = np.repeat(realityCheck['ID'], len(species))
#     # # extract the node values from all dataframes
#     final_runs1 = realityCheck.drop(['ID', 'time'], axis=1).values.flatten()
#     # we want species column to be spec1,spec2,spec3,spec4, etc.
#     species_realityCheck = np.tile(species, (10*int(number_accepted)))
#     # time 
#     firstODEyears = np.repeat(realityCheck['time'],len(species))
#     # put it in a dataframe
#     final_df = pd.DataFrame(
#         {'Abundance %': final_runs1, 'runNumber': grouping1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})
#     # calculate median 
#     m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
#     m.name = 'Median'
#     final_df = final_df.join(m, on=['Time','Ecosystem Element'])
#     # calculate quantiles
#     perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
#     perc1.name = 'ninetyfivePerc'
#     final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
#     perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
#     perc2.name = "fivePerc"
#     final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
#     # graph it
#     g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
#     g.map(sns.lineplot, 'Time', 'Median')
#     g.map(sns.lineplot, 'Time', 'fivePerc')
#     g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
#     for ax in g.axes.flat:
#         ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
#     g.fig.suptitle('Reality check: No primary producers')

#     plt.tight_layout()
#     plt.savefig('reality_check_noFood.png')

#     plt.show()




# def reality_2(): # herbivores overloaded
#     # open the csv
#     final_results = pd.read_csv('all_runs_ps2_unstable.csv')
#     all_parameters = pd.read_csv('all_parameters_ps2_unstable.csv')

#     # how many simulations were accepted? 
#     number_accepted = (len(all_parameters.loc[(all_parameters['accepted?'] == "Accepted")]))/(len(species)*2)

#     all_runs_realityCheck = []
#     all_times_realityCheck = []
#     t = np.linspace(2005, 2055, 100)
#     X0 = [2, 2, 1, 2, 2, 2, 2, 1, 1] # overloaded consumers

#     # get the accepted parameters
#     accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == "Accepted")].iloc[:,1:]
#     accepted_growth = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
#     accepted_r = pd.DataFrame(accepted_growth.values.reshape(int(number_accepted), len(species)), columns = species).to_numpy()
#     # select accepted interaction strengths
#     interaction_strength_2 = accepted_parameters.drop(["Unnamed: 0.1",'X0', 'growth', 'ID', 'accepted?'], axis=1).dropna()
#     A_reality = interaction_strength_2.to_numpy()

#     for r, A in zip(accepted_r, np.array_split(A_reality,number_accepted)):
#         # give herbivores 0 diag they don't decline - we want to make sure habitats decline
#         A[[0],2] = 5
#         A[[1],2] = 5
#         A[[3],2] = 5
#         A[[4],2] = 5
#         A[[5],2] = 5
#         A[[6],2] = 5
#         realityCheck_ABC = solve_ivp(ecoNetwork_nor, (2005, 2055), X0,  t_eval = t, args=(A, r), method = 'RK23') 
#         # append results
#         all_runs_realityCheck = np.append(all_runs_realityCheck, realityCheck_ABC.y)
#         all_times_realityCheck = np.append(all_times_realityCheck, realityCheck_ABC.t)
#     realityCheck = (np.vstack(np.hsplit(all_runs_realityCheck.reshape(len(species)*int(number_accepted), 100).transpose(),number_accepted)))
#     realityCheck = pd.DataFrame(data=realityCheck, columns=species)
#     IDs_reality = np.arange(1, 1 + number_accepted)
#     realityCheck['ID'] = np.repeat(IDs_reality,100)
#     realityCheck['time'] = all_times_realityCheck

#     # plot reality check
#     grouping1 = np.repeat(realityCheck['ID'], len(species))
#     # # extract the node values from all dataframes
#     final_runs1 = realityCheck.drop(['ID', 'time'], axis=1).values.flatten()
#     # we want species column to be spec1,spec2,spec3,spec4, etc.
#     species_realityCheck = np.tile(species, (100*int(number_accepted)))
#     # time 
#     firstODEyears = np.repeat(realityCheck['time'],len(species))
#     # put it in a dataframe
#     final_df = pd.DataFrame(
#         {'Abundance %': final_runs1, 'runNumber': grouping1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})
#     # calculate median 
#     m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
#     m.name = 'Median'
#     final_df = final_df.join(m, on=['Time','Ecosystem Element'])
#     # calculate quantiles
#     perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
#     perc1.name = 'ninetyfivePerc'
#     final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
#     perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
#     perc2.name = "fivePerc"
#     final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
#     # graph it
#     g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
#     g.map(sns.lineplot, 'Time', 'Median')
#     g.map(sns.lineplot, 'Time', 'fivePerc')
#     g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
#     for ax in g.axes.flat:
#         ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
#     g.fig.suptitle('Reality check: Overloaded herbivores')

#     plt.tight_layout()
#     plt.savefig('reality_check_overloadedHerbs.png')

#     plt.show()



# def reality_3(): # run for 100 years with no herbivory or woodland
#     # open the csv
#     final_results = pd.read_csv('all_runs_ps2_unstable.csv')
#     all_parameters = pd.read_csv('all_parameters_ps2_unstable.csv')

#     # how many simulations were accepted? 
#     number_accepted = (len(all_parameters.loc[(all_parameters['accepted?'] == "Accepted")]))/(len(species)*2)

#     all_runs_realityCheck = []
#     all_times_realityCheck = []
#     t = np.linspace(2005, 2105, 100)
#     X0 = [0, 0, 1, 0, 0, 0, 0, 1, 0] # no herbivores or wood

#     # get the accepted parameters
#     accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == "Accepted")].iloc[:,1:]
#     accepted_growth = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
#     accepted_r = pd.DataFrame(accepted_growth.values.reshape(int(number_accepted), len(species)), columns = species).to_numpy()
#     # select accepted interaction strengths
#     interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID', 'accepted?', "Unnamed: 0.1"], axis=1).dropna()
#     A_reality = interaction_strength_2.to_numpy()

#     for r, A in zip(accepted_r, np.array_split(A_reality,number_accepted)):
#         r[8] = 0 
#         realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2105), X0,  t_eval = t, args=(A, r), method = 'RK23') 
#         # append results
#         all_runs_realityCheck = np.append(all_runs_realityCheck, realityCheck_ABC.y)
#         all_times_realityCheck = np.append(all_times_realityCheck, realityCheck_ABC.t)
#     realityCheck = (np.vstack(np.hsplit(all_runs_realityCheck.reshape(len(species)*int(number_accepted), 100).transpose(),number_accepted)))
#     realityCheck = pd.DataFrame(data=realityCheck, columns=species)
#     IDs_reality = np.arange(1, 1 + number_accepted)
#     realityCheck['ID'] = np.repeat(IDs_reality,100)
#     realityCheck['time'] = all_times_realityCheck

#     # plot reality check
#     grouping1 = np.repeat(realityCheck['ID'], len(species))
#     # # extract the node values from all dataframes
#     final_runs1 = realityCheck.drop(['ID', 'time'], axis=1).values.flatten()
#     # we want species column to be spec1,spec2,spec3,spec4, etc.
#     species_realityCheck = np.tile(species, (100*int(number_accepted)))
#     # time 
#     firstODEyears = np.repeat(realityCheck['time'],len(species))
#     # put it in a dataframe
#     final_df = pd.DataFrame(
#         {'Abundance %': final_runs1, 'runNumber': grouping1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})
#     # calculate median 
#     m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
#     m.name = 'Median'
#     final_df = final_df.join(m, on=['Time','Ecosystem Element'])
#     # calculate quantiles
#     perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
#     perc1.name = 'ninetyfivePerc'
#     final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
#     perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
#     perc2.name = "fivePerc"
#     final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
#     # graph it
#     g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
#     g.map(sns.lineplot, 'Time', 'Median')
#     g.map(sns.lineplot, 'Time', 'fivePerc')
#     g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
#     for ax in g.axes.flat:
#         ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
#     g.fig.suptitle('Reality check: No woodland, consumers')

#     plt.tight_layout()
#     plt.savefig('reality_check_noHerbivoryWood.png')

#     plt.show()


# def reality_4(): # run for 100 years with no herbivory, scrub or wood
#     # open the csv
#     final_results = pd.read_csv('all_runs_ps2_unstable.csv')
#     all_parameters = pd.read_csv('all_parameters_ps2_unstable.csv')

#     # how many simulations were accepted? 
#     number_accepted = (len(all_parameters.loc[(all_parameters['accepted?'] == "Accepted")]))/(len(species)*2)

#     all_runs_realityCheck = []
#     all_times_realityCheck = []
#     t = np.linspace(2005, 2105, 100)
#     X0 = [0, 0, 1, 0, 0, 0, 0, 0, 0] # no herbivores, scrub, wood

#     # get the accepted parameters
#     accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == "Accepted")].iloc[:,1:]
#     accepted_growth = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
#     accepted_r = pd.DataFrame(accepted_growth.values.reshape(int(number_accepted), len(species)), columns = species).to_numpy()
#     # select accepted interaction strengths
#     interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID', 'accepted?', "Unnamed: 0.1"], axis=1).dropna()
#     A_reality = interaction_strength_2.to_numpy()

#     for r, A in zip(accepted_r, np.array_split(A_reality,number_accepted)):
#         r[8] = 0
#         r[7] = 0
#         realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2105), X0,  t_eval = t, args=(A, r), method = 'RK23') 
#         # append results
#         all_runs_realityCheck = np.append(all_runs_realityCheck, realityCheck_ABC.y)
#         all_times_realityCheck = np.append(all_times_realityCheck, realityCheck_ABC.t)
#     realityCheck = (np.vstack(np.hsplit(all_runs_realityCheck.reshape(len(species)*int(number_accepted), 100).transpose(),number_accepted)))
#     realityCheck = pd.DataFrame(data=realityCheck, columns=species)
#     IDs_reality = np.arange(1, 1 + number_accepted)
#     realityCheck['ID'] = np.repeat(IDs_reality,100)
#     realityCheck['time'] = all_times_realityCheck

#     # plot reality check
#     grouping1 = np.repeat(realityCheck['ID'], len(species))
#     # # extract the node values from all dataframes
#     final_runs1 = realityCheck.drop(['ID', 'time'], axis=1).values.flatten()
#     # we want species column to be spec1,spec2,spec3,spec4, etc.
#     species_realityCheck = np.tile(species, (100*int(number_accepted)))
#     # time 
#     firstODEyears = np.repeat(realityCheck['time'],len(species))
#     # put it in a dataframe
#     final_df = pd.DataFrame(
#         {'Abundance %': final_runs1, 'runNumber': grouping1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})
#     # calculate median 
#     m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
#     m.name = 'Median'
#     final_df = final_df.join(m, on=['Time','Ecosystem Element'])
#     # calculate quantiles
#     perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
#     perc1.name = 'ninetyfivePerc'
#     final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
#     perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
#     perc2.name = "fivePerc"
#     final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
#     # graph it
#     g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
#     g.map(sns.lineplot, 'Time', 'Median')
#     g.map(sns.lineplot, 'Time', 'fivePerc')
#     g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
#     for ax in g.axes.flat:
#         ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
#     g.fig.suptitle('Reality check: No woodland, scrubland, consumers')

#     plt.tight_layout()
#     plt.savefig('reality_check_noHerbivoryWoodScrub.png')

#     plt.show()



# def reality_5(): # run for 100 years with no herbivory
#     # open the csv
#     final_results = pd.read_csv('all_runs_ps2_unstable.csv')
#     all_parameters = pd.read_csv('all_parameters_ps2_unstable.csv')

#     # how many simulations were accepted? 
#     number_accepted = (len(all_parameters.loc[(all_parameters['accepted?'] == "Accepted")]))/(len(species)*2)

#     all_runs_realityCheck = []
#     all_times_realityCheck = []
#     t = np.linspace(2005, 2105, 100)
#     X0 = [0, 0, 1, 0, 0, 0, 0, 1, 1] # no herbivores

#     # get the accepted parameters
#     accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == "Accepted")].iloc[:,1:]
#     accepted_growth = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
#     accepted_r = pd.DataFrame(accepted_growth.values.reshape(int(number_accepted), len(species)), columns = species).to_numpy()
#     # select accepted interaction strengths
#     interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID', 'accepted?', "Unnamed: 0.1"], axis=1).dropna()
#     A_reality = interaction_strength_2.to_numpy()

#     for r, A in zip(accepted_r, np.array_split(A_reality,number_accepted)):
#         realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2105), X0,  t_eval = t, args=(A, r, "ps2"), method = 'RK23') 
#         # append results
#         all_runs_realityCheck = np.append(all_runs_realityCheck, realityCheck_ABC.y)
#         all_times_realityCheck = np.append(all_times_realityCheck, realityCheck_ABC.t)
#     realityCheck = (np.vstack(np.hsplit(all_runs_realityCheck.reshape(len(species)*int(number_accepted), 100).transpose(),number_accepted)))
#     realityCheck = pd.DataFrame(data=realityCheck, columns=species)
#     IDs_reality = np.arange(1, 1 + number_accepted)
#     realityCheck['ID'] = np.repeat(IDs_reality,100)
#     realityCheck['time'] = all_times_realityCheck

#     # plot reality check
#     grouping1 = np.repeat(realityCheck['ID'], len(species))
#     # # extract the node values from all dataframes
#     final_runs1 = realityCheck.drop(['ID', 'time'], axis=1).values.flatten()
#     # we want species column to be spec1,spec2,spec3,spec4, etc.
#     species_realityCheck = np.tile(species, (100*int(number_accepted)))
#     # time 
#     firstODEyears = np.repeat(realityCheck['time'],len(species))
#     # put it in a dataframe
#     final_df = pd.DataFrame(
#         {'Abundance %': final_runs1, 'runNumber': grouping1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})
#     # calculate median 
#     m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
#     m.name = 'Median'
#     final_df = final_df.join(m, on=['Time','Ecosystem Element'])
#     # calculate quantiles
#     perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
#     perc1.name = 'ninetyfivePerc'
#     final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
#     perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
#     perc2.name = "fivePerc"
#     final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
#     # graph it
#     g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
#     g.map(sns.lineplot, 'Time', 'Median')
#     g.map(sns.lineplot, 'Time', 'fivePerc')
#     g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
#     for ax in g.axes.flat:
#         ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
#     g.fig.suptitle('Reality check: No primary producers')

#     plt.tight_layout()
#     plt.savefig('reality_check_noHerbivory.png')

#     plt.show()


# # histograms of passed filters
# def hist_passed_filters():
#     # open the csvs and look at one year, and only at accepted runs

#     final_results_ps1 = pd.read_csv('all_runs_ps1_unstable.csv')
#     final_results_ps1["parameter_set"] = "PS1 - no stability"
#     final_results_ps1 = final_results_ps1.loc[(final_results_ps1['accepted?'] == "Accepted") & (final_results_ps1['time'] == 2020.75)] 

#     print(((final_results_ps1[["passed_filters"]])/29).mean())
#     final_results_ps2 = pd.read_csv('all_runs_ps2_unstable.csv')
#     final_results_ps2["parameter_set"] = "PS2 - no stability"
#     final_results_ps2 = final_results_ps2.loc[(final_results_ps2['accepted?'] == "Accepted") & (final_results_ps2['time'] == 2020.75)] 
#     print(((final_results_ps2[["passed_filters"]])/29).mean())

#     final_results_ps1_stable = pd.read_csv('all_runs_ps1_stable.csv')
#     final_results_ps1_stable["parameter_set"] = "PS1 - with stability"
#     final_results_ps1_stable = final_results_ps1_stable.loc[(final_results_ps1_stable['accepted?'] == "Accepted") & (final_results_ps1_stable['time'] == 2020.75)] 
#     print(((final_results_ps1_stable[["passed_filters"]])/29).mean())

#     final_results_ps2_stable = pd.read_csv('all_runs_ps2_stable.csv')
#     final_results_ps2_stable["parameter_set"] = "PS2 - with stability"
#     final_results_ps2_stable = final_results_ps2_stable.loc[(final_results_ps2_stable['accepted?'] == "Accepted") & (final_results_ps2_stable['time'] == 2020.75)] 
#     print(((final_results_ps2_stable[["passed_filters"]])/29).mean())


#     # concat them
#     filters = pd.concat([final_results_ps1, final_results_ps2, final_results_ps1_stable, final_results_ps2_stable]).reset_index(drop=True)
#     # histogram of percentage filters passed for the accepted runs
#     filters["passed_filters"] = (filters[["passed_filters"]])/29 # show percentage passed

#     # d = np.diff(np.unique(filters[['passed_filters']])).min()
#     # left_of_first_bin = np.unique(filters[['passed_filters']]).min() - float(d)/2
#     # right_of_last_bin = np.unique(filters[['passed_filters']]).max() + float(d)/2

#     fig, ax = plt.subplots()
#     # sns.histplot(
#     #     data=filters, x='passed_filters', multiple='stack',
#     #     ax=ax, palette="hls", hue='parameter_set', bins=np.arange(left_of_first_bin, right_of_last_bin + d, d)
#     # )
#     sns.histplot(
#         data=filters, x='passed_filters', multiple='stack',
#         ax=ax, palette="hls", hue='parameter_set'
#     )
#     plt.xlabel("Percentage of filters passed")
#     plt.ylabel("Count")
#     plt.title("Percentage of filters passed by the accepted runs")
#     plt.savefig('histograms_filters_withStability')
#     plt.show()


# def graph_accepted_rejected():
#     # open dataframes
#     final_results = pd.read_csv('all_runs_ps2_unstable.csv').iloc[:,2:]
#     first_ode = final_results.loc[final_results['time'] <= 2008.75]
#     second_ode = final_results.loc[final_results['time'] > 2008.75]
#     # pick 100 runs (to not overload the ram) - first for accepted runs
#     final_results_accepted_1 = first_ode.loc[(first_ode['accepted?'] == "Accepted")].iloc[0:(100*32),:] # show some first ode accepted runs
#     final_results_accepted_2 = second_ode.loc[(second_ode['accepted?'] == "Accepted")] # show all second ode accepted
#     # and same for the rejected runs (only show a selection)
#     final_results_rejected_1 = first_ode.loc[(first_ode['accepted?'] == "Rejected")].iloc[0:(100*32),:] # for the first ode
#     final_results_rejected_2 = second_ode.loc[(second_ode['accepted?'] == "Rejected")].iloc[0:(100*32),:] # for the first ode
#     # concat them
#     final_results = pd.concat([final_results_accepted_1, final_results_accepted_2,final_results_rejected_1,final_results_rejected_2])

#     # put it in a dataframe
#     y_values = final_results[["exmoorPony", "fallowDeer", "grasslandParkland", "longhornCattle", "redDeer", "roeDeer", "tamworthPig","thornyScrub", "woodland"]].values.flatten()
#     species_list = np.tile(["exmoorPony", "fallowDeer", "grasslandParkland", "longhornCattle", "redDeer", "roeDeer", "tamworthPig","thornyScrub", "woodland"],len(final_results)) 
#     indices = np.repeat(final_results['time'], 9)
#     runType = np.repeat(final_results['accepted?'], 9)
#     # here's the final dataframe
#     final_results = pd.DataFrame(
#         {'Abundance %': y_values, 'Ecosystem Element': species_list, 'Time': indices, 'runType': runType})

#     # calculate median 
#     m = final_results.groupby(['Time', 'runType', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
#     m.name = 'Median'
#     final_results = final_results.join(m, on=['Time', 'runType', 'Ecosystem Element'])
#     # calculate quantiles
#     perc1 = final_results.groupby(['Time', 'runType', 'Ecosystem Element'])['Abundance %'].quantile(.95)
#     perc1.name = 'ninetyfivePerc'
#     final_results = final_results.join(perc1, on=['Time', 'runType', 'Ecosystem Element'])
#     perc2 = final_results.groupby(['Time', 'runType', 'Ecosystem Element'])['Abundance %'].quantile(.05)
#     perc2.name = "fivePerc"
#     final_results = final_results.join(perc2, on=['Time','runType', 'Ecosystem Element'])
#     final_df = final_results.reset_index(drop=True)

#     # now graph it
#     palette=['#db5f57', '#57d3db', '#57db5f','#5f57db', '#db57d3']
#     f = sns.FacetGrid(final_df, col="Ecosystem Element", hue = "runType", palette = palette, col_wrap=4, sharey = False)
#     f.map(sns.lineplot, 'Time', 'Median')
#     f.map(sns.lineplot, 'Time', 'fivePerc')
#     f.map(sns.lineplot, 'Time', 'ninetyfivePerc')
#     for ax in f.axes.flat:
#         ax.fill_between(ax.lines[2].get_xdata(),ax.lines[2].get_ydata(), ax.lines[4].get_ydata(),color="#db5f57", alpha =0.2)
#         ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[5].get_ydata(), color="#57d3db", alpha=0.2)
#         ax.set_ylabel('Abundance')
#         ax.set_xlabel('Time (Years)')

#     # add subplot titles
#     axes = f.axes.flatten()
#     # fill between the quantiles
#     axes[0].set_title("Exmoor pony")
#     axes[1].set_title("Fallow deer")
#     axes[2].set_title("Grassland")
#     axes[3].set_title("Longhorn cattle")
#     axes[4].set_title("Red deer")
#     axes[5].set_title("Roe deer")
#     axes[6].set_title("Tamworth pig")
#     axes[7].set_title("Thorny scrub")
#     axes[8].set_title("Woodland")
#     # add filter lines
#     # 2009
#     f.axes[2].vlines(x=2008.75,ymin=0.18,ymax=1, color='r')
#     f.axes[5].vlines(x=2008.75,ymin=1.7,ymax=3.3, color='r')
#     f.axes[7].vlines(x=2008.75,ymin=1,ymax=4.9, color='r')
#     f.axes[8].vlines(x=2008.75,ymin=1,ymax=2.9, color='r')
#     f.axes[0].plot(2009, 1, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2009, 1, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2009, 1, 'go',markersize=2.5) # and forcings
#     # 2010
#     f.axes[1].plot(2010, 1, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2010, 1.5, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2010, 0.85, 'go',markersize=2.5) # and forcings
#     # 2011
#     f.axes[0].plot(2011, 0.65, 'go',markersize=2.5) # and forcings
#     f.axes[1].plot(2011, 1.9, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2011, 1.7, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2011, 1.1, 'go',markersize=2.5) # and forcings
#     # 2012
#     f.axes[0].plot(2012, 0.74, 'go',markersize=2.5) # and forcings
#     f.axes[1].plot(2012, 2.4, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2012, 2.2, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2012, 1.7, 'go',markersize=2.5) # and forcings
#     # 2013
#     f.axes[0].plot(2013, 0.43, 'go',markersize=2.5) # and forcings
#     f.axes[1].plot(2013, 2.4, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2013, 2.4, 'go',markersize=2.5) # and forcings
#     f.axes[4].plot(2013, 1, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2013, 0.3, 'go',markersize=2.5) # and forcings
#     # 2014
#     f.axes[0].plot(2014, 0.44, 'go',markersize=2.5) # and forcings
#     f.axes[1].plot(2014, 2.4, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2014, 5, 'go',markersize=2.5) # and forcings
#     f.axes[4].plot(2014, 1, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2014, 0.9, 'go',markersize=2.5) # and forcings
#     # 2015
#     f.axes[0].vlines(x=2015.75,ymin=0.39,ymax=0.49, color='r')
#     f.axes[1].vlines(x=2015.75,ymin=2.7,ymax=3.9, color='r')
#     f.axes[3].vlines(x=2015.75,ymin=2,ymax=2.9, color='r')
#     f.axes[4].vlines(x=2015.75,ymin=0.6,ymax=1.4, color='r')
#     f.axes[6].vlines(x=2015.75,ymin=0.6,ymax=1.6, color='r')
#     f.axes[0].plot(2015, 0.43, 'go',markersize=2.5) # and forcings
#     f.axes[1].plot(2015, 2.4, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2015, 2, 'go',markersize=2.5) # and forcings
#     f.axes[4].plot(2015, 1, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2015, 0.71, 'go',markersize=2.5) # and forcings
#     # 2016
#     f.axes[0].vlines(x=2016.75,ymin=0.43,ymax=0.5, color='r')
#     f.axes[1].vlines(x=2016.75,ymin=3.3,ymax=4.5, color='r')
#     f.axes[3].vlines(x=2016.75,ymin=1.6,ymax=2.5, color='r')
#     f.axes[4].vlines(x=2016.75,ymin=1.6,ymax=2.4, color='r')
#     f.axes[6].vlines(x=2016.75,ymin=0.35,ymax=1.4, color='r')
#     f.axes[0].plot(2016, 0.48, 'go',markersize=2.5) # and forcings
#     f.axes[1].plot(2016, 3.3, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2016, 1.7, 'go',markersize=2.5) # and forcings
#     f.axes[4].plot(2016, 2, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2016, 0.7, 'go',markersize=2.5) # and forcings
#     # 2017
#     f.axes[0].vlines(x=2017.75,ymin=0.39,ymax=0.48, color='r')
#     f.axes[1].vlines(x=2017.75,ymin=5.4,ymax=6.6, color='r')
#     f.axes[3].vlines(x=2017.75,ymin=1.6,ymax=2.5, color='r')
#     f.axes[4].vlines(x=2017.75,ymin=0.7,ymax=1.5, color='r')
#     f.axes[6].vlines(x=2017.75,ymin=0.6,ymax=1.6, color='r')
#     f.axes[0].plot(2017, 0.43, 'go',markersize=2.5) # and forcings
#     f.axes[1].plot(2017, 3.9, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2017, 1.7, 'go',markersize=2.5) # and forcings
#     f.axes[4].plot(2017, 1.1, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2017, 0.95, 'go',markersize=2.5) # and forcings
#     # 2018
#     f.axes[1].vlines(x=2018.75,ymin=6.9,ymax=8.1, color='r')
#     f.axes[3].vlines(x=2018.75,ymin=1.7,ymax=2.7, color='r')
#     f.axes[4].vlines(x=2018.75,ymin=1.5,ymax=2.2, color='r')
#     f.axes[6].vlines(x=2018.75,ymin=0.7,ymax=1.7, color='r')
#     f.axes[0].plot(2018, 0, 'go',markersize=2.5) # and forcings
#     f.axes[1].plot(2018, 6.0, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2018, 1.9, 'go',markersize=2.5) # and forcings
#     f.axes[4].plot(2018, 1.9, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2018, 0.84, 'go',markersize=2.5) # and forcings
#     # 2019
#     f.axes[1].vlines(x=2019.75,ymin=7.7,ymax=8.9, color='r')
#     f.axes[3].vlines(x=2019.75,ymin=1.6,ymax=2.5, color='r')
#     f.axes[4].vlines(x=2019.75,ymin=2.5,ymax=3.2, color='r')
#     f.axes[6].vlines(x=2019.75,ymin=0,ymax=1, color='r')
#     f.axes[0].plot(2019, 0, 'go',markersize=2.5) # and forcings
#     f.axes[1].plot(2019, 6.6, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2019, 1.7, 'go',markersize=2.5) # and forcings
#     f.axes[4].plot(2019, 2.9, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2019, 0.44, 'go',markersize=2.5) # and forcings
#     # 2020
#     f.axes[0].vlines(x=2020.75,ymin=0.61,ymax=0.7, color='r')
#     f.axes[6].vlines(x=2020.75,ymin=0.5,ymax=1.5, color='r')
#     f.axes[2].vlines(x=2020.75,ymin=0.18,ymax=0.41, color='r')
#     f.axes[5].vlines(x=2020.75,ymin=1.7,ymax=6.7, color='r')
#     f.axes[7].vlines(x=2020.75,ymin=9.7,ymax=14.4, color='r')
#     f.axes[8].vlines(x=2020.75,ymin=2.8,ymax=4.6, color='r')
#     f.axes[0].plot(2020, 0.65, 'go',markersize=2.5) # and forcings
#     f.axes[1].plot(2020, 5.9, 'go',markersize=2.5) # and forcings
#     f.axes[3].plot(2020, 1.5, 'go',markersize=2.5) # and forcings
#     f.axes[4].plot(2020, 2.7, 'go',markersize=2.5) # and forcings
#     f.axes[6].plot(2020, 0.55, 'go',markersize=2.5) # and forcings

#     # stop the plots from overlapping
#     f.fig.suptitle('Accepted vs. Rejected Runs')
#     plt.tight_layout()
#     plt.legend(labels=['Accepted Runs', 'Rejected Runs'],bbox_to_anchor=(2.2, 0), loc='lower right', fontsize=12)
#     plt.savefig('rejected_accepted_runs.png')
#     plt.show()