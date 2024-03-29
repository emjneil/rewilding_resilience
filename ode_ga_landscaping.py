# ------ Optimization of the ODE --------
from geneticalgorithm import geneticalgorithm as ga
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
import statistics


# alter one species at a time to get to the following end states: 
# END STATE 1: 45% grassland, 0% woodland, 0.55% thorny scrub. 
# END STATE 2: 4.8% grassland, 15% woodland, 50.5% thorny scrub
# END STATE 3: 5.8% grassland, 76.4% woodland, 15.2% thorny scrub


# ------ Optimization --------


species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']


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



def objectiveFunction(x):

    accepted_parameters = pd.read_csv('ode_best_ps.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]

    r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    r = pd.DataFrame(r.values.reshape(1, len(species)), columns = species).to_numpy().flatten()

    A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()

    exmoor_stocking_change = x[0]
    fallow_stocking_change = x[1]
    cattle_stocking_change = x[2]
    red_stocking_change = x[3]
    pig_stocking_change = x[4]




    # # # # First, run it from 2005-2020 # # # #

    all_times = []
    t_init = np.linspace(0, 3.75, 12)
    results = solve_ivp(ecoNetwork, (0, 3.75), X0, t_eval = t_init, args=(A, r), method = 'RK23')
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

    # now run it for 500 years - does it reach this steady state?
    years_covered = 16
    combined_future_runs = []
    combined_future_times = []


    for y in range(500): 
        years_covered += 1

        # after 1k years, try to engineer it back
        starting_values = last_values_2020.copy()
        starting_values[0] = 0.65
        starting_values[1] = 5.9
        starting_values[3] = 1.5
        starting_values[4] = 2.7
        starting_values[6] = 0.55

        if years_covered >= 17:
            # starting_values[0] = 0.65 * exmoor_stocking_change

            starting_values[0] = 0.65 * exmoor_stocking_change
            starting_values[1] = 5.9 * fallow_stocking_change
            starting_values[3] = 1.5 * cattle_stocking_change
            starting_values[4] = 2.7 * red_stocking_change
            starting_values[6] = 0.55 * pig_stocking_change


        # keep track of time
        t_steady = np.linspace(years_covered, years_covered+0.75, 3)
        # run model
        ABC_future = solve_ivp(ecoNetwork, (years_covered, years_covered+0.75), starting_values,  t_eval = t_steady, args=(A, r), method = 'RK23')   
        # append last values
        last_values_2020 = ABC_future.y[0:11, 2:3].flatten()
        # and save the outputs
        combined_future_runs.append(ABC_future.y)
        combined_future_times.append(ABC_future.t)


    # reshape outputs
    combined_future = np.hstack(combined_future_runs)
    combined_future_times = np.hstack(combined_future_times)
    y_future = (np.vstack(np.hsplit(combined_future.reshape(len(species), 1500).transpose(),1)))
    y_future = pd.DataFrame(data=y_future, columns=species)
    # put into df
    y_future['Time'] = combined_future_times

    # get 2005-2020 time-series
    combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
    combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
    # reshape the outputs
    y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
    y_2 = pd.DataFrame(data=y_2, columns=species)
    y_2['Time'] = combined_times

    # concat all time-series
    timeseries = pd.concat([y_2, y_future])


    # target the end state we want
    # print(y_future.loc[y_future["Time"] == 516.75, 'grasslandParkland'].item())
    # print(y_future.loc[y_future["Time"] == 516.75, 'thornyScrub'].item())

    filtered_result = (
        # end state should be = to previous state
        ((((((y_future.loc[y_future["Time"] == 516.75, 'grasslandParkland'].item())))-0.5)/0.5)**2)
        # ((((((y_future.loc[y_future["Time"] == 516.75, 'thornyScrub'].item())))-11.6)/11.6)**2) 
        # ((((((y_future.loc[y_future["Time"] == 516.75, 'woodland'].item())))-13.2)/13.2)**2)

        )


# END STATE 1: 45% grassland, 0.55% thorny scrub, 0% woodland
# END STATE 2: 4.8% grassland, 50.5% thorny scrub, 15% woodland
# END STATE 3: 5.8% grassland, 15.2% thorny scrub, 76.4% woodland




    # print
    print("n:", filtered_result)

    if filtered_result < 0.4:
        print(filtered_result, last_values_2020)
    
    # return the output
    return filtered_result
   





def run_optimizer():
    # Define the bounds
    bds = np.array([
        [0,20], [0,20], [0,20], [0,20], [0,20]
        # [1,1], [1,1], [1,1], [1,1], [1,1]
        ])


    # popsize and maxiter are defined at the top of the page, was 10x100
    algorithm_param = {'max_num_iteration':5,
                    'population_size':100,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv': 2}


    optimization =  ga(function = objectiveFunction, dimension = 5, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=6000)
    optimization.run()
    outputs = list(optimization.output_dict["variable"]) + [(optimization.output_dict["function"])]
    # return excel with rows = output values and number of filters passed
    pd.DataFrame(outputs).to_excel('eem_optim_outputs_landscaping.xlsx', header=False, index=False)

    return optimization.output_dict





run_optimizer()










### graph outputs 

def graph_outputs():

    # Exmoor (fit: 1.0); fallow deer (0.61); longhorn cattle (1.0); red deer (0.44); tamworth pig (1.0)
    stocking_changes = [30, 4.2, 11.9, 17.6, 25] 
    index_species = [0,1,3,4,6]

    accepted_parameters = pd.read_csv('eem_accepted_parameters.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]

    all_r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    all_r = pd.DataFrame(all_r.values.reshape(100, len(species)), columns = species).to_numpy()

    all_A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()


    all_timeseries = []
    run_number = 0

    for r, A in zip(all_r, np.array_split(all_A,100)):

        for i,j in zip(stocking_changes, index_species):

            run_number += 1

            # # # # First, run it from 2005-2020 # # # #

            all_times = []
            t_init = np.linspace(0, 3.75, 12)
            results = solve_ivp(ecoNetwork, (0, 3.75), X0, t_eval = t_init, args=(A, r), method = 'RK23')
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

            # now run it for 500 years - does it reach this steady state?
            years_covered = 16
            combined_future_runs = []
            combined_future_times = []


            for y in range(500): 
                years_covered += 1

                # after 1k years, try to engineer it back
                starting_values = last_values_2020.copy()
                starting_values[0] = 0.65
                starting_values[1] = 5.9
                starting_values[3] = 1.5
                starting_values[4] = 2.7
                starting_values[6] = 0.55

                if years_covered >= 17:
                    starting_values[j] = starting_values[j] * i

                # keep track of time
                t_steady = np.linspace(years_covered, years_covered+0.75, 3)
                # run model
                ABC_future = solve_ivp(ecoNetwork, (years_covered, years_covered+0.75), starting_values,  t_eval = t_steady, args=(A, r), method = 'RK23')   
                # append last values
                last_values_2020 = ABC_future.y[0:11, 2:3].flatten()
                # and save the outputs
                combined_future_runs.append(ABC_future.y)
                combined_future_times.append(ABC_future.t)


            # reshape outputs
            combined_future = np.hstack(combined_future_runs)
            combined_future_times = np.hstack(combined_future_times)
            y_future = (np.vstack(np.hsplit(combined_future.reshape(len(species), 1500).transpose(),1)))
            y_future = pd.DataFrame(data=y_future, columns=species)
            # put into df
            y_future['Time'] = combined_future_times
            y_future["Stocking"] = i
            y_future["Species"] = species[j]
            y_future["run_number"] = run_number


            # get 2005-2020 time-series
            combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
            combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
            # reshape the outputs
            y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
            y_2 = pd.DataFrame(data=y_2, columns=species)
            y_2['Time'] = combined_times
            y_2["Stocking"] = i
            y_2["Species"] = species[j]
            y_2["run_number"] = run_number


            # concat all time-series
            timeseries = pd.concat([y_2, y_future])
            all_timeseries.append(timeseries)

            print(run_number)


    all_timeseries = pd.concat(all_timeseries)
    print(all_timeseries)
    all_timeseries.to_csv("eem_ga_timeseries_stockingchanges.csv")


# graph_outputs()








def all_varied_together():


    accepted_parameters = pd.read_csv('eem_accepted_parameters.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]


    all_r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    all_r = pd.DataFrame(all_r.values.reshape(100, len(species)), columns = species).to_numpy()

    all_A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()


    all_timeseries = []
    run_number = 0

    for r, A in zip(all_r, np.array_split(all_A,100)):


        run_number += 1

        # # # # First, run it from 2005-2020 # # # #

        all_times = []
        t_init = np.linspace(0, 3.75, 12)
        results = solve_ivp(ecoNetwork, (0, 3.75), X0, t_eval = t_init, args=(A, r), method = 'RK23')
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

        # now run it for 500 years - does it reach this steady state?
        years_covered = 16
        combined_future_runs = []
        combined_future_times = []


        for y in range(500): 
            years_covered += 1

            # # after 1k years, try to engineer it back
            starting_values = last_values_2020.copy()
            starting_values[0] = 0.65
            starting_values[1] = 5.9
            starting_values[3] = 1.5
            starting_values[4] = 2.7
            starting_values[6] = 0.55

            if years_covered >= 17:
                starting_values[0] = 0.65 * 0.64
                starting_values[1] = 5.9 * 0.06
                starting_values[3] = 1.5 *  1.4
                starting_values[4] = 2.7 * 0.32
                starting_values[6] = 0.55 * 0.26
            # keep track of time
            t_steady = np.linspace(years_covered, years_covered+0.75, 3)
            # run model
            ABC_future = solve_ivp(ecoNetwork, (years_covered, years_covered+0.75), starting_values,  t_eval = t_steady, args=(A, r), method = 'RK23')   
            # append last values
            last_values_2020 = ABC_future.y[0:11, 2:3].flatten()
            # and save the outputs
            combined_future_runs.append(ABC_future.y)
            combined_future_times.append(ABC_future.t)


        # reshape outputs
        combined_future = np.hstack(combined_future_runs)
        combined_future_times = np.hstack(combined_future_times)
        y_future = (np.vstack(np.hsplit(combined_future.reshape(len(species), 1500).transpose(),1)))
        y_future = pd.DataFrame(data=y_future, columns=species)
        # put into df
        y_future['Time'] = combined_future_times
        y_future["Stocking"] = 0
        y_future["Species"] = "75% Woodland"
        # y_future["Species"] = "No control"
        # y_future["Species"] = "Current dynamics"

        y_future["run_number"] = run_number


        # get 2005-2020 time-series
        combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # reshape the outputs
        y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
        y_2 = pd.DataFrame(data=y_2, columns=species)
        y_2['Time'] = combined_times
        y_2["Stocking"] = 0
        y_2["Species"] = "75% Woodland"
        # y_2["Species"] = "No control"
        # y_2["Species"] = "Current dynamics"

        y_2["run_number"] = run_number


        # concat all time-series
        timeseries = pd.concat([y_2, y_future])
        all_timeseries.append(timeseries)

        print(run_number)


    all_timeseries = pd.concat(all_timeseries)
    print(all_timeseries)
    # all_timeseries.to_csv("eem_ga_timeseries_stockingchanges_alltogether_aes1.csv")    
    # all_timeseries.to_csv("eem_ga_timeseries_stockingchanges_alltogether_aes2.csv")    
    all_timeseries.to_csv("eem_ga_timeseries_stockingchanges_alltogether_aes3.csv")    

    # all_timeseries.to_csv("eem_ga_timeseries_stockingchanges_currentdynamics.csv")
    # all_timeseries.to_csv("eem_ga_timeseries_stockingchanges_nocontrol.csv")



# all_varied_together()