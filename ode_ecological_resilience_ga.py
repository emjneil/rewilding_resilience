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



# Ecological resilience - can we stop the system from shifting? 



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





                                                        # # # # # Perturbation 1: Growth rates temporarily reduced # # # # # 


# stop it shifting from current stable state to new one (grassland = 0) - want to keep it where it is even with high magnitude change
# look at 35 years and -50% reduction in growth rates

def objectiveFunction_perturb1(x):  

    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]

    # we want to check if our original PS passes the requirements or not
    accepted_parameters = pd.read_csv('ode_best_ps.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

    r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    r = pd.DataFrame(r.values.reshape(1, len(species)), columns = species).to_numpy().flatten()

    A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()


    exmoor_stocking = x[0]
    fallow_stocking = x[1]
    cattle_stocking = x[2]
    red_deer_stocking = x[3]
    pig_stocking = x[4]


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


    for y in range(1000): 
        years_covered += 1
        # run the model for that year 
        starting_values = last_values_2020.copy()

        if years_covered >= 516: 
            starting_values[0] = 0.65
            starting_values[1] = 5.9 
            starting_values[3] = 1.5
            starting_values[4] = 2.7
            starting_values[6] = 0.55

        # perturb it at time 516-magnitude 
        if years_covered <= 516 + 35 and years_covered >= 516: 

            starting_values[0] = 0.65 * x[0]
            starting_values[1] = 5.9 * x[1]
            starting_values[3] = 1.5 * x[2]
            starting_values[4] = 2.7 * x[3]
            starting_values[6] = 0.55 * x[4]

            r[2] = 0.91 * -0.5
            r[7] = 0.35 * -0.5
            r[8] = 0.1 * -0.5
        else: 
            r[2] = 0.91
            r[7] = 0.35
            r[8] = 0.1       

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
    y_future = (np.vstack(np.hsplit(combined_future.reshape(len(species), 3000).transpose(),1)))
    y_future = pd.DataFrame(data=y_future, columns=species)
    # put into df
    y_future['Time'] = combined_future_times

    # what is the mean across the last 100 years? to account for stochasticity
    filtered_result = (
        # end state should be = to previous state
        # ((((statistics.mean(list(y_future.loc[y_future["Time"] > 900, 'grasslandParkland'])))-0.08)/0.08)**2) +
        # ((((statistics.mean(list(y_future.loc[y_future["Time"] > 900, 'thornyScrub'])))-6.7)/6.7)**2) +
        # ((((statistics.mean(list(y_future.loc[y_future["Time"] > 900, 'woodland'])))-9.42)/9.42)**2)
        ((((((y_future.loc[y_future["Time"] == 1016, 'thornyScrub'].item())))-6.7)/6.7)**2) +
        ((((((y_future.loc[y_future["Time"] == 1016, 'woodland'].item())))-9.42)/9.42)**2) +
        ((((((y_future.loc[y_future["Time"] == 1016, 'grasslandParkland'].item())))-0.08)/0.08)**2)

        )


    # print
    print("n:", filtered_result, "grass", last_values_2020[2], "scrub", last_values_2020[7], "wood", last_values_2020[8])

    # return the output
    return filtered_result
   










                                                        # # # # # Perturbation 2: Increasing amounts of noise # # # # # 






def objectiveFunction_perturb2(x): # stochasticity experiment

    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]

    # we want to check if our original PS passes the requirements or not
    accepted_parameters = pd.read_csv('ode_best_ps.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

    r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    r = pd.DataFrame(r.values.reshape(1, len(species)), columns = species).to_numpy().flatten()

    A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()


    exmoor_stocking = x[0]
    fallow_stocking = x[1]
    cattle_stocking = x[2]
    red_deer_stocking = x[3]
    pig_stocking = x[4]


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

    # start with mean of 1 and sd of 0
    sd = 0
    noise = np.random.normal(1, sd, 1).item()


    for y in range(1000): 
        years_covered += 1            
        starting_values = last_values_2020.copy()

        # apply the perturbations; don't let it go below 0
        if years_covered >= 516: 
            # increase the amount of noise over time
            sd += 0.003
            # noise term is SD and at every point, choose a random normal value that has mean of 0 and that SD.
            noise = np.random.normal(1, sd, 1).item()
            r[2] = 0.91 * noise
            r[7] = 0.35 * noise
            r[8] = 0.1 * noise

            starting_values[0] = 0.65 *exmoor_stocking
            starting_values[1] = 5.9 *fallow_stocking
            starting_values[3] = 1.5 *cattle_stocking
            starting_values[4] = 2.7 *red_deer_stocking
            starting_values[6] = 0.55 *pig_stocking

        else:
            r[2] = 0.91
            r[7] = 0.35
            r[8] = 0.1

            starting_values[0] = 0.65
            starting_values[1] = 5.9
            starting_values[3] = 1.5
            starting_values[4] = 2.7
            starting_values[6] = 0.55


        # after 100 years of perturbations (616) decrease woodland percentages by a number
        if years_covered == 616: 
            starting_values[8] = last_values_2020[8]  * 0.5
        else:
            starting_values[8] = last_values_2020[8]

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
    y_future = (np.vstack(np.hsplit(combined_future.reshape(len(species), 3000).transpose(),1)))
    y_future = pd.DataFrame(data=y_future, columns=species)
    # put into df
    y_future['Time'] = combined_future_times

    # what is the mean across the last 100 years? to account for stochasticity
    filtered_result = (
        # end state should be = to previous state
        ((((statistics.mean(list(y_future.loc[y_future["Time"] > 900, 'grasslandParkland'])))-0.08)/0.08)**2) +
        ((((statistics.mean(list(y_future.loc[y_future["Time"] > 900, 'thornyScrub'])))-6.7)/6.7)**2) +
        ((((statistics.mean(list(y_future.loc[y_future["Time"] > 900, 'woodland'])))-9.42)/9.42)**2))
        # ((((((y_future.loc[y_future["Time"] == 1016, 'thornyScrub'].item())))-6.7)/6.7)**2) +
        # ((((((y_future.loc[y_future["Time"] == 1016, 'woodland'].item())))-9.42)/9.42)**2) +
        # ((((((y_future.loc[y_future["Time"] == 1016, 'grasslandParkland'].item())))-0.08)/0.08)**2))


    # print
    if filtered_result < 1: 
        print("n:", filtered_result, last_values_2020)

    # return the output
    return filtered_result





                                                        # # # # # Run the optimiser # # # # # 



def run_optimizer():
    # Define the bounds: 0-100x their current levels
    bds = np.array([
        # [0,10],[0,10],[0,10],[0,10],[0,10]])
        [1,1],[1,1],[1,1],[1,1],[1,1]])



    # popsize and maxiter are defined at the top of the page, was 10x100
    algorithm_param = {'max_num_iteration':1,
                    'population_size':1,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv': 2}


    optimization =  ga(function = objectiveFunction_perturb2, dimension = 5, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=6000)
    optimization.run()
    outputs = list(optimization.output_dict["variable"]) + [(optimization.output_dict["function"])]
    # return excel with rows = output values and number of filters passed
    pd.DataFrame(outputs).to_excel('eem_optim_outputs_perturb2.xlsx', header=False, index=False)
    return optimization.output_dict







                                                        # # # # # Graph and save outputs of Perturbation 1 # # # # # 




def uncertainty_perturb1():
    # open the accepted parameters

    output_parameters = run_optimizer()


    exmoor_stocking = output_parameters["variable"][0]
    fallow_stocking = output_parameters["variable"][1]
    cattle_stocking =output_parameters["variable"][2]
    red_deer_stocking =output_parameters["variable"][3]
    pig_stocking = output_parameters["variable"][4]



    accepted_parameters = pd.read_csv('eem_accepted_parameters.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]

    all_r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    all_r = pd.DataFrame(all_r.values.reshape(100, len(species)), columns = species).to_numpy()

    all_A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()

    # run number
    run_number = 0
    all_timeseries = []
    final_df = {}


    for r, A in zip(all_r, np.array_split(all_A,100)):


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
        all_noise = []

        # start with mean of 1 and sd of 0
        sd = 0
        noise = np.random.normal(1, sd, 1).item()


        for y in range(1000): 
            years_covered += 1
            # run the model for that year 
            starting_values = last_values_2020.copy()

            if years_covered >= 516: # change stocking densities when perturbation happens
                starting_values[0] = 0.65 * exmoor_stocking
                starting_values[1] = 5.9 * fallow_stocking
                starting_values[3] = 1.5 * cattle_stocking
                starting_values[4] = 2.7 * red_deer_stocking
                starting_values[6] = 0.55 * pig_stocking
            else: 
                starting_values[0] = 0.65
                starting_values[1] = 5.9 
                starting_values[3] = 1.5
                starting_values[4] = 2.7
                starting_values[6] = 0.55

            # perturb it at time 516-magnitude 
            if years_covered <= 516 + 35 and years_covered >= 516: 
                r[2] = 0.91 * -0.5
                r[7] = 0.35 * -0.5
                r[8] = 0.1 * -0.5
            else: 
                r[2] = 0.91
                r[7] = 0.35
                r[8] = 0.1       

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
        y_future = (np.vstack(np.hsplit(combined_future.reshape(len(species), 3000).transpose(),1)))
        y_future = pd.DataFrame(data=y_future, columns=species)
        # put into df
        y_future['time'] = combined_future_times
        y_future["Perturbation"] = "One"
        y_future["State"] = "Optimised"
        y_future["run_number"] = run_number

        # get 2005-2020 time-series
        combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # reshape the outputs
        y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
        y_2 = pd.DataFrame(data=y_2, columns=species)
        y_2['time'] = combined_times
        y_2["Perturbation"] = "One"
        y_2["State"] = "Optimised"        
        y_2["run_number"] = run_number


        # concat all time-series
        timeseries = pd.concat([y_2, y_future])

        all_timeseries.append(timeseries)


        run_number += 1
        print(run_number)

        all_timeseries.append(timeseries)


    landscape = pd.DataFrame.from_dict(final_df, "index")

    all_timeseries = pd.concat(all_timeseries)
    all_timeseries.to_csv("eem_ga_timeseries_uncertainty_ga_perturb1.csv")



# uncertainty_perturb1()








                                                        # # # # # Graph and save outputs of Perturbation 2: Increasing amounts of noise # # # # # 




### remember to change stocking and noise 

def uncertainty_perturb2():
    # open the accepted parameters

    output_parameters = run_optimizer()


    exmoor_stocking = output_parameters["variable"][0]
    fallow_stocking = output_parameters["variable"][1]
    cattle_stocking =output_parameters["variable"][2]
    red_deer_stocking =output_parameters["variable"][3]
    pig_stocking = output_parameters["variable"][4]



    accepted_parameters = pd.read_csv('eem_accepted_parameters.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]

    all_r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    all_r = pd.DataFrame(all_r.values.reshape(100, len(species)), columns = species).to_numpy()

    all_A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()

    # run number
    run_number = 0
    all_timeseries = []
    final_df = {}


    for r, A in zip(all_r, np.array_split(all_A,100)):


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
        all_noise = []

        # start with mean of 1 and sd of 0
        sd = 0
        noise = np.random.normal(1, sd, 1).item()


        for y in range(1000): 
            years_covered += 1
            
            starting_values = last_values_2020.copy()

            # apply the perturbations; don't let it go below 0
            if years_covered >= 516: 
                # increase the amount of noise over time
                sd += 0.003
                # noise term is SD and at every point, choose a random normal value that has mean of 0 and that SD.
                noise = np.random.normal(1, sd, 1).item()

                r[2] = 0.91 * noise
                r[7] = 0.35 * noise
                r[8] = 0.1  * noise

                starting_values[0] = 0.65 *exmoor_stocking
                starting_values[1] = 5.9 *fallow_stocking
                starting_values[3] = 1.5 *cattle_stocking
                starting_values[4] = 2.7 *red_deer_stocking
                starting_values[6] = 0.55 *pig_stocking


            else: 
                r[2] = 0.91
                r[7] = 0.35
                r[8] = 0.1

                starting_values[0] = 0.65
                starting_values[1] = 5.9
                starting_values[3] = 1.5
                starting_values[4] = 2.7
                starting_values[6] = 0.55

            # after 100 years of perturbations (616) decrease woodland percentages by a number
            if years_covered == 616: 
                starting_values[8] = last_values_2020[8] * 0.5
            else:
                starting_values[8] = last_values_2020[8]

            # keep track of time
            t_steady = np.linspace(years_covered, years_covered+0.75, 3)
            # run model
            ABC_future = solve_ivp(ecoNetwork, (years_covered, years_covered+0.75), starting_values,  t_eval = t_steady, args=(A, r), method = 'RK23')   
            # append last values
            last_values_2020 = ABC_future.y[0:11, 2:3].flatten()

            # and save the outputs
            combined_future_runs.append(ABC_future.y)
            combined_future_times.append(ABC_future.t)
            all_noise.append(noise)

        # reshape outputs
        combined_future = np.hstack(combined_future_runs)
        combined_future_times = np.hstack(combined_future_times)
        y_future = (np.vstack(np.hsplit(combined_future.reshape(len(species), 3000).transpose(),1)))
        y_future = pd.DataFrame(data=y_future, columns=species)
        # put into df
        y_future['time'] = combined_future_times
        y_future["Perturbation"] = "Two"
        y_future["State"] = "Original"
        y_future["run_number"] = run_number
        # get 2005-2020 time-series
        combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # reshape the outputs
        y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
        y_2 = pd.DataFrame(data=y_2, columns=species)
        y_2['time'] = combined_times
        y_2["Perturbation"] = "Two"
        y_2["State"] = "Original"
        y_2["run_number"] = run_number


        # concat all time-series
        timeseries = pd.concat([y_2, y_future])



        all_timeseries.append(timeseries)


        run_number += 1
        print(run_number)

        all_timeseries.append(timeseries)

    all_timeseries = pd.concat(all_timeseries)
    # all_timeseries.to_csv("eem_ga_timeseries_uncertainty_ga_perturb2.csv")
    # all_timeseries.to_csv("eem_ga_timeseries_uncertainty_ga_original.csv")


uncertainty_perturb2()