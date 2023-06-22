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



# perturbation 1: veg repro rates are temporarily reduced at different magnitudes/durations
def vegetation_reproduction():

    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]

    # we want to check if our original PS passes the requirements or not
    r = [0, 0, 0.91, 0, 0, 0, 0, 0.35, 0.1]

    # interaction matrix
    A = [
        # exmoor pony - special case, no growth
        [0, 0, 2.54, 0, 0, 0, 0, 0.17, 0.42],
        # fallow deer 
        [0, 0, 3.87, 0, 0, 0, 0, 0.39, 0.46],
        # grassland parkland
        [-0.0023, -0.0033, -0.91, -0.016, -0.0029, -0.001, -0.009, -0.045, -0.048],
        # longhorn cattle  
        [0, 0, 5.3, 0, 0, 0, 0, 0.23, 0.42],
        # red deer  
        [0, 0, 2.9, 0, 0, 0, 0, 0.29, 0.42],
        # roe deer 
        [0, 0, 5, 0, 0, 0, 0, 0.25, 0.39],
        # tamworth pig 
        [0, 0, 3.55, 0, 0,0, 0, 0.22, 0.45],  
        # thorny scrub
        [-0.0014, -0.0047, 0, -0.0047, -0.0036, -0.0023, -0.0016, -0.014, -0.02],
        # woodland
        [-0.0045, -0.0063, 0, -0.0089, -0.0041, -0.0035, -0.0038, 0.0083, -0.006]
        ]


    # how many simulations? 
    num_simulations = 1000
    # run number
    run_number = 0

    # keep track of the time-series and final conditions
    all_timeseries = []
    final_df = {}

    # run it to 500 years (equilibrium) 
    for i in range(num_simulations):
        # generate the reproduction changes
        repro_change = random.uniform(-1,0)
        # generate the magnitudes - how many time-steps will it be decreased for? 
        magnitude = random.randint(0,10)


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

        # gather final conditions and run to equilbrium (500 years)
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
            # perturb it at time 516-magnitude 
            if years_covered == 516 + magnitude: 
                r[2] = 0.91 * repro_change
                r[7] = 0.35 * repro_change
                r[8] = 0.1 * repro_change
            else: 
                r[2] = 0.91
                r[7] = 0.35
                r[8] = 0.1       

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
        y_future["magnitude"] = magnitude
        y_future["repro_changes"] = repro_change

        # get 2005-2020 time-series
        combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # reshape the outputs
        y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
        y_2 = pd.DataFrame(data=y_2, columns=species)
        y_2['time'] = combined_times
        y_2["run_number"] = run_number
        y_2["magnitude"] = magnitude
        y_2["repro_changes"] = repro_change
        # concat all time-series
        timeseries = pd.concat([y_2, y_future])
        run_number += 1
        print(run_number)

        end = timeseries.iloc[-1]

        # then append outputs
        final_df[run_number] = {"Grassland Initial": 89.9, "Grassland End": end[2]*89.9, 
                                "Woodland Initial": 5.8, "Woodland End":  end[8]*5.8, 
                                "Thorny Scrub Initial": 4.3, "Thorny Scrub End":  end[7]*4.3, 
                                "Magnitude": magnitude,
                                "Reproduction Change": repro_change,
                                "Run Number": run_number
        }

        all_timeseries.append(timeseries)


    forecasting = pd.DataFrame.from_dict(final_df, "index")
    print(forecasting)
    all_timeseries = pd.concat(all_timeseries)
    print(all_timeseries)

    # save to csv
    forecasting.to_csv("eem_growth_exp_landscape.csv")
    all_timeseries.to_csv("eem_growth_exp_timeseries.csv")


vegetation_reproduction()