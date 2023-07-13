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


    # how many simulations? 
    num_simulations = 100
    # run number
    run_number = 0

    # keep track of the time-series and final conditions
    all_timeseries = []
    final_df = {}

    # run it to 500 years (equilibrium) 
    for i in range(num_simulations):

        X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]

        # we want to check if our original PS passes the requirements or not
        accepted_parameters = pd.read_csv('ode_best_ps.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

        r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
        r = pd.DataFrame(r.values.reshape(1, len(species)), columns = species).to_numpy().flatten()

        A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()


        # generate the reproduction changes
        repro_change = random.choice([-0.5, -0.1, 0, 0.1, 0.5, 1, 1.5])
        # generate the magnitudes - how many time-steps will it be decreased for? 
        magnitude = random.uniform(0, 50)


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
            starting_values[0] = 0.65
            starting_values[1] = 5.9
            starting_values[3] = 1.5
            starting_values[4] = 2.7
            starting_values[6] = 0.55

            # perturb it at time 516-magnitude 
            if years_covered <= 516 + magnitude and years_covered >= 516: 
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
            last_values_2020 = ABC_future.y[0:11, 2:3].flatten()

            # and save the outputs
            combined_future_runs.append(ABC_future.y)
            combined_future_times.append(ABC_future.t)

        combined_future = np.hstack(combined_future_runs)
        combined_future_times = np.hstack(combined_future_times)
        y_future = (np.vstack(np.hsplit(combined_future.reshape(len(species), 3000).transpose(),1)))
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
    all_timeseries = pd.concat(all_timeseries)


    # save to csv
    forecasting.to_csv("eem_growth_exp_landscape.csv")
    all_timeseries.to_csv("eem_growth_exp_timeseries.csv")


# vegetation_reproduction()






# # # # # # # # # Perturbation 2 # # # # # # # # #


# each year (516-616), give veg growth rates noisy perturbation (pos or neg)
# these increase in magnitude over time
# after 100 years (616) decrease woodland percentages in varying increments


def stochastic_exp():


    # how many simulations? 
    num_simulations = 100
    # run number
    run_number = 0

    # keep track of the time-series and final conditions
    all_timeseries = []
    final_df = {}


    # run it to 500 years (equilibrium) 
    for i in range(num_simulations):

        X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]

        # we want to check if our original PS passes the requirements or not
        accepted_parameters = pd.read_csv('ode_best_ps.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

        r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
        r = pd.DataFrame(r.values.reshape(1, len(species)), columns = species).to_numpy().flatten()

        A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()


        # is there a critical amount of noise?
        noise_amount = random.choice([0, 0.001, 0.002, 0.003])


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
            starting_values[0] = 0.65
            starting_values[1] = 5.9
            starting_values[3] = 1.5
            starting_values[4] = 2.7
            starting_values[6] = 0.55

            # apply the perturbations; don't let it go below 0
            if years_covered >= 516: 
                # increase the amount of noise over time
                sd += noise_amount
                # noise term is SD and at every point, choose a random normal value that has mean of 0 and that SD.
                noise = np.random.normal(1, sd, 1).item()
                # change the repro rate              
                r[2] = 0.91 * noise
                r[7] = 0.35 * noise
                r[8] = 0.1 * noise
            else: 
                r[2] = 0.91
                r[7] = 0.35
                r[8] = 0.1
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
        noise_reshape = np.repeat(all_noise, 3)
        # put into df
        y_future['time'] = combined_future_times
        y_future["run_number"] = run_number
        y_future["Noise"] = noise_amount

        # get 2005-2020 time-series
        combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # reshape the outputs
        y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
        y_2 = pd.DataFrame(data=y_2, columns=species)
        y_2['time'] = combined_times
        y_2["run_number"] = run_number
        y_2["Noise"] = 0
        # concat all time-series
        timeseries = pd.concat([y_2, y_future])
        run_number += 1
        print(run_number)

        end = timeseries.iloc[-1]

        # then append outputs
        final_df[run_number] = {"Grassland Initial": 89.9, "Grassland End": end[2]*89.9, 
                                "Woodland Initial": 5.8, "Woodland End":  end[8]*5.8, 
                                "Thorny Scrub Initial": 4.3, "Thorny Scrub End":  end[7]*4.3, 
                                "Noise": all_noise,
                                "Run Number": run_number
        }

        all_timeseries.append(timeseries)


    forecasting = pd.DataFrame.from_dict(final_df, "index")
    print(forecasting)
    all_timeseries = pd.concat(all_timeseries)
    print(all_timeseries)

    # save to csv
    forecasting.to_csv("eem_stochastic_exp_landscape.csv")
    all_timeseries.to_csv("eem_stochastic_exp_timeseries.csv")



stochastic_exp()




def tipping_point_veg():


    # put it in middle location

    num_simulations = 90
    # run number
    run_number = 0

    # keep track of the time-series and final conditions
    all_timeseries = []
    final_df = {}

    # run it to 500 years (equilibrium) 
    for i in range(num_simulations):

        grass = random.uniform(0, 0.08)
        # grass = random.uniform(0,0.00000001)
        roe = random.uniform(2.4, 3.2)
        thorny = random.uniform(6.1, 6.7)
        wood = random.uniform(9.4, 10.1)

        # we want to check if our original PS passes the requirements or not
        accepted_parameters = pd.read_csv('ode_best_ps.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

        r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
        r = pd.DataFrame(r.values.reshape(1, len(species)), columns = species).to_numpy().flatten()

        A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()


        # # # # First, run it from 2005-2020 # # # #

        years_covered = 500
        combined_future_runs = []
        combined_future_times = []
        
        X0 = [0.65, 5.9, grass, 1.5, 2.7, roe, 0.55, thorny, wood]

        for y in range(1000): 
            years_covered += 1
            # run the model for that year 

            X0[0] = 0.65
            X0[1] = 5.9
            X0[3] = 1.5
            X0[4] = 2.7
            X0[6] = 0.55

            # keep track of time
            t_steady = np.linspace(years_covered, years_covered+0.75, 3)
            # run model
            ABC_future = solve_ivp(ecoNetwork, (years_covered, years_covered+0.75), X0,  t_eval = t_steady, args=(A, r), method = 'RK23')   
            # append last values
            X0 = ABC_future.y[0:11, 2:3].flatten()

            # and save the outputs
            combined_future_runs.append(ABC_future.y)
            combined_future_times.append(ABC_future.t)

        combined_future = np.hstack(combined_future_runs)
        combined_future_times = np.hstack(combined_future_times)
        timeseries = (np.vstack(np.hsplit(combined_future.reshape(len(species), 3000).transpose(),1)))
        timeseries = pd.DataFrame(data=timeseries, columns=species)
        timeseries['time'] = combined_future_times
        timeseries["run_number"] = run_number
        timeseries["New Equilibrium"] = "No"

        # concat all time-series
        run_number += 1
        print(run_number)


        all_timeseries.append(timeseries)


    all_timeseries = pd.concat(all_timeseries)


    # save to csv
    all_timeseries.to_csv("eem_growth_exp_tipping.csv")


# tipping_point_veg()