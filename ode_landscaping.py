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
accepted_parameters = pd.read_csv('ode_best_ps.csv').drop(['Unnamed: 0', 'accepted?', 'ID'], axis=1)

r = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
r = pd.DataFrame(r.values.reshape(1, len(species)), columns = species).to_numpy().flatten()

A = accepted_parameters.drop(['X0', 'growth', 'Unnamed: 0.1'], axis=1).dropna().to_numpy()




# generate starting conditions
def generate_rescaled_list(j, i):
    total_sum = sum(j)
    random_numbers = [random.random() for _ in j]
    scaled_numbers = [num / sum(random_numbers) for num in random_numbers]
    rescaled_numbers = [scaled_num * total_sum for scaled_num in scaled_numbers]
    rescaled_i = [num / sum(rescaled_numbers) for num in rescaled_numbers]
    return rescaled_i




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



def landscaping():
    run_number = 0

    final_df = {}
    all_timeseries = []

    # pick a few starting conditions - we want the first run to be initial conditions
    initial_stocking = [1]

    # stocking_changes = np.random.uniform(low=0,high=100,size=999) # this one is for looking at landscaping
    # stocking_changes = np.insert(stocking_changes, 0, 1, axis=0)

    stocking_changes = np.random.uniform(low=1,high=1,size=1000)

    # now do initial vegetation values - we start with everything normalised to 1
    initial_wood_list = [1]
    initial_grass_list = [1]
    initial_scrub_list = [1]
    counter = 0
    # add the rest
    while counter < 1000:
        wood = random.uniform(0, 17.2) # this is equivalent to 100%
        grass = random.uniform(0, 1.1)
        scrub = random.uniform(0, 23.3)
        initial_veg = [(wood*0.058), (grass*0.899), (scrub*0.043)]
        # append to list
        if (wood*0.058) + (grass*0.899) + (scrub*0.043) <= 1:
            counter+=1
            initial_wood_list.append(wood)
            initial_grass_list.append(grass)
            initial_scrub_list.append(scrub)


    # calculate the steady state (where X0 = 2020 conditions)
    for i, w, g, s in zip(stocking_changes, initial_wood_list, initial_grass_list, initial_scrub_list):

        # make the starting conditions based on this
        X0_init = [0, 0, g, 0, 0, 1, 0, s, w]

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
        t_2020 = np.linspace(15, 15.75, 3)
        ABC_2020 = solve_ivp(ecoNetwork, (15,15.75), starting_values_2020,  t_eval = t_2020, args=(A, r), method = 'RK23')     
        # get those last values
        last_values_2020 = ABC_2020.y[0:11, 2:3].flatten()

        # gather final conditions and run to equilbrium
        X0 = [0.65, 5.9, last_values_2020[2], 1.5, 2.7, last_values_2020[5], 0.55, last_values_2020[7], last_values_2020[8]]

        # now run it for 500 years
        years_covered = 16
        combined_future_runs = []
        combined_future_times = []
        
        for y in range(500): 
            years_covered += 1
            # run the model for that year 
            starting_values = last_values_2020.copy()
            starting_values[0] = 0.65 * i
            starting_values[1] = 5.9 * i
            starting_values[3] = 1.5 * i
            starting_values[4] = 2.7 * i
            starting_values[6] = 0.55 * i
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
        run_number += 1
        print(run_number)

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
                # then append outputs
                final_df[run_number] = {"Grassland Initial": X0_init[2]*89.9, "Grassland End": end[2]*89.9, 
                                        "Woodland Initial": X0_init[8]*5.8, "Woodland End":  end[8]*5.8, 
                                        "Thorny Scrub Initial": X0_init[7]*4.3, "Thorny Scrub End":  end[7]*4.3, 
                                        "Stocking Density": i,
                                        "Run Number": run_number
                }

                all_timeseries.append(timeseries)


    forecasting = pd.DataFrame.from_dict(final_df, "index")
    all_timeseries = pd.concat(all_timeseries)

    # save to csv
    forecasting.to_csv("resilience_landscape_eem_current.csv")
    all_timeseries.to_csv("timeseries_eem_current.csv")




landscaping()





# run it to steady state (500 years) with current conditions
# then change stocking densities by 0-0.1x, where is tipping point?
# then change again, where is tipping point? 


def tipping_points():
    run_number = 0

    final_df = {}
    all_timeseries = []

    # pick a few starting conditions - we want the first run to be initial conditions
    stocking_changes = np.random.uniform(low=0.95,high=1.05,size=99) # this one is for looking at tipping points (smaller view)
    stocking_changes = np.insert(stocking_changes, 0, 1, axis=0)

    # calculate the steady state (where X0 = 2020 conditions) - initial condition doesn't matter
    for i in stocking_changes:

        # make the starting conditions based on this
        X0_init = [0, 0, 1, 0, 0, 1, 0, 1, 1]

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
        t_2020 = np.linspace(15, 15.75, 3)
        ABC_2020 = solve_ivp(ecoNetwork, (15,15.75), starting_values_2020,  t_eval = t_2020, args=(A, r), method = 'RK23')     
        # get those last values
        last_values_2020 = ABC_2020.y[0:11, 2:3].flatten()

        # gather final conditions and run to equilbrium
        X0 = [0.65, 5.9, last_values_2020[2], 1.5, 2.7, last_values_2020[5], 0.55, last_values_2020[7], last_values_2020[8]]

        # now run it for 500 years
        years_covered = 16
        combined_future_runs = []
        combined_future_times = []
        
        for y in range(1000): 
            years_covered += 1
            # run the model for that year 
            starting_values = last_values_2020.copy()

            if years_covered >= 516: 
                starting_values[0] = 0.65 * i
                starting_values[1] = 5.9 * i
                starting_values[3] = 1.5 * i
                starting_values[4] = 2.7 * i
                starting_values[6] = 0.55 * i
            else:
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
        y_future["stocking"] = i


        # get 2005-2020 time-series
        combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # reshape the outputs
        y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
        y_2 = pd.DataFrame(data=y_2, columns=species)
        y_2['time'] = combined_times
        y_2["run_number"] = run_number
        y_2["stocking"] = i

        # concat all time-series
        timeseries = pd.concat([y_2, y_future])
        run_number += 1
        print(run_number)

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
                # then append outputs
                final_df[run_number] = {"Grassland Initial": 89.9, "Grassland End": end[2]*89.9, 
                                        "Woodland Initial": 5.8, "Woodland End":  end[8]*5.8, 
                                        "Thorny Scrub Initial": 4.3, "Thorny Scrub End":  end[7]*4.3, 
                                        "Stocking Density": i,
                                        "Run Number": run_number
                }

                all_timeseries.append(timeseries)


    forecasting = pd.DataFrame.from_dict(final_df, "index")
    all_timeseries = pd.concat(all_timeseries)


    forecasting.to_csv("resilience_tippingpoints_eem.csv")
    all_timeseries.to_csv("timeseries_tippingpoints_eem.csv")


# tipping_points()