from model import KneppModel 
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# we want to choose the best parameter set and see how many filters the 100 GA outputs passed

def run_model():


    # open the parameters
    final_parameters = pd.read_excel("optimizer_outputs_resilience.xlsx",  engine="openpyxl")
    final_parameters = final_parameters.loc[final_parameters["run_number"] <= 100]


    final_results_list = []
    failed_roe = 0
    failed_equilibrium = 0

    for index, row in final_parameters.iterrows():

        print(row["run_number"])

        chance_reproduceSapling = row["chance_reproduceSapling"]
        chance_reproduceYoungScrub =  row["chance_reproduceYoungScrub"]
        chance_regrowGrass =  row["chance_regrowGrass"]
        chance_saplingBecomingTree =  row["chance_saplingBecomingTree"]
        chance_youngScrubMatures =  row["chance_youngScrubMatures"]
        chance_scrubOutcompetedByTree =  row["chance_scrubOutcompetedByTree"]
        chance_grassOutcompetedByTree =  row["chance_grassOutcompetedByTree"]
        chance_grassOutcompetedByScrub = row["chance_grassOutcompetedByScrub"]

        initial_roe = 12
        fallowDeer_stocking = 247
        cattle_stocking = 81
        redDeer_stocking = 35
        tamworthPig_stocking = 7
        exmoor_stocking = 15
        initial_wood = 0.058
        initial_grass = 0.899
        initial_scrub = 0.043

        roe_deer_reproduce = row["roe_deer_reproduce"]
        roe_deer_gain_from_grass =  row["roe_deer_gain_from_grass"]
        roe_deer_gain_from_trees =  row["roe_deer_gain_from_trees"]
        roe_deer_gain_from_scrub =  row["roe_deer_gain_from_scrub"]
        roe_deer_gain_from_saplings =  row["roe_deer_gain_from_saplings"]
        roe_deer_gain_from_young_scrub =  row["roe_deer_gain_from_young_scrub"]
        ponies_gain_from_grass =  row["ponies_gain_from_grass"]
        ponies_gain_from_trees =  row["ponies_gain_from_trees"]
        ponies_gain_from_scrub =  row["ponies_gain_from_scrub"]
        ponies_gain_from_saplings =  row["ponies_gain_from_saplings"]
        ponies_gain_from_young_scrub =  row["ponies_gain_from_young_scrub"]
        cattle_reproduce =  row["cattle_reproduce"]
        cows_gain_from_grass =  row["cows_gain_from_grass"]
        cows_gain_from_trees =  row["cows_gain_from_trees"]
        cows_gain_from_scrub =  row["cows_gain_from_scrub"]
        cows_gain_from_saplings =  row["cows_gain_from_saplings"]
        cows_gain_from_young_scrub =  row["cows_gain_from_young_scrub"]
        fallow_deer_reproduce =  row["fallow_deer_reproduce"]
        fallow_deer_gain_from_grass =  row["fallow_deer_gain_from_grass"]
        fallow_deer_gain_from_trees =  row["fallow_deer_gain_from_trees"]
        fallow_deer_gain_from_scrub =  row["fallow_deer_gain_from_scrub"]
        fallow_deer_gain_from_saplings =  row["fallow_deer_gain_from_saplings"]
        fallow_deer_gain_from_young_scrub =  row["fallow_deer_gain_from_young_scrub"]   
        red_deer_reproduce =  row["red_deer_reproduce"]
        red_deer_gain_from_grass =  row["red_deer_gain_from_grass"]
        red_deer_gain_from_trees =  row["red_deer_gain_from_trees"]
        red_deer_gain_from_scrub =  row["red_deer_gain_from_scrub"]
        red_deer_gain_from_saplings =  row["red_deer_gain_from_saplings"]
        red_deer_gain_from_young_scrub =  row["red_deer_gain_from_young_scrub"]
        tamworth_pig_reproduce =  row["tamworth_pig_reproduce"]
        tamworth_pig_gain_from_grass =  row["tamworth_pig_gain_from_grass"]
        tamworth_pig_gain_from_trees = row["tamworth_pig_gain_from_trees"]
        tamworth_pig_gain_from_scrub = row["tamworth_pig_gain_from_scrub"]
        tamworth_pig_gain_from_saplings =  row["tamworth_pig_gain_from_saplings"]
        tamworth_pig_gain_from_young_scrub =  row["tamworth_pig_gain_from_young_scrub"]

        # euro bison parameters
        european_bison_reproduce = 0
        # bison should have higher impact than any other consumer
        european_bison_gain_from_grass =  0
        european_bison_gain_from_trees =0
        european_bison_gain_from_scrub =0
        european_bison_gain_from_saplings = 0
        european_bison_gain_from_young_scrub = 0  
        # euro elk parameters
        european_elk_reproduce = 0
        # bison should have higher impact than any other consumer
        european_elk_gain_from_grass =  0
        european_elk_gain_from_trees = 0
        european_elk_gain_from_scrub = 0
        european_elk_gain_from_saplings =  0
        european_elk_gain_from_young_scrub =  0
        # reindeer parameters
        reindeer_reproduce = 0
        # reindeer should have impacts between red and fallow deer
        reindeer_gain_from_grass = 0
        reindeer_gain_from_trees =0
        reindeer_gain_from_scrub =0
        reindeer_gain_from_saplings = 0
        reindeer_gain_from_young_scrub = 0
        # forecasting parameters
        fallowDeer_stocking_forecast = 247
        cattle_stocking_forecast = 81
        redDeer_stocking_forecast = 35
        tamworthPig_stocking_forecast = 7
        exmoor_stocking_forecast = 15
        introduced_species_stocking_forecast = 0
        chance_scrub_saves_saplings = row["chance_scrub_saves_saplings"]

        # new resilience experiment parameters - not relevant here
        exp_chance_reproduceSapling = 0
        exp_chance_reproduceYoungScrub =  0
        exp_chance_regrowGrass = 0
        duration = 0
        tree_reduction = 0


        random.seed(1)
        np.random.seed(1)


        model = KneppModel(initial_roe, roe_deer_reproduce, roe_deer_gain_from_saplings, roe_deer_gain_from_trees, roe_deer_gain_from_scrub, roe_deer_gain_from_young_scrub, roe_deer_gain_from_grass,
                            chance_youngScrubMatures, chance_saplingBecomingTree, chance_reproduceSapling,chance_reproduceYoungScrub, chance_regrowGrass, 
                            chance_grassOutcompetedByTree, chance_grassOutcompetedByScrub, chance_scrubOutcompetedByTree, 
                            ponies_gain_from_saplings, ponies_gain_from_trees, ponies_gain_from_scrub, ponies_gain_from_young_scrub, ponies_gain_from_grass, 
                            cattle_reproduce, cows_gain_from_grass, cows_gain_from_trees, cows_gain_from_scrub, cows_gain_from_saplings, cows_gain_from_young_scrub, 
                            fallow_deer_reproduce, fallow_deer_gain_from_saplings, fallow_deer_gain_from_trees, fallow_deer_gain_from_scrub, fallow_deer_gain_from_young_scrub, fallow_deer_gain_from_grass,
                            red_deer_reproduce, red_deer_gain_from_saplings, red_deer_gain_from_trees, red_deer_gain_from_scrub, red_deer_gain_from_young_scrub, red_deer_gain_from_grass,
                            tamworth_pig_reproduce, tamworth_pig_gain_from_saplings,tamworth_pig_gain_from_trees,tamworth_pig_gain_from_scrub,tamworth_pig_gain_from_young_scrub,tamworth_pig_gain_from_grass,
                            european_bison_reproduce, european_bison_gain_from_grass, european_bison_gain_from_trees, european_bison_gain_from_scrub, european_bison_gain_from_saplings, european_bison_gain_from_young_scrub,
                            european_elk_reproduce, european_elk_gain_from_grass, european_elk_gain_from_trees, european_elk_gain_from_scrub, european_elk_gain_from_saplings, european_elk_gain_from_young_scrub,
                            fallowDeer_stocking_forecast, cattle_stocking_forecast, redDeer_stocking_forecast, tamworthPig_stocking_forecast, exmoor_stocking_forecast,
                            chance_scrub_saves_saplings, initial_wood, initial_grass, initial_scrub,
                            exp_chance_reproduceSapling, exp_chance_reproduceYoungScrub, exp_chance_regrowGrass, duration, tree_reduction,
                            reintroduction = True, introduce_euroBison = False, introduce_elk = False, 
                            experiment_growth = False, experiment_wood = False, experiment_linear_growth = False)


        model.reset_randomizer(seed=1)

        model.run_model()


        # remember the results of the model (dominant conditions, # of agents)
        results = model.datacollector.get_model_vars_dataframe()
        results['run_number'] = row["run_number"]


        # get the last year
        end = results.iloc[-1]
        # did it pass the roe non-explosion/collapse requirements? 
        if end["Time"] < 5999:
            failed_roe += 1
            print(results)
            print("failed roe:", (failed_roe/row["run_number"])*100,"%")
        

        else:
            # calculate gradient - has it reached equilibrium?
            last_hundred = results.loc[results["Time"] >= 4800]
            # only species I care about - ones that aren't controlled
            uncontrolled_only = last_hundred[['Grassland', 'Woodland', 'Thorny Scrub', 'Roe deer']]
            
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
            
            # if it's at equilibrium, save it
            if round(abs(gradients["Roe deer"])) <= 0.01 and round(abs(gradients["Thorny Scrub"])) <= 0.01 and round(abs(gradients["Grassland"])) <= 0.01 and round(abs(gradients["Woodland"])) <= 0.01:
                # save it
                final_results_list.append(results)
            else: 
                print(abs(gradients))
                print(round(abs(gradients)))
                failed_equilibrium += 1
                print("failed equilibrium::", (failed_equilibrium/row["run_number"])*100,"%")



    # save those that failed ecological criteria
    with open('failed_ecological_allGAOutputs.txt', 'w') as f:
        lines = ["failed roe deer:", str((failed_roe/row["run_number"])*100),'%', "failed equilibrium:", str((failed_equilibrium/row["run_number"])*100),"%"]
        f.writelines(lines)


    # append accepted ones to dataframe
    final_results = pd.concat(final_results_list)

    # which filters were the most difficult to pass?
    difficult_filters = pd.DataFrame({
    'filter_number': np.arange(0,63),
    'times_passed': np.zeros(63)})

    # filter the runs and tag the dataframe; keep track of how many filters passed
    final_results["passed_filters"] = 0
    # pre-reintroduction model
    my_time = final_results.loc[final_results['Time'] == 49]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Roe deer"] <= 40) & (row["Roe deer"] >= 12) & (row["Grassland"] <= 80) & (row["Grassland"] >= 20) & (row["Woodland"] <= 17) & (row["Woodland"] >= 5.8) & (row["Thorny Scrub"] <= 51.8) & (row["Thorny Scrub"] >= 4.3):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[0,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    
    # April 2015
    my_time = final_results.loc[final_results['Time'] == 122]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) & (row["Longhorn cattle"] <= 140) & (row["Longhorn cattle"] >= 90) & (row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[1,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # May 2015
    my_time = final_results.loc[final_results['Time'] == 123]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 154) & (row["Longhorn cattle"] >= 104) & (row["Tamworth pigs"] <= 24) & (row["Tamworth pigs"] >= 4) & (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[2,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # June 2015
    my_time = final_results.loc[final_results['Time'] == 124]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 154) & (row["Longhorn cattle"] >= 104) & (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) & (row["Tamworth pigs"] <= 24) & (row["Tamworth pigs"] >= 4):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[3,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # July 2015
    my_time = final_results.loc[final_results['Time'] == 125]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 154) & (row["Longhorn cattle"] >= 104) & (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) & (row["Tamworth pigs"] <= 24) & (row["Tamworth pigs"] >= 4):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[4,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Aug 2015
    my_time = final_results.loc[final_results['Time'] == 126]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 154) & (row["Longhorn cattle"] >= 104) & (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) & (row["Tamworth pigs"] <= 24) & (row["Tamworth pigs"] >= 4):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[5,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Sept 2015
    my_time = final_results.loc[final_results['Time'] == 127]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 155) & (row["Longhorn cattle"] >= 105) & (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) & (row["Tamworth pigs"] <= 24) & (row["Tamworth pigs"] >= 4):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[6,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Oct 2015
    my_time = final_results.loc[final_results['Time'] == 128]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 116) & (row["Longhorn cattle"] >= 66) & (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) & (row["Tamworth pigs"] <= 24) & (row["Tamworth pigs"] >= 4):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[7,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Nov 2015
    my_time = final_results.loc[final_results['Time'] == 129]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 116) & (row["Longhorn cattle"] >= 66) & (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) & (row["Tamworth pigs"] <= 23) & (row["Tamworth pigs"] >= 3):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[8,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Dec 2015
    my_time = final_results.loc[final_results['Time'] == 130]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 111) & (row["Longhorn cattle"] >= 61) &(row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Tamworth pigs"] <= 23) & (row["Tamworth pigs"] >= 3):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[9,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Jan 2016
    my_time = final_results.loc[final_results['Time'] == 131]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 111) & (row["Longhorn cattle"] >= 61) & (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) & (row["Tamworth pigs"] <= 20) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[10,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Feb 2016
    my_time = final_results.loc[final_results['Time'] == 132]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) & (row["Longhorn cattle"] <= 111) & (row["Longhorn cattle"] >= 61) &(row["Tamworth pigs"] <= 20) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[11,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # March 2016
    my_time = final_results.loc[final_results['Time'] == 133]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) & (row["Longhorn cattle"] <= 111) & (row["Longhorn cattle"] >= 61) & (row["Fallow deer"] <= 190) & (row["Fallow deer"] >= 90) & (row["Red deer"] <= 31) & (row["Red deer"] >= 21) & (row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[12,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # April 2016
    my_time = final_results.loc[final_results['Time'] == 134]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) & (row["Longhorn cattle"] <= 128) & (row["Longhorn cattle"] >= 78) & (row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[13,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # May 2016
    my_time = final_results.loc[final_results['Time'] == 135]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) & (row["Longhorn cattle"] <= 133) & (row["Longhorn cattle"] >= 83) &(row["Tamworth pigs"] <= 27) & (row["Tamworth pigs"] >= 7):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[14,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # June 2016
    my_time = final_results.loc[final_results['Time'] == 136]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) & (row["Longhorn cattle"] <= 114) & (row["Longhorn cattle"] >= 64) & (row["Tamworth pigs"] <= 27) & (row["Tamworth pigs"] >= 7):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[15,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # July 2016
    my_time = final_results.loc[final_results['Time'] == 137]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) & (row["Longhorn cattle"] <= 112) & (row["Longhorn cattle"] >= 62) & (row["Tamworth pigs"] <= 27) & (row["Tamworth pigs"] >= 7):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[16,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Aug 2016
    my_time = final_results.loc[final_results['Time'] == 138]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) & (row["Longhorn cattle"] <= 112) & (row["Longhorn cattle"] >= 62) & (row["Tamworth pigs"] <= 27) & (row["Tamworth pigs"] >= 7):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[17,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Sept 2016
    my_time = final_results.loc[final_results['Time'] == 139]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) &(row["Longhorn cattle"] <= 122) & (row["Longhorn cattle"] >= 72) & (row["Tamworth pigs"] <= 27) & (row["Tamworth pigs"] >= 7):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[18,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Oct 2016
    my_time = final_results.loc[final_results['Time'] == 140]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) & (row["Longhorn cattle"] <= 122) & (row["Longhorn cattle"] >= 72) & (row["Tamworth pigs"] <= 27) & (row["Tamworth pigs"] >= 7):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[19,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Nov 2016
    my_time = final_results.loc[final_results['Time'] == 141]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) &(row["Longhorn cattle"] <= 117) & (row["Longhorn cattle"] >= 67) &(row["Tamworth pigs"] <= 27) & (row["Tamworth pigs"] >= 7):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[20,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Dec 2016
    my_time = final_results.loc[final_results['Time'] == 142]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) &(row["Longhorn cattle"] <= 104) & (row["Longhorn cattle"] >= 54)& (row["Tamworth pigs"] <= 23) & (row["Tamworth pigs"] >= 3):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[21,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Jan 2017
    my_time = final_results.loc[final_results['Time'] == 143]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) & (row["Longhorn cattle"] <= 104) & (row["Longhorn cattle"] >= 54) & (row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[22,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Feb 2017
    my_time = final_results.loc[final_results['Time'] == 144]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 12) & (row["Exmoor pony"] >= 10) & (row["Longhorn cattle"] <= 104) & (row["Longhorn cattle"] >= 54) & (row["Tamworth pigs"] <= 17) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[23,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 


    # March 2017
    my_time = final_results.loc[final_results['Time'] == 145]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) & (row["Fallow deer"] <= 215) & (row["Fallow deer"] >= 115) &(row["Longhorn cattle"] <= 104) & (row["Longhorn cattle"] >= 54) &(row["Tamworth pigs"] <= 17) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[24,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # April 2017
    my_time = final_results.loc[final_results['Time'] == 146]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 125) & (row["Longhorn cattle"] >= 75) &(row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[25,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # May 2017
    my_time = final_results.loc[final_results['Time'] == 147]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if  (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 134) & (row["Longhorn cattle"] >= 84) &(row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[26,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # June 2017
    my_time = final_results.loc[final_results['Time'] == 148]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 119) & (row["Longhorn cattle"] >= 69) &(row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[27,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # July 2017
    my_time = final_results.loc[final_results['Time'] == 149]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 119) & (row["Longhorn cattle"] >= 69)&(row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[28,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Aug 2017
    my_time = final_results.loc[final_results['Time'] == 150]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 119) & (row["Longhorn cattle"] >= 69)&(row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[29,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Sept 2017
    my_time = final_results.loc[final_results['Time'] == 151]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 115) & (row["Longhorn cattle"] >= 65)& (row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[30,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Oct 2017
    my_time = final_results.loc[final_results['Time'] == 152]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 113) & (row["Longhorn cattle"] >= 63)&(row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[31,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Nov 2017
    my_time = final_results.loc[final_results['Time'] == 153]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 113) & (row["Longhorn cattle"] >= 63)&(row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[32,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Dec 2017
    my_time = final_results.loc[final_results['Time'] == 154]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 113) & (row["Longhorn cattle"] >= 63)&(row["Tamworth pigs"] <= 28) & (row["Tamworth pigs"] >= 8):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[33,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # January 2018
    my_time = final_results.loc[final_results['Time'] == 155]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 113) & (row["Longhorn cattle"] >= 63)&(row["Tamworth pigs"] <= 21) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[34,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # February 2018
    my_time = final_results.loc[final_results['Time'] == 156]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 11) & (row["Exmoor pony"] >= 9) &(row["Longhorn cattle"] <= 113) & (row["Longhorn cattle"] >= 63)&(row["Tamworth pigs"] <= 26) & (row["Tamworth pigs"] >= 6):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[35,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
   

    # March 2018
    my_time = final_results.loc[final_results['Time'] == 157]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 10) & (row["Exmoor pony"] >= 8) & (row["Longhorn cattle"] <= 113) & (row["Longhorn cattle"] >= 63) &(row["Red deer"] <= 29) & (row["Red deer"] >= 19) &(row["Tamworth pigs"] <= 26) & (row["Tamworth pigs"] >= 6):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[36,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # April 2018
    my_time = final_results.loc[final_results['Time'] == 158]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 10) & (row["Exmoor pony"] >= 8) & (row["Longhorn cattle"] <= 126) & (row["Longhorn cattle"] >= 76) &(row["Tamworth pigs"] <= 26) & (row["Tamworth pigs"] >= 6):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[37,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # May 2018
    my_time = final_results.loc[final_results['Time'] == 159]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 10) & (row["Exmoor pony"] >= 8) &(row["Longhorn cattle"] <= 142) & (row["Longhorn cattle"] >= 92)&(row["Tamworth pigs"] <= 33) & (row["Tamworth pigs"] >= 13):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[38,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # June 2018
    my_time = final_results.loc[final_results['Time'] == 160]
    accepted_runs = []
    for index, row in my_time.iterrows(): 
        if (row["Exmoor pony"] <= 10) & (row["Exmoor pony"] >= 8) & (row["Longhorn cattle"] <= 128) & (row["Longhorn cattle"] >= 78)& (row["Tamworth pigs"] <= 33) & (row["Tamworth pigs"] >= 13):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[39,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # July 2018
    my_time = final_results.loc[final_results['Time'] == 161]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if  (row["Exmoor pony"] <= 10) & (row["Exmoor pony"] >= 8) &(row["Longhorn cattle"] <= 128) & (row["Longhorn cattle"] >= 78)&(row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[40,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Aug 2018
    my_time = final_results.loc[final_results['Time'] == 162]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 127) & (row["Longhorn cattle"] >= 77) &(row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[41,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Sept 2018
    my_time = final_results.loc[final_results['Time'] == 163]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 131) & (row["Longhorn cattle"] >= 81) & (row["Tamworth pigs"] <= 32) & (row["Tamworth pigs"] >= 12):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[42,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Oct 2018
    my_time = final_results.loc[final_results['Time'] == 164]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 126) & (row["Longhorn cattle"] >= 76) & (row["Tamworth pigs"] <= 31) & (row["Tamworth pigs"] >= 11):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[43,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Nov 2018
    my_time = final_results.loc[final_results['Time'] == 165]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 118) & (row["Longhorn cattle"] >= 68) & (row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[44,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Dec 2018
    my_time = final_results.loc[final_results['Time'] == 166]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 114) & (row["Longhorn cattle"] >= 64) & (row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[45,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Jan 2019
    my_time = final_results.loc[final_results['Time'] == 167]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 114) & (row["Longhorn cattle"] >= 64) & (row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[46,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Feb 2019
    my_time = final_results.loc[final_results['Time'] == 168]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 112) & (row["Longhorn cattle"] >= 62) &(row["Tamworth pigs"] <= 20) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[47,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # March 2019
    my_time = final_results.loc[final_results['Time'] == 169]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Fallow deer"] <= 303) & (row["Fallow deer"] >= 253) &(row["Longhorn cattle"] <= 112) & (row["Longhorn cattle"] >= 62) &(row["Red deer"] <= 42) & (row["Red deer"] >= 32) & (row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[48,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # April 2019
    my_time = final_results.loc[final_results['Time'] == 170]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 126) & (row["Longhorn cattle"] >= 76) &(row["Tamworth pigs"] <= 18) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[49,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # May 2019
    my_time = final_results.loc[final_results['Time'] == 171]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 135) & (row["Longhorn cattle"] >= 85) & (row["Tamworth pigs"] <= 18) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[50,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # June 2019
    my_time = final_results.loc[final_results['Time'] == 172]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 114) & (row["Longhorn cattle"] >= 64) & (row["Tamworth pigs"] <= 18) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[51,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # July 2019
    my_time = final_results.loc[final_results['Time'] == 173]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 116) & (row["Longhorn cattle"] >= 66) &(row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[52,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Aug 2019
    my_time = final_results.loc[final_results['Time'] == 174]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 116) & (row["Longhorn cattle"] >= 66) & (row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[53,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Sept 2019
    my_time = final_results.loc[final_results['Time'] == 175]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 118) & (row["Longhorn cattle"] >= 68) & (row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[54,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Oct 2019
    my_time = final_results.loc[final_results['Time'] == 176]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 113) & (row["Longhorn cattle"] >= 63) &(row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[55,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Nov 2019
    my_time = final_results.loc[final_results['Time'] == 177]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 112) & (row["Longhorn cattle"] >= 62) & (row["Tamworth pigs"] <= 19) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[56,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Dec 2019
    my_time = final_results.loc[final_results['Time'] == 178]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 105) & (row["Longhorn cattle"] >= 55) & (row["Tamworth pigs"] <= 20) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[57,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Jan 2020
    my_time = final_results.loc[final_results['Time'] == 179]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 105) & (row["Longhorn cattle"] >= 55) & (row["Tamworth pigs"] <= 20) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[58,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # Feb 2020
    my_time = final_results.loc[final_results['Time'] == 180]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Longhorn cattle"] <= 104) & (row["Longhorn cattle"] >= 54) & (row["Tamworth pigs"] <= 18) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[59,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
  
    # March 2020
    my_time = final_results.loc[final_results['Time'] == 181]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Fallow deer"] <= 272) & (row["Fallow deer"] >= 222) & (row["Red deer"] <= 40) & (row["Red deer"] >= 32) &(row["Longhorn cattle"] <= 106) & (row["Longhorn cattle"] >= 56)&(row["Tamworth pigs"] <= 17) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[60,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # April 2020
    my_time = final_results.loc[final_results['Time'] == 182]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Exmoor pony"] <= 17) & (row["Exmoor pony"] >= 14) &(row["Longhorn cattle"] <= 106) & (row["Longhorn cattle"] >= 56) &(row["Tamworth pigs"] <= 17) & (row["Tamworth pigs"] >= 1):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[61,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 
    # May 2020
    my_time = final_results.loc[final_results['Time'] == 183]
    accepted_runs = []
    for index, row in my_time.iterrows():
        if (row["Tamworth pigs"] <= 29) & (row["Tamworth pigs"] >= 9) &(row["Exmoor pony"] <= 17) & (row["Exmoor pony"] >= 14) &(row["Longhorn cattle"] <= 106) & (row["Longhorn cattle"] >= 56) &(row["Roe deer"] <= 80) & (row["Roe deer"] >= 20) & (row["Grassland"] <= 36.8) & (row["Grassland"] >= 16.8) & (row["Thorny Scrub"] <= 61.8) & (row["Thorny Scrub"] >= 41.8) &(row["Woodland"] <= 31.5) & (row["Woodland"] >= 11.5):
            accepted_runs.append(row["run_number"])
            difficult_filters.loc[62,'times_passed'] += 1
    final_results['passed_filters'] = np.where(final_results['run_number'].isin(accepted_runs),final_results['passed_filters']+1,final_results['passed_filters']) 

    print("Time 49", final_results.loc[final_results["Time"] == 49])
    print("Time 183", final_results.loc[final_results["Time"] == 183])
    # save the results
    final_results.to_csv('abm_final_results_all_GA_outputs.csv')


run_model()