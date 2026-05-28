import os
import sys
import math
import pickle
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

import gurobipy as gb
from gurobipy import GRB
from gurobipy import LinExpr


MIPGAP = 0.01 # can change up to 1.5 (0.1, 1.0, 1.5 with increment of 0.1)
ARC_PER_TS = 30
S = 100  
totalInfeasibleScenarios = 0

# Hybrid MILP Assistance Flags
ASSIST_MILP_WITH_ONOFF = True
ASSIST_MILP_WITH_ARC = True
ASSIST_MILP_WITH_ESCONOFF = True
ASSIST_MILP_WITH_ESDONOFF = True
ASSIST_MILP_WITH_CC = True

# Import to CSV flags
IMPORT_TO_CSV_ONOFF = True
IMPORT_TO_CSV_ARC = True
IMPORT_TO_CSV_ESCONOFF = True
IMPORT_TO_CSV_ESDONOFF = True
IMPORT_TO_CSV_CC = True
IMPORT_TO_CSV_OPTIMISED_COST_AND_SOLUTION_TIME = True

INVOKE_CONFLICT_REFINER = 0
confFile = "iloconflictex1.lp"

EXPORT_TO_LP = 1

lpFilePath = r"D:\Savini\Wen Codes\savini_results\Optimum schedule\TESS_CC.lp"
rootPath = r"D:\Savini\Wen Codes\savini_new_results\CNN_results\optimum schedule"
aiPredictedPath = r"D:\Savini\Wen Codes\savini_new_results\CNN_results"

# Determine if using warm start or hybrid or pure MILP
# val = -1		(pure MILP)
# val = 200		HA-1 (hybrid: problem size reduction for FEASIBLE AI prediction, WARM start for INFEASIBLE ones)
# val = 300		HA-2 (hybrid: problem size reduction for FEASIBLE AI prediction, COLD start for INFEASIBLE ones)
# val = {0, 100} (val% warm start, (100-val)% problem size reduction)

# Export path of MILP output
milpOptimumOnOffFileName = rf"{rootPath}\MILP_Optimum_OnOff.csv"  # REDUNDANT if IMPORT_TO_CSV_ONOFF = 0
milpOptimumESCOnOffFileName = rf"{rootPath}\MILP_Optimum_ESCOnOff.csv"  # REDUNDANT if IMPORT_TO_CSV_ESCONOFF = 0
milpOptimumESDOnOffFileName = rf"{rootPath}\MILP_Optimum_ESDOnOff.csv"  # REDUNDANT if IMPORT_TO_CSV_ESDONOFF = 0
milpOptimumArcFileName = rf"{rootPath}\MILP_Optimum_Arc.csv"  # REDUNDANT if IMPORT_TO_CSV_ARC = 0
milpOptimumCCFileName = rf"{rootPath}\MILP_Optimum_CC.csv"  # REDUNDANT if IMPORT_TO_CSV_CC = 0


predictedOnOffFileName = rf"{aiPredictedPath}\predicted\Predicted_Generator_Schedules.csv"
predictedArcFileName = rf"{aiPredictedPath}\predicted\Predicted_Arcs.csv"
predictedESCOnOffFileName = rf"{aiPredictedPath}\predicted\Predicted_Storage_Charge.csv"
predictedESDOnOffFileName = rf"{aiPredictedPath}\predicted\Predicted_Storage_Discharge.csv"
predictedCCFileName = rf"{aiPredictedPath}\predicted\Predicted_ChanceConstraints.csv"

predictedconfidenceFileName = rf"{aiPredictedPath}\predicted\Confidence_Generation.csv"
predictedNBESFileName = rf"{aiPredictedPath}\predicted\nBES_Generation.csv"
solarfile = rf"{aiPredictedPath}\predicted\Solar_Generation.csv"
windfile = rf"{aiPredictedPath}\predicted\Wind_Generation.csv"

predictedOnOffFeasibilityIndexFileName = rf"{aiPredictedPath}\feasibility\Predicted_Generator_Schedules_Feasibility_Index.csv"
predictedArcFeasibilityIndexFileName = rf"{aiPredictedPath}\feasibility\Predicted_Arcs_TS_Feasibility_Index.csv"
predictedESCOnOffFeasibilityIndexFileName = rf"{aiPredictedPath}\feasibility\Predicted_Storage_Charge_Feasibility_Index.csv"
predictedESDOnOffFeasibilityIndexFileName = rf"{aiPredictedPath}\feasibility\Predicted_Storage_Discharge_Feasibility_Index.csv"
predictedCCFeasibilityIndexFileName = rf"{aiPredictedPath}\feasibility\Predicted_ChanceConstraints_Feasibility_Index.csv"

out_OPTIMISED_COST_AND_SOLUTION_TIME = rf"{rootPath}\Results.csv"
# data_dir = r"D:\Savini\Wen Codes\data"

# Optimization results storage
# MAX_ITER = 1
# OptimisedCost = [0] * MAX_ITER
# SolutionTime = [0] * MAX_ITER

# Create Gurobi Model
# gurobi_env = gb.Env()
# gurobi_env.setParam("OutputFlag", 0)
# now = datetime.now()
# print("Current time:", now.strftime("%H:%M:%S"))


# Load CSV function
def load_csv(filename, rows, cols):
    df = pd.read_csv(filename, header=None)
    return df.iloc[:rows, :cols].values.tolist()

def load_and_filter_ai_data(target_conf, target_nBES, nbGen, nbHour):
    global totalInfeasibleScenarios

    confidence_data = pd.read_csv(predictedconfidenceFileName, header=None).values
    confidences = confidence_data[:, 0]
    confidence_idx = np.where(np.isclose(confidences, target_conf))[0]

    nbes_data = pd.read_csv(predictedNBESFileName, header=None).values
    active_nbes_data = nbes_data[confidence_idx]

    # important: map back to original absolute row indices
    match_mask = np.sum(active_nbes_data, axis=1) == target_nBES
    final_idx = confidence_idx[match_mask]

    predictedOnOffFeasibilityIndex = pd.read_csv(predictedOnOffFeasibilityIndexFileName, header=None).values
    predictedOnOffFeasibilityIndex = predictedOnOffFeasibilityIndex[final_idx, :].reshape(len(final_idx), nbGen)

    predictedArcFeasibilityIndex = pd.read_csv(predictedArcFeasibilityIndexFileName, header=None).values
    # predictedArcFeasibilityIndex = predictedArcFeasibilityIndex[final_idx, :].reshape(len(final_idx), nbHour - 1)
    predictedArcFeasibilityIndex = predictedArcFeasibilityIndex[final_idx, :target_nBES * (nbHour)]
    predictedArcFeasibilityIndex = predictedArcFeasibilityIndex.reshape(len(final_idx), target_nBES, nbHour)

    predictedESCOnOffFeasibilityIndex = pd.read_csv(predictedESCOnOffFeasibilityIndexFileName, header=None).values
    # predictedESCOnOffFeasibilityIndex = predictedESCOnOffFeasibilityIndex[final_idx, :].reshape(len(final_idx), nbHour - 1)
    predictedESCOnOffFeasibilityIndex = predictedESCOnOffFeasibilityIndex[final_idx, :target_nBES * (nbHour)]
    predictedESCOnOffFeasibilityIndex = predictedESCOnOffFeasibilityIndex.reshape(len(final_idx), target_nBES, nbHour)

    predictedESDOnOffFeasibilityIndex = pd.read_csv(predictedESDOnOffFeasibilityIndexFileName, header=None).values
    # predictedESDOnOffFeasibilityIndex = predictedESDOnOffFeasibilityIndex[final_idx, :].reshape(len(final_idx), nbHour - 1)
    predictedESDOnOffFeasibilityIndex = predictedESDOnOffFeasibilityIndex[final_idx, :target_nBES * (nbHour)]
    predictedESDOnOffFeasibilityIndex = predictedESDOnOffFeasibilityIndex.reshape(len(final_idx), target_nBES, nbHour)

    predictedCCFeasibilityIndex = pd.read_csv(predictedCCFeasibilityIndexFileName, header=None).values
    predictedCCFeasibilityIndex = predictedCCFeasibilityIndex[final_idx].reshape(len(final_idx))

    predictedOnOffData = pd.read_csv(predictedOnOffFileName, header=None).values
    filtered_data_gen = predictedOnOffData[final_idx, :].reshape(len(final_idx), 54, 24)

    predictedArcData = pd.read_csv(predictedArcFileName, header=None).values
    predictedArcData = predictedArcData[final_idx, :target_nBES * 30 * (nbHour)]
    filtered_data_arc = predictedArcData.reshape(len(final_idx), target_nBES, nbHour, 30)

    predictedESCData = pd.read_csv(predictedESCOnOffFileName, header=None).values
    predictedESCData = predictedESCData[final_idx, :target_nBES * (nbHour)]
    filtered_data_esc = predictedESCData.reshape(len(final_idx), target_nBES, nbHour)
    # filtered_data_esc = predictedESCData[final_idx, :].reshape(len(final_idx), target_nBES, 23)

    predictedESDData = pd.read_csv(predictedESDOnOffFileName, header=None).values
    predictedESDData = predictedESDData[final_idx, :target_nBES * (nbHour)]
    filtered_data_esd = predictedESDData.reshape(len(final_idx), target_nBES, nbHour)
    # filtered_data_esd = predictedESDData[final_idx, :].reshape(len(final_idx), target_nBES, 23)

    predictedCCData = pd.read_csv(predictedCCFileName, header=None).values
    filtered_data_cc = predictedCCData[final_idx, :].reshape(len(final_idx), 100)

    # feature files saved as: total_rows = num_samples * 100, each row has 24 hours
    solar = pd.read_csv(solarfile, header=None).values
    num_samples = solar.shape[0] // 100
    solar = solar.reshape(num_samples, 100, 24)
    filtered_solar = solar[final_idx, :, :]   # (nmatch, 100, 24)

    wind = pd.read_csv(windfile, header=None).values
    num_samples_wind = wind.shape[0] // 100
    wind = wind.reshape(num_samples_wind, 100, 24)
    filtered_wind = wind[final_idx, :, :]     # (nmatch, 100, 24)

    return {
        "final_idx": final_idx,
        "predictedOnOffFeasibilityIndex": predictedOnOffFeasibilityIndex,
        "predictedArcFeasibilityIndex": predictedArcFeasibilityIndex,
        "predictedESCOnOffFeasibilityIndex": predictedESCOnOffFeasibilityIndex,
        "predictedESDOnOffFeasibilityIndex": predictedESDOnOffFeasibilityIndex,
        "predictedCCFeasibilityIndex": predictedCCFeasibilityIndex,
        "filtered_data_gen": filtered_data_gen,
        "filtered_data_arc": filtered_data_arc,
        "filtered_data_esc": filtered_data_esc,
        "filtered_data_esd": filtered_data_esd,
        "filtered_data_cc": filtered_data_cc,
        "filtered_solar": filtered_solar,
        "filtered_wind": filtered_wind,
    }


def build_and_solve_one_case(final_idx, scenario_idx, p, ai, target_conf, target_nBES):
    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== initial setup =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------
    gurobi_env = gb.Env()
    gurobi_env.setParam("OutputFlag", 0)
    global totalInfeasibleScenarios
    # Initialise model
    model = gb.Model("TESS_UC", env=gurobi_env)
    now = datetime.now()
    print("Current time:", now.strftime("%H:%M:%S"))

    data_dir = r"D:\Savini\Wen Codes\data"

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== generator data =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------

    data_file = os.path.join(data_dir, "data118test.csv")
    gen_data = pd.read_csv(data_file, header = None)

    # extracting generator parameters from the csv file
    MaxCapacity = np.nan_to_num(gen_data.iloc[0:54, 1].astype(float).to_numpy())
    MinCapacity = np.nan_to_num(gen_data.iloc[0:54, 2].astype(float).to_numpy())
    MinUptime = np.nan_to_num(gen_data.iloc[0:54, 3].astype(float).to_numpy())
    MinDowntime = np.nan_to_num(gen_data.iloc[0:54, 4].astype(float).to_numpy())
    RampUp = np.nan_to_num(gen_data.iloc[0:54, 5].astype(float).to_numpy())
    RampDown = np.nan_to_num(gen_data.iloc[0:54, 6].astype(float).to_numpy())
    FuelCost = np.nan_to_num(gen_data.iloc[0:54, 7].astype(float).to_numpy())
    HotSUCost = np.nan_to_num(gen_data.iloc[0:54, 8].astype(float).to_numpy())
    ColdSUCost = np.nan_to_num(gen_data.iloc[0:54, 9].astype(float).to_numpy())
    ColdTime = np.nan_to_num(gen_data.iloc[0:54, 10].astype(float).to_numpy())
    IncHeatRateA = np.nan_to_num(gen_data.iloc[0:54, 11].astype(float).to_numpy())
    IncHeatRateB = np.nan_to_num(gen_data.iloc[0:54, 12].astype(float).to_numpy())
    IncHeatRateC = np.nan_to_num(gen_data.iloc[0:54, 13].astype(float).to_numpy())
    SDCost = np.nan_to_num(gen_data.iloc[0:54, 14].astype(float).to_numpy())
    InitialCon = np.nan_to_num(gen_data.iloc[0:54, 15].astype(int).to_numpy())
    PmP = np.nan_to_num(gen_data.iloc[0:54, 16].astype(float).to_numpy())

    # Power Grid Parameters
    nbGen = 54            # Number of generators
    nbBus = 118            # Number of buses
    nbLine = 186           # Number of transmission lines
    nbHour = 24          # Time horizon (24 hours)
    reserve = 1.1        # Reserve margin 
    dlload = 5           # Number of load points

    # Energy Storage System (ESS) Parameters
    nbES = int(target_nBES)     # Number of energy storage units
    Ncharge = 0.9        # Charging efficiency (90%)
    Ndischarge = 0.9     # Discharging efficiency (90%)
    ESCCost = 0.5        # Charging cost
    ESDCost = 0.1        # Discharging cost
    MaxCharge = 200    # Maximum charge level
    MinCharge = 0        # Minimum charge level
    Maxqc = 200        # Maximum charging rate
    Minqc = 0            # Minimum charging rate
    Maxqd = 200      # Maximum discharging rate
    Minqd = 0            # Minimum discharging rate
    nbScenarios = 100  # Number of scenarios (100 rows in the CSV)
    nbScenarios_chance = nbScenarios


    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== demand data =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------

    demand_file = os.path.join(data_dir, "load118test.csv")

    demand_data = pd.read_csv(demand_file, header=None)
    load = demand_data.to_numpy()

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== ptdf data =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------

    ptdf_file = os.path.join(data_dir, "ptdf118.csv")

    ptdf_data = pd.read_csv(ptdf_file, header=None)
    ptdf = ptdf_data.to_numpy()

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== solar snd wind data =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------
    # solar_file = os.path.join(data_dir, "SolarGeneration_Savini.csv")
    # wind_file  = os.path.join(data_dir, "Practical_WindGeneration_HighVar_Savini.csv")

    # solar_data = pd.read_csv(solar_file, header=None).to_numpy()
    # wind_data  = pd.read_csv(wind_file,  header=None).to_numpy()

    
    # rng = np.random.default_rng(1000 + run_id)  # reproducible per run

    # idx_s = rng.choice(solar_data.shape[0], size=nbScenarios, replace=False)
    # idx_w = rng.choice(wind_data.shape[0],  size=nbScenarios, replace=False)

    generated_solar = ai["filtered_solar"][scenario_idx].T   # (24,100)
    generated_wind = ai["filtered_wind"][scenario_idx].T     # (24,100)
   
    PEN_CAP = 0.3
    EPS = 1e-9

    daily_load = float(np.sum(load[:nbHour, 2]))  # total demand over 24h

    # total renewable per scenario (length = nbScenarios)
    scenario_totals = np.sum(generated_solar + generated_wind, axis=0)  # sum over time

    mean_total_avail = float(np.mean(scenario_totals))
    target_total = PEN_CAP * daily_load

    alpha_all = 0.0 if mean_total_avail <= EPS else (target_total / mean_total_avail)

    # scale scenarios so that *average* daily renewable ≈ 30% of daily load
    generated_solar *= alpha_all
    generated_wind  *= alpha_all

    # recompute deterministic caps from the scaled scenarios
    solar_expected = generated_solar.mean(axis=1)
    wind_expected  = generated_wind.mean(axis=1)

    

    # -------------------------------------------------------------------------
    # ============ FIRST-STAGE renewable decision variables ===================
    # -------------------------------------------------------------------------
    SolarUsed = model.addVars(nbHour, lb=0, vtype=GRB.CONTINUOUS, name="SolarUsed")
    WindUsed  = model.addVars(nbHour, lb=0, vtype=GRB.CONTINUOUS, name="WindUsed")


    # -------------------------------------------------------------------------
    # ============ Only cap with mean ==========
    # -------------------------------------------------------------------------
    for r in range(nbHour):
        model.addConstr(SolarUsed[r] <= solar_expected[r] , name=f"SolarAvail_0_{r}")
        model.addConstr(WindUsed[r]  <= wind_expected[r],  name=f"WindAvail_0_{r}")


    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== Define: matrix of variables =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------
    x = []  # Generator power output
    x2 = []  # Additional generator variable (e.g., for quadratic cost terms)
    StartUpCost = []  # Start-up cost for each generator
    ShutDownCost = []  # Shut-down cost for each generator
    pwrFlow = []  # Power flow in each transmission line
    qcharge = []  # Charging power of the ESS
    qdischarge = []  # Discharging power of the ESS
    qlevel = []  # Energy level of the ESS

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== Chance Constraint Policy 3 =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------
    epsilon = 1 - float(target_conf) # e.g., 90% confidence
    beta = 0.8     # Scaling factor for uncertainty (adjust this as necessary)
    M = 10000
    # K = int(math.ceil((1 - epsilon) * nbScenarios))
    SOLAR_ZERO_TOL = 1e-6

    # -------------------------
    # Chance Constraint Policy 1 (daily / whole horizon)
    # -------------------------

    # -------------------------------------------------------------------------
    # Chance Constraint Policy 1 (24h / whole horizon renewable utilization)
    # Enforce: in at least (1-eps) scenarios,
    #    sum_t (SolarUsed[t] + WindUsed[t])  >=  beta * sum_t (solar[t,s] + wind[t,s])
    # cc_sc[s] is the 0/1 label: 1 if scenario s satisfies, else 0
    # -------------------------------------------------------------------------

    # force solar usage to 0 at night if deterministic solar cap is 0
    for t in range(nbHour):
        if float(solar_expected[t]) <= SOLAR_ZERO_TOL:
            model.addConstr(SolarUsed[t] == 0.0, name=f"SolarNight_{t}")

    S = nbScenarios
    K = int(math.ceil((1 - epsilon) * S))

    cc_sc = model.addVars(S, vtype=GRB.BINARY, name="CC_sc")  # 1 = scenario counted as satisfied

    # Total scheduled wind over the day (decision, same for all scenarios)
    used_total = gb.quicksum(WindUsed[t] for t in range(nbHour))

    # Build a safe M (important!)
    avail_totals = np.sum(generated_wind, axis=0)          # (S,)
    avail_max = float(np.max(avail_totals))
    cap_total = float(np.sum(wind_expected))               # because WindUsed[t] <= wind_expected[t]
    # M = cap_total + beta * avail_max + 1.0

    for s in range(S):
        avail_total = gb.quicksum(generated_wind[t, s] for t in range(nbHour))  # sum over t (and buses if you have them)

    # If z[s]=1 => enforce used_total >= beta*avail_total
    # If z[s]=0 => relax by M
        model.addConstr(
            used_total >= beta * avail_total - M * (1 - cc_sc[s]),
            name=f"CC_policy1_{s}"
        )

    model.addConstr(gb.quicksum(cc_sc[s] for s in range(S)) >= K, name="ChanceConstraint_Policy1")

   
    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== Creating Variables =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------
    # Initialize decision variables for each time period (r), and generator (g)
    x  = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{r}_{g}") for g in range(nbGen)] for r in range(nbHour)]
    x2 = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x2_{r}_{g}") for g in range(nbGen)] for r in range(nbHour)]
    # Startup / Shutdown cost
    StartUpCost  = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"StartUpCost_{r}_{g}") for g in range(nbGen)] for r in range(nbHour)]
    ShutDownCost = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"ShutDownCost_{r}_{g}") for g in range(nbGen)] for r in range(nbHour)]

    # Power flow variable across each time period, and line
    pwrFlow = [[model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"pwrFlow_{r}_{l}") for l in range(nbLine)] for r in range(nbHour)]

    # Energy storage variables for each time period, and storage system
    qcharge = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"qcharge_{r}_{es}") for es in range(nbES)] for r in range(nbHour)]
    qdischarge = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"qdischarge_{r}_{es}") for es in range(nbES)] for r in range(nbHour)]
    qlevel = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"qlevel_{r}_{es}") for es in range(nbES)] for r in range(nbHour)]

    # Linearized variables for energy flows (lin_qc_arc and lin_qd_arc) across each hour and bus for nbHour-1, as its time step variable
    # lin_qc_arc = arc * qcharge using linear constraints
    # lin_qc_arc = [[[model.addVar(vtype=GRB.CONTINUOUS, name=f"lin_qc_arc_{es}_{r}_{bus}") for bus in range(nbBus)] for r in range(nbHour - 1)] for es in range(nbES)]
    # lin_qd_arc = [[[model.addVar(vtype=GRB.CONTINUOUS, name=f"lin_qd_arc_{es}_{r}_{bus}") for bus in range(nbBus)] for r in range(nbHour - 1)] for es in range(nbES)]
    lin_qc_arc = [[[ model.addVar(vtype=GRB.CONTINUOUS, name=f"lin_qc_arc_{r}_{bus}_{es}") for es in range(nbES)] for bus in range(nbBus)] for r in range(nbHour)]
    lin_qd_arc = [[[model.addVar(vtype=GRB.CONTINUOUS, name=f"lin_qd_arc_{r}_{bus}_{es}") for es in range(nbES)] for bus in range(nbBus)] for r in range(nbHour)]

    arc = []

    # arc
    for es in range(nbES):
        arc.append([])  # One list per storage unit
        for r in range(nbHour):   # Loop over hours (nbHour - 1 because of the time step difference)
            arc[es].append([])
            for n in range(30):   # loop over 30 arcs
                arc[es][r].append(model.addVar(vtype=GRB.BINARY, name=f"arc_{es}_{r}_{n}"))


    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== Identifying bus types =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------
    # Identify bus type (load/generator)
    # totalGen -> generator bus -> bus 0, 1, 5
    # totalLoad -> load bus type -> bus 2, 3, 4
    totalGen = []
    totalLoad = []

    # per load array for 118 busses
    perLoad = np.array([0.014502809,	0.005687008,	0.011090068,	0.008531852,	0,
                0.014786757,	0.00540306,		0,				0,				0,
                0.019905868,	0.013364336,	0.009667646,	0.003980638,	0.025592877,
                0.00710943,		0.003128792,	0.017061025,	0.012796438,	0.005119111,
                0.003980638,	0.002844844,	0.001990319,	0,				0,
                0,				0.017631601,	0.004835162,	0.006825481,	0,
                0.012228541,	0.016777076,	0.006541533,	0.016777076,	0.009383698,
                0.0088158,		0,				0,				0.007232653,	0.005357521,
                0.009911413,	0.009911413,	0.004821769,	0.004286017,	0.01419743,
                0.007500529,	0.009107785,	0.005357521,	0.023305215,	0.004553893,
                0.004553893,	0.004821769,	0.006161149,	0.030269992,	0.01687619,
                0.022501587,	0.003214512,	0.003214512,	0.074201662,	0.020894331,
                0,				0.020626455,	0,				0,				0,
                0.010447165,	0.007500529,	0,				0,				0.017679818,
                0,				0,				0,				0.018215571,	0.012590174,
                0.018215571,	0.016340438,	0.019019199,	0.010447165,	0.034823885,
                0,				0.014465306,	0.005357521,	0.002946636,	0.006429025,
                0.005625397,	0,				0.01285805,		0,				0.020894331,
                0,				0.017411942,	0.003214512,	0.008036281,	0.011250794,
                0.010179289,	0.004018141,	0.009107785,	0,				0.009911413,
                0.005893273,	0.00133938,		0.006161149,	0.010179289,	0.008304157,
                0.01151867,		0.007500529,	0.000535752,	0.002143008,	0.010447165,
                0,				0.006696901,	0,				0.002274268,	0.006254905,
                0,				0.005687008,	0.008839909])

    for t in range(nbHour):
        totalGen.append([model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"totalGen_{t}_{b}") for b in range(nbBus)])
        totalLoad.append([model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"totalLoad_{t}_{b}") for b in range(nbBus)])

    # Add constraints to the model
    for r in range(nbHour):
        
    # -------------------
    # Generator bus injections
    # -------------------

        model.addConstr(totalGen[r][0] == 0 )
        model.addConstr(totalGen[r][1] == 0 )
        model.addConstr(totalGen[r][2] == 0)
        model.addConstr(totalGen[r][3] == x[r][0])
        model.addConstr(totalGen[r][4] == 0)
        model.addConstr(totalGen[r][5] == x[r][1])
        model.addConstr(totalGen[r][6] == 0)
        model.addConstr(totalGen[r][7] == x[r][2])
        model.addConstr(totalGen[r][8] == 0)
        model.addConstr(totalGen[r][9] == x[r][3])
        model.addConstr(totalGen[r][10] == 0)
        model.addConstr(totalGen[r][11] == x[r][4])
        model.addConstr(totalGen[r][12] == 0)
        model.addConstr(totalGen[r][13] == 0)
        model.addConstr(totalGen[r][14] == x[r][5])
        model.addConstr(totalGen[r][15] == 0)
        model.addConstr(totalGen[r][16] == 0)
        model.addConstr(totalGen[r][17] == x[r][6])
        model.addConstr(totalGen[r][18] == x[r][7])
        model.addConstr(totalGen[r][19] == 0)
        model.addConstr(totalGen[r][20] == 0)
        model.addConstr(totalGen[r][21] == 0)
        model.addConstr(totalGen[r][22] == 0)
        model.addConstr(totalGen[r][23] == x[r][8])
        model.addConstr(totalGen[r][24] == x[r][9] + gb.quicksum(lin_qd_arc[r][24][es] for es in range(nbES)))
        model.addConstr(totalGen[r][25] == x[r][10])
        model.addConstr(totalGen[r][26] == x[r][11])
        model.addConstr(totalGen[r][27] == 0)
        model.addConstr(totalGen[r][28] == 0)
        model.addConstr(totalGen[r][29] == 0)
        model.addConstr(totalGen[r][30] == x[r][12])
        model.addConstr(totalGen[r][31] == x[r][13])
        model.addConstr(totalGen[r][32] == 0)
        model.addConstr(totalGen[r][33] == x[r][14])
        model.addConstr(totalGen[r][34] == 0)
        model.addConstr(totalGen[r][35] == x[r][15])
        model.addConstr(totalGen[r][36] == 0)
        model.addConstr(totalGen[r][37] == gb.quicksum(lin_qd_arc[r][37][es] for es in range(nbES)))
        model.addConstr(totalGen[r][38] == 0)
        model.addConstr(totalGen[r][39] == x[r][16])
        model.addConstr(totalGen[r][40] == 0)
        model.addConstr(totalGen[r][41] == x[r][17])
        model.addConstr(totalGen[r][42] == 0)
        model.addConstr(totalGen[r][43] == 0)
        model.addConstr(totalGen[r][44] == 0)
        model.addConstr(totalGen[r][45] == x[r][18])
        model.addConstr(totalGen[r][46] == gb.quicksum(lin_qd_arc[r][46][es] for es in range(nbES)))
        model.addConstr(totalGen[r][47] == 0)
        model.addConstr(totalGen[r][48] == x[r][19])
        model.addConstr(totalGen[r][49] == 0)
        model.addConstr(totalGen[r][50] == 0)
        model.addConstr(totalGen[r][51] == 0)
        model.addConstr(totalGen[r][52] == 0)
        model.addConstr(totalGen[r][53] == x[r][20])
        model.addConstr(totalGen[r][54] == x[r][21])
        model.addConstr(totalGen[r][55] == x[r][22])
        model.addConstr(totalGen[r][56] == 0)
        model.addConstr(totalGen[r][57] == 0)
        model.addConstr(totalGen[r][58] == x[r][23])
        model.addConstr(totalGen[r][59] == 0)
        model.addConstr(totalGen[r][60] == x[r][24])
        model.addConstr(totalGen[r][61] == x[r][25])
        model.addConstr(totalGen[r][62] == 0)
        model.addConstr(totalGen[r][63] == 0)
        model.addConstr(totalGen[r][64] == x[r][26])
        model.addConstr(totalGen[r][65] == x[r][27])
        model.addConstr(totalGen[r][66] == 0)
        model.addConstr(totalGen[r][67] == 0)
        model.addConstr(totalGen[r][68] == x[r][28]+ gb.quicksum(lin_qd_arc[r][68][es] for es in range(nbES)))
        model.addConstr(totalGen[r][69] == x[r][29])
        model.addConstr(totalGen[r][70] == 0)
        model.addConstr(totalGen[r][71] == x[r][30])
        model.addConstr(totalGen[r][72] == x[r][31])
        model.addConstr(totalGen[r][73] == x[r][32])
        model.addConstr(totalGen[r][74] == 0)
        model.addConstr(totalGen[r][75] == x[r][33])
        model.addConstr(totalGen[r][76] == x[r][34] + gb.quicksum(lin_qd_arc[r][76][es] for es in range(nbES)) + SolarUsed[r] + WindUsed[r])
        model.addConstr(totalGen[r][77] == 0)
        model.addConstr(totalGen[r][78] == 0)
        model.addConstr(totalGen[r][79] == x[r][35])
        model.addConstr(totalGen[r][80] == 0)
        model.addConstr(totalGen[r][81] == x[r][36])
        model.addConstr(totalGen[r][82] == gb.quicksum(lin_qd_arc[r][82][es] for es in range(nbES)))
        model.addConstr(totalGen[r][83] == 0)
        model.addConstr(totalGen[r][84] == x[r][37])
        model.addConstr(totalGen[r][85] == 0)
        model.addConstr(totalGen[r][86] == x[r][38])
        model.addConstr(totalGen[r][87] == 0)
        model.addConstr(totalGen[r][88] == x[r][39])
        model.addConstr(totalGen[r][89] == x[r][40])
        model.addConstr(totalGen[r][90] == x[r][41])
        model.addConstr(totalGen[r][91] == x[r][42] + gb.quicksum(lin_qd_arc[r][91][es]for es in range(nbES)))
        model.addConstr(totalGen[r][92] == 0)
        model.addConstr(totalGen[r][93] == 0)
        model.addConstr(totalGen[r][94] == 0)
        model.addConstr(totalGen[r][95] == 0)
        model.addConstr(totalGen[r][96] == 0)
        model.addConstr(totalGen[r][97] == 0)
        model.addConstr(totalGen[r][98] == x[r][43])
        model.addConstr(totalGen[r][99] == x[r][44])
        model.addConstr(totalGen[r][100] == 0)
        model.addConstr(totalGen[r][101] == 0)
        model.addConstr(totalGen[r][102] == x[r][45])
        model.addConstr(totalGen[r][103] == x[r][46])
        model.addConstr(totalGen[r][104] == x[r][47])
        model.addConstr(totalGen[r][105] == 0)
        model.addConstr(totalGen[r][106] == x[r][48])
        model.addConstr(totalGen[r][107] == 0)
        model.addConstr(totalGen[r][108] == 0)
        model.addConstr(totalGen[r][109] == x[r][49])
        model.addConstr(totalGen[r][110] == x[r][50])
        model.addConstr(totalGen[r][111] == x[r][51])
        model.addConstr(totalGen[r][112] == x[r][52])
        model.addConstr(totalGen[r][113] == 0)
        model.addConstr(totalGen[r][114] == 0)
        model.addConstr(totalGen[r][115] == x[r][53])
        model.addConstr(totalGen[r][116] ==  gb.quicksum(lin_qd_arc[r][116][es] for es in range(nbES)))
        model.addConstr(totalGen[r][117] == 0)

        # -------------------
        # Load bus injections
        # -------------------

        model.addConstr(totalLoad[r][0] == perLoad[0] * load[r, 2])
        model.addConstr(totalLoad[r][1] == perLoad[1] * load[r, 2] )
        model.addConstr(totalLoad[r][2] == perLoad[2] * load[r, 2])
        model.addConstr(totalLoad[r][3] == perLoad[3] * load[r, 2])
        model.addConstr(totalLoad[r][4] == perLoad[4] * load[r, 2])
        model.addConstr(totalLoad[r][5] == perLoad[5] * load[r, 2])
        model.addConstr(totalLoad[r][6] == perLoad[6] * load[r, 2])
        model.addConstr(totalLoad[r][7] == perLoad[7] * load[r, 2])
        model.addConstr(totalLoad[r][8] == perLoad[8] * load[r, 2])
        model.addConstr(totalLoad[r][9] == perLoad[9] * load[r, 2])
        model.addConstr(totalLoad[r][10] == perLoad[10] * load[r, 2])
        model.addConstr(totalLoad[r][11] == perLoad[11] * load[r, 2])
        model.addConstr(totalLoad[r][12] == perLoad[12] * load[r, 2])
        model.addConstr(totalLoad[r][13] == perLoad[13] * load[r, 2])
        model.addConstr(totalLoad[r][14] == perLoad[14] * load[r, 2])
        model.addConstr(totalLoad[r][15] == perLoad[15] * load[r, 2])
        model.addConstr(totalLoad[r][16] == perLoad[16] * load[r, 2])
        model.addConstr(totalLoad[r][17] == perLoad[17] * load[r, 2])
        model.addConstr(totalLoad[r][18] == perLoad[18] * load[r, 2])
        model.addConstr(totalLoad[r][19] == perLoad[19] * load[r, 2])
        model.addConstr(totalLoad[r][20] == perLoad[20] * load[r, 2])
        model.addConstr(totalLoad[r][21] == perLoad[21] * load[r, 2])
        model.addConstr(totalLoad[r][22] == perLoad[22] * load[r, 2])
        model.addConstr(totalLoad[r][23] == perLoad[23] * load[r, 2])
        model.addConstr(totalLoad[r][24] == perLoad[24] * load[r, 2] + gb.quicksum(lin_qc_arc[r][24][es] for es in range(nbES)))
        model.addConstr(totalLoad[r][25] == perLoad[25] * load[r, 2])
        model.addConstr(totalLoad[r][26] == perLoad[26] * load[r, 2])
        model.addConstr(totalLoad[r][27] == perLoad[27] * load[r, 2])
        model.addConstr(totalLoad[r][28] == perLoad[28] * load[r, 2])
        model.addConstr(totalLoad[r][29] == perLoad[29] * load[r, 2])
        model.addConstr(totalLoad[r][30] == perLoad[30] * load[r, 2])
        model.addConstr(totalLoad[r][31] == perLoad[31] * load[r, 2])
        model.addConstr(totalLoad[r][32] == perLoad[32] * load[r, 2])
        model.addConstr(totalLoad[r][33] == perLoad[33] * load[r, 2])
        model.addConstr(totalLoad[r][34] == perLoad[34] * load[r, 2])
        model.addConstr(totalLoad[r][35] == perLoad[35] * load[r, 2])
        model.addConstr(totalLoad[r][36] == perLoad[36] * load[r, 2])
        model.addConstr(totalLoad[r][37] == perLoad[37] * load[r, 2] + gb.quicksum(lin_qc_arc[r][37][es] for es in range(nbES)))
        model.addConstr(totalLoad[r][38] == perLoad[38] * load[r, 2])
        model.addConstr(totalLoad[r][39] == perLoad[39] * load[r, 2])
        model.addConstr(totalLoad[r][40] == perLoad[40] * load[r, 2])
        model.addConstr(totalLoad[r][41] == perLoad[41] * load[r, 2])
        model.addConstr(totalLoad[r][42] == perLoad[42] * load[r, 2])
        model.addConstr(totalLoad[r][43] == perLoad[43] * load[r, 2])
        model.addConstr(totalLoad[r][44] == perLoad[44] * load[r, 2])
        model.addConstr(totalLoad[r][45] == perLoad[45] * load[r, 2])
        model.addConstr(totalLoad[r][46] == perLoad[46] * load[r, 2] + gb.quicksum(lin_qc_arc[r][46][es] for es in range(nbES)))
        model.addConstr(totalLoad[r][47] == perLoad[47] * load[r, 2])
        model.addConstr(totalLoad[r][48] == perLoad[48] * load[r, 2])
        model.addConstr(totalLoad[r][49] == perLoad[49] * load[r, 2])
        model.addConstr(totalLoad[r][50] == perLoad[50] * load[r, 2])
        model.addConstr(totalLoad[r][51] == perLoad[51] * load[r, 2])
        model.addConstr(totalLoad[r][52] == perLoad[52] * load[r, 2])
        model.addConstr(totalLoad[r][53] == perLoad[53] * load[r, 2])
        model.addConstr(totalLoad[r][54] == perLoad[54] * load[r, 2])
        model.addConstr(totalLoad[r][55] == perLoad[55] * load[r, 2])
        model.addConstr(totalLoad[r][56] == perLoad[56] * load[r, 2])
        model.addConstr(totalLoad[r][57] == perLoad[57] * load[r, 2])
        model.addConstr(totalLoad[r][58] == perLoad[58] * load[r, 2])
        model.addConstr(totalLoad[r][59] == perLoad[59] * load[r, 2])
        model.addConstr(totalLoad[r][60] == perLoad[60] * load[r, 2])
        model.addConstr(totalLoad[r][61] == perLoad[61] * load[r, 2])
        model.addConstr(totalLoad[r][62] == perLoad[62] * load[r, 2])
        model.addConstr(totalLoad[r][63] == perLoad[63] * load[r, 2])
        model.addConstr(totalLoad[r][64] == perLoad[64] * load[r, 2])
        model.addConstr(totalLoad[r][65] == perLoad[65] * load[r, 2])
        model.addConstr(totalLoad[r][66] == perLoad[66] * load[r, 2])
        model.addConstr(totalLoad[r][67] == perLoad[67] * load[r, 2])
        model.addConstr(totalLoad[r][68] == perLoad[68] * load[r, 2] + gb.quicksum(lin_qc_arc[r][68][es] for es in range(nbES)))
        model.addConstr(totalLoad[r][69] == perLoad[69] * load[r, 2])
        model.addConstr(totalLoad[r][70] == perLoad[70] * load[r, 2])
        model.addConstr(totalLoad[r][71] == perLoad[71] * load[r, 2])
        model.addConstr(totalLoad[r][72] == perLoad[72] * load[r, 2])
        model.addConstr(totalLoad[r][73] == perLoad[73] * load[r, 2])
        model.addConstr(totalLoad[r][74] == perLoad[74] * load[r, 2])
        model.addConstr(totalLoad[r][75] == perLoad[75] * load[r, 2])
        model.addConstr(totalLoad[r][76] == perLoad[76] * load[r, 2] + gb.quicksum(lin_qc_arc[r][76][es] for es in range(nbES))) 
        model.addConstr(totalLoad[r][77] == perLoad[77] * load[r, 2])
        model.addConstr(totalLoad[r][78] == perLoad[78] * load[r, 2])
        model.addConstr(totalLoad[r][79] == perLoad[79] * load[r, 2])
        model.addConstr(totalLoad[r][80] == perLoad[80] * load[r, 2])
        model.addConstr(totalLoad[r][81] == perLoad[81] * load[r, 2])
        model.addConstr(totalLoad[r][82] == perLoad[82] * load[r, 2] + gb.quicksum(lin_qc_arc[r][82][es] for es in range(nbES))) 
        model.addConstr(totalLoad[r][83] == perLoad[83] * load[r, 2])
        model.addConstr(totalLoad[r][84] == perLoad[84] * load[r, 2])
        model.addConstr(totalLoad[r][85] == perLoad[85] * load[r, 2])
        model.addConstr(totalLoad[r][86] == perLoad[86] * load[r, 2])
        model.addConstr(totalLoad[r][87] == perLoad[87] * load[r, 2])
        model.addConstr(totalLoad[r][88] == perLoad[88] * load[r, 2])
        model.addConstr(totalLoad[r][89] == perLoad[89] * load[r, 2])
        model.addConstr(totalLoad[r][90] == perLoad[90] * load[r, 2])
        model.addConstr(totalLoad[r][91] == perLoad[91] * load[r, 2] + gb.quicksum(lin_qc_arc[r][91][es] for es in range(nbES))) 
        model.addConstr(totalLoad[r][92] == perLoad[92] * load[r, 2])
        model.addConstr(totalLoad[r][93] == perLoad[93] * load[r, 2])
        model.addConstr(totalLoad[r][94] == perLoad[94] * load[r, 2])
        model.addConstr(totalLoad[r][95] == perLoad[95] * load[r, 2])
        model.addConstr(totalLoad[r][96] == perLoad[96] * load[r, 2])
        model.addConstr(totalLoad[r][97] == perLoad[97] * load[r, 2])
        model.addConstr(totalLoad[r][98] == perLoad[98] * load[r, 2])
        model.addConstr(totalLoad[r][99] == perLoad[99] * load[r, 2])
        model.addConstr(totalLoad[r][100] == perLoad[100] * load[r, 2])
        model.addConstr(totalLoad[r][101] == perLoad[101] * load[r, 2])
        model.addConstr(totalLoad[r][102] == perLoad[102] * load[r, 2])
        model.addConstr(totalLoad[r][103] == perLoad[103] * load[r, 2])
        model.addConstr(totalLoad[r][104] == perLoad[104] * load[r, 2])
        model.addConstr(totalLoad[r][105] == perLoad[105] * load[r, 2])
        model.addConstr(totalLoad[r][106] == perLoad[106] * load[r, 2])
        model.addConstr(totalLoad[r][107] == perLoad[107] * load[r, 2])
        model.addConstr(totalLoad[r][108] == perLoad[108] * load[r, 2])
        model.addConstr(totalLoad[r][109] == perLoad[109] * load[r, 2])
        model.addConstr(totalLoad[r][110] == perLoad[110] * load[r, 2])
        model.addConstr(totalLoad[r][111] == perLoad[111] * load[r, 2])
        model.addConstr(totalLoad[r][112] == perLoad[112] * load[r, 2])
        model.addConstr(totalLoad[r][113] == perLoad[113] * load[r, 2])
        model.addConstr(totalLoad[r][114] == perLoad[114] * load[r, 2])
        model.addConstr(totalLoad[r][115] == perLoad[115] * load[r, 2])
        model.addConstr(totalLoad[r][116] == perLoad[116] * load[r, 2] + gb.quicksum(lin_qc_arc[r][116][es] for es in range(nbES))) 
        model.addConstr(totalLoad[r][117] == perLoad[117] * load[r, 2])

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== linearized arcs =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  
    '''only for the bus with TES stations at those buses'''
    # loops over storage units and scenarios for each hour
    '''why nbHour-1?'''
    for r in range(nbHour): 
        for es in range(nbES):
    # TES Stations are located at bus indices: 24, 37, 46, 68, 76, 82, 91, 116

    # -------------------
    # Charge Constraints
    # -------------------

        # ===== Bus 24 (arc 12) =====
            # If the arc switch is OFF (0), force lin_qc_arc to be 0. If it’s ON (1), allow lin_qc_arc to be as large as the maximum possible charge rate.
            model.addConstr(lin_qc_arc[r][24][es] <= arc[es][r][12] * Maxqc)
            # the amount of charge you attribute to bus 24 via the arc (the linearized variable lin_qc_arc) can never be larger than the actual charging power qcharge at that hour r.
            model.addConstr(lin_qc_arc[r][24][es] <= qcharge[r][es])
            # lin_qc_arc can drop to 0 when the arc is OFF but must match qcharge when the arc is ON.
            model.addConstr(lin_qc_arc[r][24][es] >= qcharge[r][es] - (1 - arc[es][r][12]) * Maxqc)

            # ===== Bus 37 (arc 20) =====
            model.addConstr(lin_qc_arc[r][37][es] <= arc[es][r][20] * Maxqc)
            model.addConstr(lin_qc_arc[r][37][es] <= qcharge[r][es])
            model.addConstr(lin_qc_arc[r][37][es] >= qcharge[r][es] - (1 - arc[es][r][20]) * Maxqc)

            # ===== Bus 46 (arc 3) =====
            model.addConstr(lin_qc_arc[r][46][es] <= arc[es][r][3] * Maxqc)
            model.addConstr(lin_qc_arc[r][46][es] <= qcharge[r][es])
            model.addConstr(lin_qc_arc[r][46][es] >= qcharge[r][es] - (1 - arc[es][r][3]) * Maxqc)
            
            # ===== Bus 68 (arc 6) =====
            model.addConstr(lin_qc_arc[r][68][es] <= arc[es][r][6] * Maxqc)
            model.addConstr(lin_qc_arc[r][68][es] <= qcharge[r][es])
            model.addConstr(lin_qc_arc[r][68][es] >= qcharge[r][es] - (1 - arc[es][r][6]) * Maxqc)

            # ===== Bus 76 (arc 25) =====
            model.addConstr(lin_qc_arc[r][76][es] <= arc[es][r][25] * Maxqc)
            model.addConstr(lin_qc_arc[r][76][es] <= qcharge[r][es])
            model.addConstr(lin_qc_arc[r][76][es] >= qcharge[r][es] - (1 - arc[es][r][25]) * Maxqc)

            # ===== Bus 82 (arc 16) =====
            model.addConstr(lin_qc_arc[r][82][es] <= arc[es][r][16] * Maxqc)
            model.addConstr(lin_qc_arc[r][82][es] <= qcharge[r][es])
            model.addConstr(lin_qc_arc[r][82][es] >= qcharge[r][es] - (1 - arc[es][r][16]) * Maxqc)

            # ===== Bus 91 (arc 9) =====
            model.addConstr(lin_qc_arc[r][91][es] <= arc[es][r][9] * Maxqc)
            model.addConstr(lin_qc_arc[r][91][es] <= qcharge[r][es])
            model.addConstr(lin_qc_arc[r][91][es] >= qcharge[r][es] - (1 - arc[es][r][9]) * Maxqc)

            # ===== Bus 116 (arc 0) =====
            model.addConstr(lin_qc_arc[r][116][es] <= arc[es][r][0] * Maxqc)
            model.addConstr(lin_qc_arc[r][116][es] <= qcharge[r][es])
            model.addConstr(lin_qc_arc[r][116][es] >= qcharge[r][es] - (1 - arc[es][r][0]) * Maxqc)

        # -------------------
        # Discharge Constraints
        # -------------------  

            # ===== Bus 24 (arc 12) =====
            # If the switch is OFF (0), this forces lin_qd_arc = 0. If it’s ON (1), it just says lin_qd_arc can be as big as maximum discharging rate
            model.addConstr(lin_qd_arc[r][24][es] <= arc[es][r][12] * Maxqd)
            # You can’t claim more discharge at each bus than the TES is actually discharging overall.
            model.addConstr(lin_qd_arc[r][24][es] <= qdischarge[r][es])
            # If the switch is ON (1), this becomes lin_qd_arc ≥ qdischarge. Together with (2), that pins it to lin_qd_arc = qdischarge.
            # If the switch is OFF (0), it becomes lin_qd_arc ≥ qdischarge − Maxqd, which doesn’t force anything positive, so combined with (1) you get lin_qd_arc = 0.
            model.addConstr(lin_qd_arc[r][24][es] >= qdischarge[r][es] - (1 - arc[es][r][12]) * Maxqd)

            # ===== Bus 37 (arc 20) =====
            model.addConstr(lin_qd_arc[r][37][es] <= arc[es][r][20] * Maxqd)
            model.addConstr(lin_qd_arc[r][37][es] <= qdischarge[r][es])
            model.addConstr(lin_qd_arc[r][37][es] >= qdischarge[r][es] - (1 - arc[es][r][20]) * Maxqd)

            # ===== Bus 46 (arc 3) =====
            model.addConstr(lin_qd_arc[r][46][es] <= arc[es][r][3] * Maxqd)
            model.addConstr(lin_qd_arc[r][46][es] <= qdischarge[r][es])
            model.addConstr(lin_qd_arc[r][46][es] >= qdischarge[r][es] - (1 - arc[es][r][3]) * Maxqd)

            # ===== Bus 68 (arc 6) =====
            model.addConstr(lin_qd_arc[r][68][es] <= arc[es][r][6] * Maxqd)
            model.addConstr(lin_qd_arc[r][68][es] <= qdischarge[r][es])
            model.addConstr(lin_qd_arc[r][68][es] >= qdischarge[r][es] - (1 - arc[es][r][6]) * Maxqd)

            # ===== Bus 76 (arc 25) =====
            model.addConstr(lin_qd_arc[r][76][es] <= arc[es][r][25] * Maxqd)
            model.addConstr(lin_qd_arc[r][76][es] <= qdischarge[r][es])
            model.addConstr(lin_qd_arc[r][76][es] >= qdischarge[r][es] - (1 - arc[es][r][25]) * Maxqd)

            # ===== Bus 82 (arc 16) =====
            model.addConstr(lin_qd_arc[r][82][es] <= arc[es][r][16] * Maxqd)
            model.addConstr(lin_qd_arc[r][82][es] <= qdischarge[r][es])
            model.addConstr(lin_qd_arc[r][82][es] >= qdischarge[r][es] - (1 - arc[es][r][16]) * Maxqd)

            # ===== Bus 91 (arc 9) =====
            model.addConstr(lin_qd_arc[r][91][es] <= arc[es][r][9] * Maxqd)
            model.addConstr(lin_qd_arc[r][91][es] <= qdischarge[r][es])
            model.addConstr(lin_qd_arc[r][91][es] >= qdischarge[r][es] - (1 - arc[es][r][9]) * Maxqd)

            # ===== Bus 116 (arc 0) =====
            model.addConstr(lin_qd_arc[r][116][es] <= arc[es][r][0] * Maxqd)
            model.addConstr(lin_qd_arc[r][116][es] <= qdischarge[r][es])
            model.addConstr(lin_qd_arc[r][116][es] >= qdischarge[r][es] - (1 - arc[es][r][0]) * Maxqd)


    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== Ramping Limit =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  

    # CONSTRAINT: Ramping Limit (loops over generators for each hour and considers for all hours in the 24 hour window)
    for r in range(1, nbHour):
        for i in range(nbGen):
            model.addConstr(x[r][i] - x[r - 1][i] <= RampUp[i],   name=f"RampUp_{r}_{i}")
            model.addConstr(x[r - 1][i] - x[r][i] <= RampDown[i], name=f"RampDown_{r}_{i}")

    # for r in range(1, nbHour):
    #     for s in range(nbScenarios):
    #         for i in range(nbGen):
    #             model.addConstr(x[r][s][i] - x[r - 1][s][i] <= RampUp[i], name=f"RampUp_{r}_{s}_{i}")
    #             model.addConstr(x[r - 1][s][i] - x[r][s][i] <= RampDown[i], name=f"RampDown_{r}_{s}_{i}")


    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== Defining On/Off Variables =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  

    # Define: max hour
    tempHour = np.zeros(nbGen)
    AddTime = 0

    # Calculate tempHour and find the maximum AddTime
    for r in range(nbGen):
        tempHour[r] = max(MinUptime[r], MinDowntime[r])  # Find the maximum of MinUptime and MinDowntime for each generator
        if tempHour[r] > AddTime:
            AddTime = tempHour[r]  # Update AddTime to the maximum value found

    AddTime += 1  # Increment AddTime by 1

    # Define: OnOff matrix
    OnOff = []

    for t in range(int(nbHour + AddTime)):
        OnOff.append(
                [model.addVar(vtype=GRB.BINARY, name=f"OnOff_{t}_{g}")
                for g in range(nbGen)]
            )

    # for t in range(int(nbHour + AddTime)):
    #     OnOff.append([])  # Time dimension
    #     for s in range(nbScenarios):
    #         # Binary variables for each generator (0 or 1)
    #         OnOff[t].append([model.addVar(vtype=GRB.BINARY, name=f"OnOff_{t}_{s}_{g}") for g in range(nbGen)])


    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== Minimum Up/Down Constraints =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  

    # Minimum up time constraint
    for g in range(nbGen):
        for r in range(1, nbHour + int(AddTime) - int(MinUptime[g]) - 1):
            for k in range(r, int(MinUptime[g] + r - 1)):
                model.addConstr(
                    -OnOff[r - 1][g] + OnOff[r][g] - OnOff[k][g] <= 0,
                    name=f"MinUptime_{r}_{k}_{g}"
                )

    # Minimum down time constraint
    for g in range(nbGen):
        for r in range(1, nbHour + int(AddTime) - int(MinDowntime[g]) - 1):
            # When a unit turns on at hour r, you must keep it on for k = r … r+L-1. because if min time = L = 4, r counts as 1hour of min time (r+3 min hours)
            for k in range(r, int(MinDowntime[g] + r - 1)):
                model.addConstr(
                    OnOff[r - 1][g] - OnOff[r][g] + OnOff[k][g] <= 1,
                    name=f"MinDowntime_{r}_{k}_{g}"
                )

    # # Minimum up time constraint(loops over each hour, each generator for each scenario and considers for all scenarios)
    # for s in range(nbScenarios): 
    #     for g in range(nbGen):
    #         for r in range(1, nbHour + int(AddTime) - int(MinUptime[g])-1):
    #             for k in range(r, int(MinUptime[g] + r-1)):
    #                 model.addConstr(-OnOff[r - 1][s][g] + OnOff[r][s][g] - OnOff[k][s][g] <= 0, 
    #                                 name=f"MinUptime_{r}_{k}_{s}_{g}")

    # # Minimum down time constraint (loops over each hour, each generator for each scenario and considers for all scenarios)
    # for s in range(nbScenarios):
    #     for g in range(nbGen):
    #         for r in range(1, nbHour + int(AddTime) - int(MinDowntime[g])-1):
    #             for k in range(r, int(MinDowntime[g] + r-1)):
    #                 model.addConstr(OnOff[r - 1][s][g] - OnOff[r][s][g] + OnOff[k][s][g] <= 1, 
    #                                 name=f"MinDowntime_{r}_{k}_{s}_{g}")


    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== Power Balance, Spinning Reserve, Generation Limit =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  

    # -------------------------
    # Power Balance 
    # -------------------------

    for r in range(nbHour):
    # For every hour r and scenario s,generation − charging into storage + discharging from storage + solar dispatched + wind dispatched
    # must exactly equal the demand at that hour.
        model.addConstr(gb.quicksum(x[r][g] for g in range(nbGen)) - gb.quicksum(qcharge[r][es] for es in range(nbES))
            + gb.quicksum(qdischarge[r][es] for es in range(nbES)) + SolarUsed[r] + WindUsed[r]
            == load[r,2],
            name=f"PowerBalance_{r}"
        )


    # -------------------------
    # Spinning Reserve 
    # -------------------------
    # Spinning Reserve
    # for each hour r, the max capacity of all generators that are ON must be at least a safety margin above the net load
        model.addConstr(
            gb.quicksum(OnOff[r][g] * MaxCapacity[g] for g in range(nbGen)) >= reserve * (load[r,2] - SolarUsed[r] - WindUsed[r]),
            name=f"SpinningReserve_{r}"
        )

    # -------------------------
    # Generation Limits
    # -------------------------

        for g in range(nbGen):
            # -------------------------
            # Generator capacity limits 
            # -------------------------
            model.addConstr(x[r][g] <= MaxCapacity[g] * OnOff[r][g], name=f"Pmax_{r}_{g}")
            model.addConstr(x[r][g] >= MinCapacity[g] * OnOff[r][g], name=f"Pmin_{r}_{g}")


            # -------------------------
            # Piecewise linearization for quadratic term
            # -------------------------
            xpts = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525]
            ypts= [x**2 for x in xpts]

            model.addGenConstrPWL(x[r][g], x2[r][g], xpts, ypts, name=f"Piecewise_{r}_{g}")


    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================= Generation and Load Constraints =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------          
    # Define: GenLoad matrix
    GenLoad = []

    GenLoad = []
    for t in range(nbHour):
        GenLoad.append([])  # Initialize the first dimension (nbHour)
        for i in range(nbBus):
            GenLoad[t].append(
                model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"GenLoad_{t}_{i}"))

    # ------------------------------------------
    # Net injection = Generation – Load
    # ------------------------------------------

    # For each hour t and bus i, you define the net injection at that bus as
    for t in range(nbHour):
        for i in range(nbBus):
            model.addConstr(
                GenLoad[t][i] == totalGen[t][i] - totalLoad[t][i], name=f"GenLoadConstr_{t}_{i}")



    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================= Power Flow Constraints =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  

    # Calculate power flow of line k @ hour r
    FlowLimit = [175, 175, 500, 175, 175, 175, 500, 500, 500, 175,
                175, 175, 175, 175, 175, 175, 175, 175, 175, 175,
                500, 175, 175, 175, 175, 175, 175, 175, 175, 175,
                500, 500, 500, 175, 175, 500, 175, 500, 175, 175,
                140, 175, 175, 175, 175, 175, 175, 175, 175, 500,
                500, 175, 175, 175, 175, 175, 175, 175, 175, 175,
                175, 175, 175, 175, 175, 175, 175, 175, 175, 175,
                175, 175, 175, 175, 175, 175, 175, 175, 175, 175,
                175, 175, 175, 175, 175, 175, 175, 175, 175, 500,
                175, 175, 500, 500, 500, 500, 500, 500, 500, 175,
                175, 500, 175, 500, 175, 175, 500, 500, 175, 175,
                175, 175, 175, 175, 175, 500, 175, 175, 175, 175,
                175, 175, 500, 500, 175, 500, 500, 200, 200, 175,
                175, 175, 500, 500, 175, 175, 500, 500, 500, 175,
                500, 500, 175, 175, 175, 175, 175, 175, 175, 175,
                175, 175, 200, 175, 175, 175, 175, 175, 175, 175,
                175, 175, 500, 175, 175, 175, 175, 175, 175, 175,
                175, 175, 175, 175, 175, 175, 175, 175, 500, 175,
                175, 175, 500, 175, 175, 175]
    tempFlow = [0 for l in range(nbLine)] 


    for t in range(nbHour):
        for l in range(nbLine):
            # Reset tempFlow for each scenario and line
            #tempFlow[l] = OnOff[0][0] - OnOff[0][0]
            tempFlow[l] = 0

    # Populate tempFlow based on GenLoad and PTDF
        for l in range(nbLine):
            for i in range(nbBus):
                tempFlow[l] += GenLoad[t][i] * ptdf[l][i]

            # Introduce a variable to represent the absolute value of tempFlow
            abs_tempFlow = model.addVar(vtype=GRB.CONTINUOUS, name=f"abs_tempFlow_{t}_{l}")

            # Add two constraints to model the absolute value of tempFlow
            model.addConstr(abs_tempFlow >= tempFlow[l], name=f"AbsTempFlow_Pos_{t}_{l}")
            model.addConstr(abs_tempFlow >= -tempFlow[l], name=f"AbsTempFlow_Neg_{t}_{l}")
            
            # Now, use abs_tempFlow for the power flow constraint
            model.addConstr(pwrFlow[t][l] == abs_tempFlow, name=f"PowerFlow_{t}_{l}")


    for t in range(nbHour):
        for l in range(nbLine):
            model.addConstr(pwrFlow[t][l] <= FlowLimit[l], name=f"FlowLimit_{t}_{l}")



    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================= Shut Down Cost =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  

    for r in range(1, nbHour):
        for i in range(nbGen):
            model.addConstr(ShutDownCost[r][i] >= SDCost[i] * (OnOff[r-1][i] - OnOff[r][i]),
                            name=f"ShutDownCost_{r}_{i}")

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================= Start Up Cost =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  
    # Define: StarWiseCost matrix
    StarWiseCost = [[0 for g in range(nbGen)] for t in range(nbHour + int(AddTime))]

    # Identify whether it is a cold or hot start-up
    for i in range(nbGen):
        for g in range(1, int(nbHour + ColdTime[i] + MinDowntime[i])):
            if g <= ColdTime[i] + MinDowntime[i]:
                StarWiseCost[g][i] = HotSUCost[i]  # Hot startup cost
            else:
                StarWiseCost[g][i] = ColdSUCost[i]  # Cold startup cost

    '''
    example: if ColdTime=3, MinDowntime=2 → L=5.
    You need to check g = 1,2,3,4,5.
    range(1, 5) would stop at 4 (miss the boundary).
    range(1, 5+1) includes 5 — that’s the whole point of the +1.
    '''
    for r in range(nbHour):
        for i in range(nbGen):
            # Reset tempstate for each generator at each hour
            tempstate = 0

            if int(InitialCon[i]) < 0:
                # Loop over ColdTime and MinDowntime for cold start
                for g in range(1, int(ColdTime[i] + MinDowntime[i] + 1)):
                    if r - g >= 0:
                        tempstate += OnOff[r - g][i]

                        # Add constraint for startup cost
                        model.addConstr(
                            (StarWiseCost[g - InitialCon[i]][i] * (OnOff[r][i] - tempstate)) 
                            <= StartUpCost[r][i],
                            name=f"ColdStartUpCost_{r}_{i}_{g}"
                        )

                        # If it's cold start, update the InitialCon condition
                        if StarWiseCost[g - InitialCon[i]][i] == ColdSUCost[i]:
                            InitialCon[i] = 1

            elif int(InitialCon[i]) > 0:
                # Loop over ColdTime and MinDowntime for hot start
                for j in range(1, int(ColdTime[i] + MinDowntime[i] + 1)):
                    if r - j >= 0:
                        tempstate += OnOff[r - j][i]

                        # Add constraint for startup cost
                        model.addConstr(
                            (StarWiseCost[j][i] * (OnOff[r][i] - tempstate)) 
                            <= StartUpCost[r][i],
                            name=f"HotStartUpCost_{r}_{i}_{j}"
                        )
    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================= ESS Charging or Discharging Constraints =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  

    # Define: ESOnOff matrix
    # ESCOnOff[t][s][i] = 1 if Energy Storage i is charging at time t in scenario s, else 0.
    # ESDOnOff[t][s][i] = 1 if Energy Storage i is discharging at time t in scenario s, else 0.
    ESCOnOff = [[model.addVar(vtype=GRB.BINARY, name=f"ESCOnOff_{t}_{i}") for i in range(nbES)] for t in range(int(nbHour + AddTime))]
    ESDOnOff = [[model.addVar(vtype=GRB.BINARY, name=f"ESDOnOff_{t}_{i}") for i in range(nbES)] for t in range(int(nbHour + AddTime))]

    # Min/Max Charging/Discharging & Min/Max Energy Level
    # Conceptually, arc[s][i][r][…] represents a transition during the step r→r+1 (i.e., an action taken between hour r and r+1). therefore its nbHour-1
    for r in range(nbHour):
        for i in range(nbES):
            # Ensure that the ESS can charge/discharge only when on the relevant arcs, basically when staying in the same station
            model.addConstr(
                arc[i][r][0] + arc[i][r][3] + arc[i][r][6]+
                arc[i][r][9] + arc[i][r][12] + arc[i][r][16]+
                arc[i][r][20] + arc[i][r][25] 
                >= ESCOnOff[r][i] + ESDOnOff[r][i], 
                name=f"ArcCondition_{r}_{i}"
            )

            # Min/Max Charging Constraints
            model.addConstr(Minqc * ESCOnOff[r][i] <= qcharge[r][i], name=f"MinCharge_{r}_{i}")
            model.addConstr(qcharge[r][i] <= Maxqc * ESCOnOff[r][i], name=f"MaxCharge_{r}_{i}")

            # Min/Max Charging and Discharging Constraints
            model.addConstr(Minqd * ESDOnOff[r][i] <= qdischarge[r][i], name=f"MinDischarge_{r}_{i}")
            model.addConstr(qdischarge[r][i] <= Maxqd * ESDOnOff[r][i], name=f"MaxDischarge_{r}_{i}")
            
            # Min/Max Energy Level Constraints
            model.addConstr(MinCharge / MaxCharge <= qlevel[r][i], name=f"MinEnergyLevel_{r}_{i}")
            model.addConstr(qlevel[r][i] <= 1, name=f"MaxEnergyLevel_{r}_{i}")


    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================= ESS Energy Level (SOC) Constraints =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  

    # Energy level at hour 24 must be 50% of capacity for every scenario and storage.

    for i in range(nbES):
        model.addConstr(qlevel[23][i] == 0.5, name=f"EnergyLevel24_{i}")

    # Energy level @ hour 0
    for i in range(nbES):
        model.addConstr(qlevel[0][i] == 0.5 + ((Ncharge * qcharge[0][i] - qdischarge[0][i] / Ndischarge) / MaxCharge), name=f"EnergyLevel0_{i}")

    # Relating Energy @ hour r and r-1

    for r in range(1, nbHour):
        for i in range(nbES):
            model.addConstr(qlevel[r][i] == qlevel[r - 1][i] + ((Ncharge * qcharge[r][i] - qdischarge[r][i] / Ndischarge) / MaxCharge), name=f"EnergyBalance_{r}_{i}")


    #-----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================= Arc Constraints =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------  
    for i in range(nbES):
    # at hour 0 must start from station 0
        model.addConstr(arc[i][0][0] + arc[i][0][1] + arc[i][0][2] == 1, name=f"ArcLimit_0_{i}")
    # at the last hour 22 to 23, must be from
        model.addConstr(arc[i][22][9] + arc[i][22][19] + arc[i][22][29] == 1, name=f"ArcLimit_22_{i}")

    # For each TES, at each time step, they can only be on one arc
        for t in range(nbHour):
            model.addConstr(gb.quicksum(arc[i][t][arc_idx] for arc_idx in range(30)) == 1, name=f"ArcLimitSum_{t}_{i}")

        # TES ending at one bus at hour t should start from the same bus at hour t+1
        for t in range(nbHour - 1):
            model.addConstr(arc[i][t][0] + arc[i][t][15] + arc[i][t][24] == arc[i][t + 1][0] + arc[i][t + 1][1] + arc[i][t + 1][2], name=f"ArcBus1_{t}_{i}")
            model.addConstr(arc[i][t][3] + arc[i][t][7] + arc[i][t][22] == arc[i][t + 1][3] + arc[i][t + 1][4] + arc[i][t + 1][5], name=f"ArcBus2_{t}_{i}")
            model.addConstr(arc[i][t][6] + arc[i][t][5] + arc[i][t][27] == arc[i][t + 1][6] + arc[i][t + 1][7] + arc[i][t + 1][8], name=f"ArcBus3_{t}_{i}")
            model.addConstr(arc[i][t][9] + arc[i][t][19] + arc[i][t][29] == arc[i][t + 1][9] + arc[i][t + 1][10] + arc[i][t + 1][11], name=f"ArcBus4_{t}_{i}")
            model.addConstr(arc[i][t][12] + arc[i][t][1] + arc[i][t][17] + arc[i][t][21] == arc[i][t + 1][12] + arc[i][t + 1][13] + arc[i][t + 1][14] + arc[i][t + 1][15], name=f"ArcBus5_{t}_{i}")
            model.addConstr(arc[i][t][16] + arc[i][t][11] + arc[i][t][14] + arc[i][t][28] == arc[i][t + 1][16] + arc[i][t + 1][17] + arc[i][t + 1][18] + arc[i][t + 1][19], name=f"ArcBus6_{t}_{i}")
            model.addConstr(arc[i][t][20] + arc[i][t][2] + arc[i][t][4] + arc[i][t][13] + arc[i][t][26] == arc[i][t + 1][20] + arc[i][t + 1][21] + arc[i][t + 1][22] + arc[i][t + 1][23] + arc[i][t + 1][24], name=f"ArcBus7_{t}_{i}")
            model.addConstr(arc[i][t][25] + arc[i][t][8] + arc[i][t][10] + arc[i][t][18] + arc[i][t][23] == arc[i][t + 1][25] + arc[i][t + 1][26] + arc[i][t + 1][27] + arc[i][t + 1][28] + arc[i][t + 1][29], name=f"ArcBus8_{t}_{i}")


    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== Objective Function =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------

    # OBJECTIVE
    costSum = gb.LinExpr()  # Initialize the total cost expression
    TotalSUCost = gb.LinExpr()  # Initialize the total start-up and shut-down cost
    RENEWABLE_COST = 11.24


    # ----------------------------------------------------------
    # 1) TES Travel Cost  
    # ----------------------------------------------------------

    travelCost = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"travelCost_{ts}") for ts in range(nbHour)]

    TRAVEL_ARCS = [
            1, 3, 4, 5, 7, 8, 10, 11,
            13, 14, 15, 17, 18, 19, 21,
            22, 23, 24, 26, 27, 28, 29
        ]

    for r in range(nbHour):
        model.addConstr(travelCost[r] == 100 * gb.quicksum(arc[es][r][a] for es in range(nbES) for a in TRAVEL_ARCS),
            name=f"TravelCostConstr_{r}"
        )
        costSum += travelCost[r]

    # ----------------------------------------------------------
    # 2) ESS Charging / Discharging Cost (deterministic)
    # ----------------------------------------------------------
    for r in range(nbHour):
        costSum += gb.quicksum(ESCCost*qcharge[r][es] + ESDCost*qdischarge[r][es] for es in range(nbES))


    # ----------------------------------------------------------
    # 4) Generator Fuel + No-load Cost
    # ----------------------------------------------------------
    for r in range(nbHour):

        fuel_cost = (
            gb.quicksum(IncHeatRateC[g] * x2[r][g] for g in range(nbGen)) +
            gb.quicksum(IncHeatRateB[g] * x[r][g]  for g in range(nbGen)) +
            gb.quicksum(IncHeatRateA[g] * OnOff[r][g] for g in range(nbGen))
        )

        costSum += fuel_cost

    # # Compute costs for each scenario, hour, and generator
    # for s in range(nbScenarios):
    #     for r in range(nbHour):
    #         costSum += (1.0 / nbScenarios) * (gb.quicksum(IncHeatRateC[i] * x2[r][s][i] for i in range(nbGen)) +
    #                                 gb.quicksum(IncHeatRateB[i] * x[r][s][i] for i in range(nbGen)) +
    #                                 gb.quicksum(IncHeatRateA[i] * OnOff[r][s][i] for i in range(nbGen)))


    # ----------------------------------------------------------
    # 4) Startup + Shutdown Costs
    # ----------------------------------------------------------

    for r in range(nbHour):
        costSum += gb.quicksum(
            StartUpCost[r][i] + ShutDownCost[r][i]
            for i in range(nbGen)
        )

        # for i in range(nbGen):
        #     # Shut Down Cost constraint
        #     model.addConstr(ShutDownCost[r][s][i] >= 0, name=f"ShutDownCost_{r}_{s}_{i}")
        #     # Start Up Cost constraint
        #     model.addConstr(StartUpCost[r][s][i] >= 0, name=f"StartUpCost_{r}_{s}_{i}")
            
        #     # Accumulate startup and shutdown costs
        #     TotalSUCost += (1.0 / nbScenarios) * (StartUpCost[r][s][i] + ShutDownCost[r][s][i])

    # Add Total StartUp and ShutDown costs to the total cost
    # costSum += TotalSUCost


    # Set the Objective: Minimize the total cost
    model.setObjective(costSum, GRB.MINIMIZE)

    filtered_data_arc = ai["filtered_data_arc"]
    filtered_data_gen = ai["filtered_data_gen"]
    filtered_data_esc = ai["filtered_data_esc"]
    filtered_data_esd = ai["filtered_data_esd"]
    filtered_data_cc = ai["filtered_data_cc"]

    predictedOnOffFeasibilityIndex = ai["predictedOnOffFeasibilityIndex"]
    predictedArcFeasibilityIndex = ai["predictedArcFeasibilityIndex"]
    predictedESCOnOffFeasibilityIndex = ai["predictedESCOnOffFeasibilityIndex"]
    predictedESDOnOffFeasibilityIndex = ai["predictedESDOnOffFeasibilityIndex"]
    predictedCCFeasibilityIndex = ai["predictedCCFeasibilityIndex"]

    # Export Model as LP File if required
    if EXPORT_TO_LP == 1:
        model.write(lpFilePath)
        print("Model exported as LP file.")
        print("-----------------------------------------------------")

    
        
    # Apply AI Predictions
    if p == 100:
        model.reset()
            # Update the arc variables (one scenario per run)
        if ASSIST_MILP_WITH_ARC:
            for k in range(nbES):  # Loop through each energy storage unit (nbES)
                for ts in range(nbHour):  # Loop through each time step (nbHour - 1)
                    for i in range(ARC_PER_TS):  # Loop through each arc per time step
                        # Assign the predicted arc feasibility values to the model 
                        arc[k][ts][i].start = filtered_data_arc[scenario_idx, k , ts, i]

        # Update the On/Off variables (Generator status: on or off, one scenario)
        if ASSIST_MILP_WITH_ONOFF:
            for g in range(nbGen):  # Loop through each generator (nbGen)
                for t in range(nbHour):  # Loop through each time step (nbHour)
                    # Assign the predicted on/off status of the generator to the model (only one scenario in CC)
                    OnOff[t][g].start = filtered_data_gen[scenario_idx, g, t]

        # Update the ESC (Energy Storage Charging) On/Off variables (one scenario)
        if ASSIST_MILP_WITH_ESCONOFF:
            for ts in range(nbHour):  # Loop through each time step (nbHour - 1)
                for k in range(nbES):  # Loop through each energy storage unit (nbES)
                    # Assign the predicted ESC on/off values to the model (only one scenario in CC)
                    ESCOnOff[ts][k].start = filtered_data_esc[scenario_idx, k, ts]

        # Update the ESD (Energy Storage Discharging) On/Off variables (one scenario)
        if ASSIST_MILP_WITH_ESDONOFF:
            for ts in range(nbHour):  # Loop through each time step (nbHour - 1)
                for k in range(nbES):  # Loop through each energy storage unit (nbES)
                    # Assign the predicted ESD on/off values to the model (only one scenario in CC)
                    ESDOnOff[ts][k].start = filtered_data_esd[scenario_idx, k, ts]

        # Update the CC (Chance Constraint) satisfaction values (multiple scenarios)
        if ASSIST_MILP_WITH_CC:
            for s in range(S):  # Loop through each scenario (CC allows multiple scenarios)
                # Assign the predicted CC satisfaction values to the model (1 = satisfied, 0 = not satisfied)
                cc_sc[s].start = filtered_data_cc[scenario_idx, s]

    # elif p == 0:  # 100% problem size reduction

    #     # Fix Arc
    #     if ASSIST_MILP_WITH_ARC:
    #         for k in range(nbES):
    #             for ts in range(nbHour - 1):
    #                 for i in range(ARC_PER_TS):
    #                     model.addConstr(
    #                         arc[k][ts][i] == float(filtered_data_arc[scenario_idx, k, ts, i])
    #                     )

    #     # Fix OnOff
    #     if ASSIST_MILP_WITH_ONOFF:
    #         for g in range(nbGen):
    #             for t in range(nbHour):
    #                 model.addConstr(
    #                     OnOff[t][g] == float(filtered_data_gen[scenario_idx, g, t])
                    
    elif p == 200:  # HA-1: feasible = fix, infeasible = warm start
        model.reset()

        # OnOff
        if ASSIST_MILP_WITH_ONOFF:
            for g in range(nbGen):
                for t in range(nbHour):
                    if predictedOnOffFeasibilityIndex[scenario_idx][g] == 1:
                        model.addConstr(
                            OnOff[t][g] == float(filtered_data_gen[scenario_idx, g, t])
                        )
                    else:
                        OnOff[t][g].start = filtered_data_gen[scenario_idx, g, t]

        # Arc
        if ASSIST_MILP_WITH_ARC:
            for ts in range(nbHour):
                for k in range(nbES):
                    for i in range(ARC_PER_TS):
                        if predictedArcFeasibilityIndex[scenario_idx][k][ts] == 1:
                            model.addConstr(
                                arc[k][ts][i] == float(filtered_data_arc[scenario_idx, k, ts, i])
                            )
                        else:
                            arc[k][ts][i].start = filtered_data_arc[scenario_idx, k, ts, i]

        # ESCOnOff
        if ASSIST_MILP_WITH_ESCONOFF:
            for ts in range(nbHour):
                for k in range(nbES):
                    if predictedESCOnOffFeasibilityIndex[scenario_idx][k][ts] == 1:
                        model.addConstr(
                            ESCOnOff[ts][k] == float(filtered_data_esc[scenario_idx, k, ts])
                        )
                    else:
                        ESCOnOff[ts][k].start = filtered_data_esc[scenario_idx, k, ts]

        # ESDOnOff
        if ASSIST_MILP_WITH_ESDONOFF:
            for ts in range(nbHour):
                for k in range(nbES):
                    if predictedESDOnOffFeasibilityIndex[scenario_idx][k][ts] == 1:
                        model.addConstr(
                            ESDOnOff[ts][k] == float(filtered_data_esd[scenario_idx, k, ts])
                        )
                    else:
                        ESDOnOff[ts][k].start = filtered_data_esd[scenario_idx, k, ts]

        # CC labels
        if ASSIST_MILP_WITH_CC:
            for s in range(S):
                if predictedCCFeasibilityIndex[scenario_idx] == 1:
                    model.addConstr(
                        cc_sc[s] == float(filtered_data_cc[scenario_idx, s])
                    )
                else:
                    cc_sc[s].start = filtered_data_cc[scenario_idx, s]


    elif p == 300:  # HA-2: problem size reduction for feasible AI prediction, cold start for infeasible ones

        # -------------------------
        # OnOff
        # -------------------------
        if ASSIST_MILP_WITH_ONOFF:
            for g in range(nbGen):
                if predictedOnOffFeasibilityIndex[scenario_idx][g] == 1:
                    for t in range(nbHour):
                        model.addConstr(
                            OnOff[t][g] == float(filtered_data_gen[scenario_idx, g, t]))

        # -------------------------
        # Arc
        # -------------------------
        if ASSIST_MILP_WITH_ARC:
            for ts in range(nbHour):
                for k in range(nbES):
                    if predictedArcFeasibilityIndex[scenario_idx][k][ts] == 1:
                        for i in range(ARC_PER_TS):
                            model.addConstr(
                                arc[k][ts][i] == float(filtered_data_arc[scenario_idx, k, ts, i])
                    )

        # -------------------------
        # ESC On/Off
        # -------------------------
        if ASSIST_MILP_WITH_ESCONOFF:
            for ts in range(nbHour):
                for k in range(nbES):
                    if predictedESCOnOffFeasibilityIndex[scenario_idx][k][ts] == 1:
                        model.addConstr(
                            ESCOnOff[ts][k] == float(filtered_data_esc[scenario_idx, k, ts]))

        # -------------------------
        # ESD On/Off
        # -------------------------
        if ASSIST_MILP_WITH_ESDONOFF:
            for ts in range(nbHour):
                for k in range(nbES):
                    if predictedESDOnOffFeasibilityIndex[scenario_idx][k][ts] == 1:
                        model.addConstr(
                            ESDOnOff[ts][k] == float(filtered_data_esd[scenario_idx, k, ts]))

        # -------------------------
        # Chance constraint labels
        # -------------------------
        if ASSIST_MILP_WITH_CC:
            if predictedCCFeasibilityIndex[scenario_idx] == 1:
                for s in range(S):
                    model.addConstr(
                        cc_sc[s] == float(filtered_data_cc[scenario_idx, s]))
    
        # elif p == 200:  # Warm Start
    #     model.reset()  # Reset the model before setting warm start values

    #     # Update the On/Off variables (Generator status: on or off, one scenario)
    #     if ASSIST_MILP_WITH_ONOFF:
    #         Mismatch = []
    #         for t in range(nbHour):
    #             Mismatch.append([])  # List to track mismatches
    #             for g in range(nbGen):
    #                 # Binary variable: 1 if mismatch, 0 otherwise
    #                 var = model.addVar(vtype=GRB.BINARY, name=f"Mismatch_{t}_{g}")
    #                 Mismatch[t].append(var)

    #         # Apply warm start for all generators in the filtered data (only one row for CC)
    #         for g in range(nbGen):
    #             if predictedOnOffFeasibilityIndex[scenario_idx][g] == 1:  # Check feasibility for the first scenario (index 0)
    #                 for t in range(nbHour):
    #                     pred = filtered_data_gen[scenario_idx, g, t]  # Use the first row of the filtered data for warm start
    #                     onoff = OnOff[t][g]
    #                     mismatch = Mismatch[t][g]
                        
    #                     if pred == 1:
    #                         model.addConstr(mismatch >= onoff - 0)
    #                     else:
    #                         model.addConstr(mismatch >= 0 - onoff)
    #             else:
    #                 for t in range(nbHour):
    #                     OnOff[t][g].start = filtered_data_gen[scenario_idx, g, t]  # Use `.start` instead of setAttr

    #     # Update the Arc variables 
    #     if ASSIST_MILP_WITH_ARC:
    #         Mismatch = []
    #         for ts in range(nbHour - 1):
    #             Mismatch.append([])  # Track mismatch for each time step
    #             for i in range(ARC_PER_TS):
    #                 Mismatch[ts].append([])  # Track mismatch for each arc
    #                 for k in range(nbES):
    #                     # Binary variable: 1 if mismatch, 0 otherwise
    #                     var = model.addVar(vtype=GRB.BINARY, name=f"Mismatch_{ts}_{k}_{i}")
    #                     Mismatch[ts][i].append(var)

    #         # Apply warm start for all arcs in the filtered data (only one row for CC)
    #         for ts in range(nbHour - 1):
    #             for k in range(nbES):
    #                 for i in range(ARC_PER_TS):
    #                     if predictedArcFeasibilityIndex[scenario_idx][ts] == 1:  # Check feasibility for the first scenario (index 0)
    #                         pred = filtered_data_arc[scenario_idx, k , i, ts]
    #                         Arc = arc[k][ts][i]
    #                         mismatch = Mismatch[ts][i][k]
                            
    #                         if pred == 1:
    #                             model.addConstr(mismatch >= Arc - 0)
    #                         else:
    #                             model.addConstr(mismatch >= 0 - Arc)
    #                     else:
    #                         # For non-feasible cases, use warm start
    #                         arc[k][ts][i].start = filtered_data_arc[scenario_idx, k , i, ts]

    #     # Update the ESC On/Off variables (one scenario)
    #     if ASSIST_MILP_WITH_ESCONOFF:
    #         for ts in range(nbHour - 1):
    #             for k in range(nbES):
    #                 if predictedESCOnOffFeasibilityIndex[scenario_idx][ts] == 1:  # Check feasibility for the first scenario (index 0)
    #                     # If feasible, assign the value directly
    #                     model.addConstr(ESCOnOff[ts][k] == float(filtered_data_esc[scenario_idx, k, ts]))
    #                 else:
    #                     # For non-feasible cases, use warm start
    #                     ESCOnOff[ts][k].start = filtered_data_esc[scenario_idx, k, ts]

    #     # Update the ESD On/Off variables (one scenario)
    #     if ASSIST_MILP_WITH_ESDONOFF:
    #         for ts in range(nbHour - 1):
    #             for k in range(nbES):
    #                 if predictedESDOnOffFeasibilityIndex[scenario_idx][ts] == 1:  # Check feasibility for the first scenario (index 0)
    #                     # If feasible, assign the value directly
    #                     model.addConstr(ESDOnOff[ts][k] == float(filtered_data_esd[scenario_idx, k, ts]))
    #                 else:
    #                     # For non-feasible cases, use warm start
    #                     ESDOnOff[ts][k].start = filtered_data_esd[scenario_idx, k, ts]

    #     # Update the CC (Chance Constraint) satisfaction values (multiple scenarios)
    #     if ASSIST_MILP_WITH_CC:
    #         for s in range(S):
    #             if predictedCCFeasibilityIndex[scenario_idx] == 1:  # Check feasibility for the first scenario (index 0)
    #                 # If feasible, assign the value directly
    #                 model.addConstr(cc_sc[s] == filtered_data_cc[scenario_idx, s])
    #             else:
    #                 # For non-feasible cases, use warm start
    #                 cc_sc[s].start = filtered_data_cc[scenario_idx, s]
    # conflict refiner
    if INVOKE_CONFLICT_REFINER == 1:
        constraints = []
        for v in model.getVars():
            constraints.append(model.addConstr(v >= v.lb))
            constraints.append(model.addConstr(v <= v.ub))
        for c in model.getConstrs():
            constraints.append(c)

        model.optimize()
        
        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible, computing IIS...")
            model.computeIIS()
            model.write("conflict.ilp")

            # Print conflict details
            print("\n--- Infeasible Subsystem (IIS) ---")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"Constraint in conflict: {c.ConstrName}")

            for v in model.getVars():
                if v.IISLB:
                    print(f"Lower bound of variable {v.VarName} is in conflict")
                if v.IISUB:
                    print(f"Upper bound of variable {v.VarName} is in conflict")

            print("\nWrote conflict details to 'conflict.ilp':\n")

            with open("conflict.ilp", "r") as file:
                print(file.read())
        else:
            print("No conflicts found.")


    # Set Gurobi parameters
    model.setParam('MIPGap', 0.01)  # Set the optimal gap to 0.01%
    model.setParam('Threads', 1)  # Using one thread or core of CPU
    model.setParam('TimeLimit', 7200)  # Uncomment if you want to set a time limit in seconds (3600s)
    model.setParam("Presolve", 2)  # Aggressive presolve
    model.setParam("Heuristics", 0.5)  # Default is 0.05; this may help

    # Start solving
    print("Hooray!!! Start solving...")
    print("-----------------------------------------------------")
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    solve_time = end_time - start_time

    print(f"Solution status: {model.Status}")
    print("--------------------------------------------")
    SolutionTime = solve_time
    OptimisedCost = model.ObjVal if model.SolCount > 0 else None

    
    if model.SolCount > 0:
        print(f"Gurobi Obj = {model.ObjVal}")
        print("--------------------------------------------")
    else:
        print("No feasible solution stored by Gurobi.")
        print("--------------------------------------------")

    print(f"Time to solve = {solve_time:.6f} sec")
    print("--------------------------------------------")

    

       
    # Check the solution status
    status = model.Status
    print("Solver status code:", status)

    confidence_level = 1 - epsilon
    actual_final_idx = int(final_idx[scenario_idx]) if hasattr(final_idx, "__len__") else int(final_idx)

    if status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:

        if IMPORT_TO_CSV_ONOFF:
            row = [actual_final_idx, scenario_idx, int(nbES), float(confidence_level), int(p)]
            for g in range(nbGen):
                for h in range(nbHour):
                    val = OnOff[h][g].X
                    row.append(0.0 if val == -0.0 else val)
            with open(milpOptimumOnOffFileName, "a", newline="") as out:
                csv.writer(out).writerow(row)

        if IMPORT_TO_CSV_ARC:
            row = [actual_final_idx, scenario_idx, int(nbES), float(confidence_level), int(p)]
            for es in range(nbES):
                for h in range(nbHour):
                    for n in range(ARC_PER_TS):
                        val = arc[es][h][n].X
                        row.append(0.0 if val == -0.0 else val)
            with open(milpOptimumArcFileName, "a", newline="") as out:
                csv.writer(out).writerow(row)

        if IMPORT_TO_CSV_ESCONOFF:
            row = [actual_final_idx, scenario_idx, int(nbES), float(confidence_level), int(p)]
            for es in range(nbES):
                for h in range(nbHour):
                    val = ESCOnOff[h][es].X
                    row.append(0.0 if val == -0.0 else val)
            with open(milpOptimumESCOnOffFileName, "a", newline="") as out:
                csv.writer(out).writerow(row)
        
        if IMPORT_TO_CSV_ESDONOFF:
            row = [actual_final_idx, scenario_idx, int(nbES), float(confidence_level), int(p)]
            for es in range(nbES):
                for h in range(nbHour):
                    val = ESDOnOff[h][es].X
                    row.append(0.0 if val == -0.0 else val)
            with open(milpOptimumESDOnOffFileName, "a", newline="") as out:
                csv.writer(out).writerow(row)

        if IMPORT_TO_CSV_CC:
            row = [actual_final_idx, scenario_idx, int(nbES), float(confidence_level), int(p)]
            for s in range(nbScenarios):
                val = cc_sc[s].X
                row.append(0.0 if val == -0.0 else val)
            with open(milpOptimumCCFileName, "a", newline="") as out:
                csv.writer(out).writerow(row)

        if IMPORT_TO_CSV_OPTIMISED_COST_AND_SOLUTION_TIME:
            pd.DataFrame([{
                "scenario_idx": scenario_idx,
                "p": p,
                "nbES": nbES,
                "confidence_level": confidence_level,
                "OptimisedCost": OptimisedCost,
                "SolutionTime": SolutionTime}]).to_csv(
                out_OPTIMISED_COST_AND_SOLUTION_TIME,
                mode='a',
                header=not os.path.exists(out_OPTIMISED_COST_AND_SOLUTION_TIME),
                index=False
            )
    # elif status == GRB.INFEASIBLE:
    #     print("Model is infeasible. Computing IIS...")
    #     model.computeIIS()
    #     model.write("conflict.ilp")

    #     print("\n--- IIS constraints ---")
    #     for c in model.getConstrs():
    #         if c.IISConstr:
    #             print(c.ConstrName)

    #     print("\n--- IIS variable bounds ---")
    #     for v in model.getVars():
    #         if v.IISLB:
    #             print(f"LB conflict: {v.VarName}")
    #         if v.IISUB:
    #             print(f"UB conflict: {v.VarName}")   
    
    else:
        print("No optimal solution found.")
        totalInfeasibleScenarios = totalInfeasibleScenarios + 1
        print("Model is infeasible.")

    model.dispose()

# # --------------------------------------------------
# choose what you want to test
# --------------------------------------------------
CONFIDENCE_VALUES = [0.60, 0.70, 0.80, 0.90, 0.95]     # example
NBES_VALUES = [1, 3, 5, 7, 9]                    # example
P_VALUES = [100, 200, 300]          # your modes

NB_GEN = 54
NB_HOUR = 24

# --------------------------------------------------
# run everything
# --------------------------------------------------
for target_conf in CONFIDENCE_VALUES:
    for target_nBES in NBES_VALUES:

        print("=" * 80)
        print(f"Filtering for confidence = {target_conf}, nbES = {target_nBES}")

        ai = load_and_filter_ai_data(
            target_conf=target_conf,
            target_nBES=target_nBES,
            nbGen=NB_GEN,
            nbHour=NB_HOUR
        )

        final_idx = ai["final_idx"]
        n_cases = len(final_idx)

        print(f"Number of matched cases = {n_cases}")

        if n_cases == 0:
            print("No matching cases found. Skipping.")
            continue

        for p in P_VALUES:
            print("-" * 80)
            print(f"Running p = {p} for confidence = {target_conf}, nbES = {target_nBES}")

            for scenario_idx in range(n_cases):
                print(
                    f"Run {scenario_idx + 1}/{n_cases} | "
                    f"scenario_idx = {scenario_idx} | "
                    f"actual_final_idx = {final_idx[scenario_idx]} | "
                    f"p = {p} | nbES = {target_nBES} | confidence = {target_conf}"
                )

                build_and_solve_one_case(
                    final_idx=final_idx,
                    scenario_idx=scenario_idx,
                    p=p,
                    ai=ai,
                    target_conf=target_conf,
                    target_nBES=target_nBES
                )

print("All runs completed.")
