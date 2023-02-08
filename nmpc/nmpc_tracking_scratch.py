###
import os
import sys
sys.path.append('../')
sys.path.append('../../')
from copy import deepcopy
import time
import pandas as pd
import numpy as np
from scipy.io import savemat

import pyomo.environ as pyo
from pyomo.environ import (ConcreteModel, Var, Constraint, Param, Expression,
                           value, Objective, Suffix)
from pyomo.opt import ProblemFormat
from pyomo.dae.flatten import flatten_dae_components, generate_sliced_components
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import idaes
import idaes.core.util.scaling as iscale
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import petsc
from idaes.core.solvers import use_idaes_solver_configuration_defaults
import idaes.logger as idaeslog
idaeslog.getLogger("idaes.core.util.scaling").setLevel(idaeslog.ERROR)
import idaes.core.util.model_serializer as ms
from pyomo.dae import ContinuousSet, DerivativeVar
from soec_dynamic_flowsheet_mk2 import SoecStandaloneFlowsheet
from idaes.models.control.controller import (
    ControllerType,
    ControllerMVBoundType,
    ControllerAntiwindupType,
)

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
###


###
global alias_dict
alias_dict = {
    "fs.h2_mass_production": "h2_production_rate",
    "fs.soc_module.potential_cell": "potential",
    "fs.soc_module.fuel_outlet_mole_frac_comp_H2": "soc_fuel_outlet_mole_frac_comp_H2",
    "fs.makeup_mix._flow_mol_makeup_ref": "makeup_feed_rate",
    "fs.sweep_blower._flow_mol_inlet_ref": "sweep_feed_rate",
    "fs.feed_heater.electric_heat_duty": "feed_heater_duty",
    "fs.feed_heater._temperature_outlet_ref": "feed_heater_outlet_temperature",
    "fs.soc_module._temperature_fuel_outlet_ref": "fuel_outlet_temperature",
    "fs.sweep_heater.electric_heat_duty": "sweep_heater_duty",
    "fs.sweep_heater._temperature_outlet_ref": "sweep_heater_outlet_temperature",
    "fs.soc_module._temperature_oxygen_outlet_ref": "sweep_outlet_temperature",
    #TODO: stack_core_temperature references only apply in SOEC mode, not sure how to implement.
    "fs.stack_core_temperature": "stack_core_temperature",
    "fs.feed_recycle_split.recycle_ratio": "fuel_recycle_ratio",
    "fs.sweep_recycle_split.recycle_ratio": "sweep_recycle_ratio",
    # "fs.sweep_recycle_split.mixed_state[0].mole_frac_comp['O2']": "oxygen_out",
    # "fs.feed_recycle_mix.mixed_state[0].mole_frac_comp['H2']": "hydrogen_in",
    "fs.condenser_split.recycle_ratio": "vgr_recycle_ratio",
    "fs.condenser_flash.heat_duty": "condenser_heat_duty",
    "fs.makeup_mix.makeup_mole_frac_comp_H2": "makeup_mole_frac_comp_H2",
    "fs.makeup_mix.makeup_mole_frac_comp_H2O": "makeup_mole_frac_comp_H2O",
}

# Set up time discretization.
pred_step_num = 5
t_step = 150.0 #s
pred_horizon = pred_step_num*t_step

t_start = 0.5*60*60
t_ramp = 1*60*60
t_end = 1*60*60

sim_horizon = t_start + t_ramp + t_end + t_ramp + t_end
critical_times = [np.array([t_start,t_start])/3600,
                  np.array([t_start+t_ramp,t_start+t_ramp])/3600,
                  np.array([t_start+t_ramp+t_end,t_start+t_ramp+t_end])/3600,
                  np.array([t_start+t_ramp+t_end+t_ramp,t_start+t_ramp+t_end+t_ramp])/3600]
sim_nfe = int((sim_horizon)/t_step)
sim_time_set = np.linspace(0, sim_nfe*t_step, sim_nfe+1)
    
iter_count = 0

global t_base

up_then_down = False
###


###
#Utility functions, defined but not run with the script by default.
def check_scaling(m, large=1e3, small=1e-3):
    jac, nlp = iscale.get_jacobian(m, scaled=True)
    # jac_csc = jac.tocsc()
    # djac = jac.toarray()
    print("Badly scaled variables:")
    for i in iscale.extreme_jacobian_columns(
        jac=jac, nlp=nlp, large=large, small=small):
        print(f"    {i[0]:.2e}, [{i[1]}]")
    print("\n\n"+"Badly scaled constraints:")
    for i in iscale.extreme_jacobian_rows(
        jac=jac, nlp=nlp, large=large, small=small):
        print(f"    {i[0]:.2e}, [{i[1]}]")
    
    # jac_conditioning = {idx: (min(row), max(row)) for idx, row in enumerate(djac)}
    # solcons = nlp.get_pyomo_constraints()
    # solvars = nlp.get_pyomo_variables()
    
    # return jac, jac_conditioning, solcons, solvars

def check_DoF(m):
    print(f'DoF = {idaes.core.util.model_statistics.degrees_of_freedom(m):d}')
###


###
#Function to create setpoint trajectories.
def make_var_trajs(mode_, tset_, ts_, tr_, te_):
    df = pd.read_csv(
        "./../../soec_flowsheet_operating_conditions.csv",
        index_col=0,
    )
    
    def make_one_var_traj(alias, mode, tset, ts, tr, te):
        slope = (df[alias]["maximum_H2"] - df[alias]["minimum_H2"])/tr
        
        #Mode 1: up-down
        #Mode 2: down-up
        mode = int(mode)
        
        var_target = {t:0.0 for t in tset}
        
        for t in tset:
            if t < ts:
                if mode == 1:
                    var_target[t] = df[alias]["minimum_H2"]
                elif mode == 2:
                    var_target[t] = df[alias]["maximum_H2"]
                else:
                    raise NotImplementedError("Invalid mode.")
            elif ts <= t < ts + tr:
                if mode == 1:
                    var_target[t] = df[alias]["minimum_H2"] + slope*(t - ts)
                elif mode == 2:
                    var_target[t] = df[alias]["maximum_H2"] - slope*(t - ts)
                else:
                    raise NotImplementedError("Invalid mode.")
            elif ts + tr <= t < ts + tr + te:
                if mode == 1:
                    var_target[t] = df[alias]["maximum_H2"]
                elif mode == 2:
                    var_target[t] = df[alias]["minimum_H2"]
                else:
                    raise NotImplementedError("Invalid mode.")
            elif ts + tr + te <= t < ts + tr + te + tr:
                if mode == 1:
                    var_target[t] = df[alias]["maximum_H2"] - slope*(t - ts - tr - te)
                elif mode == 2:
                    var_target[t] = df[alias]["minimum_H2"] + slope*(t - ts - tr - te)
                else:
                    raise NotImplementedError("Invalid mode.")
            else:
                if mode == 1:
                    var_target[t] = df[alias]["minimum_H2"]
                elif mode == 2:
                    var_target[t] = df[alias]["maximum_H2"]
                else:
                    raise NotImplementedError("Invalid mode.")
        
        return var_target
    
    var_trajs = {alias:{} for alias in alias_dict.values()}
    for alias_ in var_trajs:
        var_trajs[alias_] = make_one_var_traj(alias_, mode_, tset_, ts_, tr_, te_)
    
    return var_trajs
###


###
def get_CVs(m):
    return [
        m.fs.soc_module.fuel_outlet_mole_frac_comp_H2,
        m.fs.feed_heater._temperature_outlet_ref,
        m.fs.soc_module._temperature_fuel_outlet_ref,
        m.fs.sweep_heater._temperature_outlet_ref,
        m.fs.soc_module._temperature_oxygen_outlet_ref,
        m.fs.stack_core_temperature,
    ]


def get_manipulated_variables(m):
    return [
        m.fs.soc_module.potential_cell,
        m.fs.makeup_mix._flow_mol_makeup_ref,
        m.fs.sweep_blower._flow_mol_inlet_ref,
        m.fs.condenser_split.recycle_ratio,
        m.fs.condenser_flash.heat_duty,
        m.fs.feed_heater.electric_heat_duty,
        m.fs.sweep_heater.electric_heat_duty,
        m.fs.feed_recycle_split.recycle_ratio,
        m.fs.sweep_recycle_split.recycle_ratio,
        m.fs.makeup_mix.makeup_mole_frac_comp_H2,
        m.fs.makeup_mix.makeup_mole_frac_comp_H2O,
    ]

def create_model(tset, nfe, plant=True, from_min=True):
    m = ConcreteModel()
    m.fs = SoecStandaloneFlowsheet(
        dynamic=True,
        time_set=tset,
        time_units=pyo.units.s,
        thin_electrolyte_and_oxygen_electrode=True,
        has_gas_holdup=False,
        include_interconnect=True,
    )
    
    iscale.calculate_scaling_factors(m)
    
    pyo.TransformationFactory("dae.finite_difference").apply_to(
        m.fs, nfe=nfe, wrt=m.fs.time, scheme="BACKWARD"
    )

    # Initialize model at minimum/maximum production rate.
    if from_min:
        ms.from_json(m,
                     fname="../../min_production.json.gz",
                     wts=ms.StoreSpec.value())
    else:
        ms.from_json(m,
                     fname="../../max_production.json.gz",
                     wts=ms.StoreSpec.value())

    # Copy initial conditions to rest of model for initialization
    if not plant:
        _, time_vars = flatten_dae_components(m, m.fs.time, pyo.Var)
        for t in m.fs.time:
            for v in time_vars:
                if not v[t].fixed:
                    if v[m.fs.time.first()].value is None:
                        v[t].set_value(0.0)
                    else:
                        v[t].set_value(pyo.value(v[m.fs.time.first()]))
    
    # Fix initial conditions
    m.fs.fix_initial_conditions()
    
    for v in get_manipulated_variables(m):
        if plant:
            v[:].fix(v[0].value)
        else:
            v[:].set_value(v[0].value)
            v[:].unfix()

    if plant:
        assert degrees_of_freedom(m) == 0

    return m

def petsc_initialize(m):
    idaeslog.solver_log.tee = True
    return petsc.petsc_dae_by_time_element(
        m,
        time=m.fs.time,
        keepfiles=True,
        symbolic_solver_labels=True,
        ts_options={
            "--ts_type": "beuler",
            "--ts_dt": 10,
            "--ts_rtol": 1e-03,
            "--ts_adapt_dt_min": 1e-06,
            "--ksp_rtol": 1e-10,
            "--snes_type": "newtontr",
            "--ts_monitor": "",
            "--ts_save_trajectory": 1,
            "--ts_trajectory_type": "visualization",
            "--ts_max_snes_failures": 1000
        },
        skip_initial=False,
        initial_solver="ipopt",
    )

def get_state_vars(m):
    time_derivative_vars = [
        var for var in m.component_objects(Var)
        if isinstance(var, DerivativeVar)
    ]
    state_vars = [
        dv.get_state_var() for dv in time_derivative_vars
        if m.fs.time in dv.index_set().subsets()
    ]

    return state_vars

def set_initial_conditions(target_model, source_model):

    def set_state_var_ics(state_var_target, state_var_source):
        for (t, *idxs), v in state_var_target.items():
            if t == target_model.fs.time.first():
                tN_index = tuple([source_model.fs.time.last(), *idxs])
                v.set_value(value(state_var_source[tN_index]))
        return None

    for state_var_target, state_var_source in zip(get_state_vars(target_model),
                                                  get_state_vars(source_model)):
        set_state_var_ics(state_var_target, state_var_source)
    
    return None

def apply_control_actions(controller_model, plant_model):
    # for c, p in zip(get_manipulated_variables(controller_model),
    #                 get_manipulated_variables(plant_model)):
    #     p[:].fix(value(c[controller_model.fs.time.first()]))
    controller_MVs = get_manipulated_variables(controller_model)
    plant_MVs = get_manipulated_variables(plant_model)
    # controller_MVs = controller.fs.manipulated_variables
    # plant_MVs = plant.fs.manipulated_variables

    for c, p in zip(controller_MVs, plant_MVs):
        t0 = controller_model.fs.time.first()
        t1 = controller_model.fs.time.next(t0)
        # import pdb; pdb.set_trace()
        for t, v in c.items():
            if t == t1:
                control_input = value(c[t])
                # t1_index = tuple([t, *idxs])
                p[t].set_value(control_input)
                p[t].fix()
                # p[:].set_value(control_input_0)
                # p[:].fix()
    
    return None

def create_obj_expr(m):
    h2_target = var_targets['h2_production_rate']
    potential = var_targets['potential']
    soc_fuel_outlet_mole_frac_comp_H2 = var_targets['soc_fuel_outlet_mole_frac_comp_H2']
    makeup_feed_rate = var_targets['makeup_feed_rate']
    sweep_feed_rate = var_targets['sweep_feed_rate']
    feed_heater_duty = var_targets['feed_heater_duty']
    feed_heater_outlet_temperature = var_targets['feed_heater_outlet_temperature']
    fuel_outlet_temperature = var_targets['fuel_outlet_temperature']
    sweep_heater_duty = var_targets['sweep_heater_duty']
    sweep_heater_outlet_temperature = var_targets['sweep_heater_outlet_temperature']
    sweep_outlet_temperature = var_targets['sweep_outlet_temperature']
    stack_core_temperature = var_targets['stack_core_temperature']
    fuel_recycle_ratio = var_targets['fuel_recycle_ratio']
    sweep_recycle_ratio = var_targets['sweep_recycle_ratio']
    vgr_recycle_ratio = var_targets['vgr_recycle_ratio']
    condenser_heat_duty = var_targets['condenser_heat_duty']
    makeup_mole_frac_comp_H2 = var_targets['makeup_mole_frac_comp_H2']
    makeup_mole_frac_comp_H2O = var_targets['makeup_mole_frac_comp_H2O']
    
    expr = 0
    
    expr += 1e+01 * sum((m.fs.h2_mass_production[t] - h2_target[t_base + t])**2
                        for t in m.fs.time if t != m.fs.time.first())
    
    # Penalties on manipulated variable deviations
    mv_multiplier = 1e-03
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.makeup_mix.makeup.flow_mol[t]
          - makeup_feed_rate[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.sweep_blower.inlet.flow_mol[t]
          - sweep_feed_rate[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e+00 * sum(
        (m.fs.soc_module.potential_cell[t]
          - potential[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e+01 * sum(
        (m.fs.feed_recycle_split.recycle_ratio[t]
          - fuel_recycle_ratio[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e+01 * sum(
        (m.fs.sweep_recycle_split.recycle_ratio[t]
          - sweep_recycle_ratio[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e-06 * sum(
        (m.fs.feed_heater.electric_heat_duty[t]
          - feed_heater_duty[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first()) * 1e-2
    expr += mv_multiplier * 1e-07 * sum(
        (m.fs.sweep_heater.electric_heat_duty[t]
          - sweep_heater_duty[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first()) * 1e-3
    expr += mv_multiplier * 1e+01 * sum(
        (m.fs.condenser_split.recycle_ratio[t]
          - vgr_recycle_ratio[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e-07 * sum(
        (m.fs.condenser_flash.heat_duty[t]
          - condenser_heat_duty[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first()) * 1e-3
    expr += mv_multiplier * 1e+01 * sum(
        (m.fs.makeup_mix.makeup_mole_frac_comp_H2[t]
          - makeup_mole_frac_comp_H2[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e+00 * sum(
        (m.fs.makeup_mix.makeup_mole_frac_comp_H2O[t]
          - makeup_mole_frac_comp_H2O[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())

    expr += mv_multiplier * 1e-12 * sum((m.fs.feed_heater.electric_heat_duty[t] -
                              m.fs.feed_heater.electric_heat_duty[m.fs.time.prev(t)])**2
                            for t in m.fs.time if t != m.fs.time.first())
    expr += mv_multiplier * 1e-14 * sum((m.fs.sweep_heater.electric_heat_duty[t] -
                              m.fs.sweep_heater.electric_heat_duty[m.fs.time.prev(t)])**2
                            for t in m.fs.time if t != m.fs.time.first())
    expr += mv_multiplier * 1e-14 * sum((m.fs.condenser_flash.heat_duty[t] -
                              m.fs.condenser_flash.heat_duty[m.fs.time.prev(t)])**2
                            for t in m.fs.time if t != m.fs.time.first())

    expr += mv_multiplier * 1e+00 * sum(
        (m.fs.soc_module.fuel_outlet_mole_frac_comp_H2[t]
          - soc_fuel_outlet_mole_frac_comp_H2[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.feed_heater.outlet.temperature[t]
          - feed_heater_outlet_temperature[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.sweep_heater.outlet.temperature[t]
          - sweep_heater_outlet_temperature[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.soc_module.fuel_outlet.temperature[t]
          - fuel_outlet_temperature[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.soc_module.oxygen_outlet.temperature[t]
          - sweep_outlet_temperature[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.stack_core_temperature[t]
          - stack_core_temperature[t_base + t])**2 for t in m.fs.time
        if t != m.fs.time.first())
    
    return expr

def apply_custom_scaling(m):
    sf = 410
    for _, c in m.fs.condenser_flash.control_volume.enthalpy_balances.items():
        sf_old = iscale.get_constraint_transform_applied_scaling_factor(c)
        sf_new = sf_old/sf
        iscale.constraint_scaling_transform(c, sf_new, overwrite=True)
    
    return None

def shift_dae_vars(m, dt):
    seen = set()
    
    _, dae_vars = flatten_dae_components(m, m.fs.time, pyo.Var)
    
    for var in dae_vars:
        if id(var[m.fs.time.first()]) in seen:
            continue
        else:
            seen.add(id(var[m.fs.time.first()]))
        for t in m.fs.time:
            idx = m.fs.time.find_nearest_index(t + dt)
            if idx is None:
                # ts is outside the controller's horizon
                var[t].set_value(var[m.fs.time.last()].value)
            else:
                var[t].set_value(var[m.fs.time.at(idx)].value)
    return None
###


###
# Set up ipopt.
use_idaes_solver_configuration_defaults()
idaes.cfg.ipopt.options.nlp_scaling_method = "user-scaling"
idaes.cfg.ipopt["options"]["linear_solver"] = "ma57"
idaes.cfg.ipopt.options.OF_ma57_automatic_scaling = "yes"
idaes.cfg.ipopt["options"]["max_iter"] = 250
idaes.cfg.ipopt["options"]["halt_on_ampl_error"] = "no"
idaes.cfg.ipopt["options"]["bound_relax_factor"] = 1e-08
idaes.cfg.ipopt["options"]["bound_push"] = 1e-06
idaes.cfg.ipopt["options"]["mu_init"] = 1e-05
idaes.cfg.ipopt["options"]["tol"] = 1e-08
solver = pyo.SolverFactory("ipopt")
###


###
#Initialize models from steady-state files.
print("Building plant model...")
plant = create_model(np.linspace(0,t_step,2), 1, from_min=up_then_down)
plant.name = "Plant"
print("Building controller model...")
olnmpc = create_model(np.linspace(0,pred_horizon,pred_step_num+1), pred_step_num, plant=False, from_min=up_then_down)
olnmpc.name = "Controller"

#Scale models.
apply_custom_scaling(plant)
apply_custom_scaling(olnmpc)
iscale.scale_time_discretization_equations(plant, plant.fs.time, 1/t_step)
iscale.scale_time_discretization_equations(olnmpc, olnmpc.fs.time, 1/t_step)

#Generate initial plant model results.
print('Solving initial plant model...\n')
petsc_initialize(plant)
solver.solve(plant, tee=True)

def make_states_dict(m):
    states_dict = {c.name: {} for c in get_state_vars(m)}
    for c in get_state_vars(m):
        for (t, *idxs), _ in c.items():
            # Save the last state
            if t == m.fs.time.last():
                states_dict[c.name][tuple(idxs)] = []
    return states_dict
def save_states(m, states_dict):
    for c in get_state_vars(m):
        for (t, *idxs), v in c.items():
            if t == m.fs.time.last():
                states_dict[c.name][tuple(idxs)].append(value(v))
    return None
states_dict = make_states_dict(plant)
save_states(plant, states_dict)

#Loosen ipopt tolerances before controller loop.
idaes.cfg.ipopt["options"]["tol"] = 1e-04
idaes.cfg.ipopt["options"]["constr_viol_tol"] = 1e-04
idaes.cfg.ipopt["options"]["acceptable_tol"] = 1e-04
idaes.cfg.ipopt["options"]["dual_inf_tol"] = 1e+02
solver = pyo.SolverFactory("ipopt")

global h2_production_rate
h2_production_rate = []
def save_h2_production_rate(m):
    h2_production_rate.append(value(m.fs.h2_mass_production[m.fs.time.last()]))
    return None

objective = []

global controls_dict
controls_dict = {c.name: [] for c in get_manipulated_variables(olnmpc)}
def save_controls(m, controls_dict):
    for c in get_manipulated_variables(m):
        t0 = m.fs.time.first()
        t1 = m.fs.time.next(t0)
        controls_dict[c.name].append(value(c[t1]))
    return None

global CVs_dict
CVs_dict = {c.name: [] for c in get_CVs(plant)}
def save_CVs(m, CVs_dict):
    for c in get_CVs(m):
        CVs_dict[c.name].append(value(c[m.fs.time.last()]))
    return None
###


###
#Create variable setpoint trajectories.
global var_targets
var_targets = make_var_trajs(2, sim_time_set, t_start, t_ramp, t_end)

timer_start = time.time()

for iter_count, t in enumerate(sim_time_set):
    t_base = t
    
    # Set up the control problem
    # Fix initial conditions for states
    set_initial_conditions(target_model=olnmpc, source_model=plant)
    olnmpc.fs.fix_initial_conditions()
    
    if hasattr(olnmpc, 'obj'):
        olnmpc.del_component('obj')
    olnmpc.obj = pyo.Objective(rule=create_obj_expr, sense=pyo.minimize)
    
    print(f'\nSolving {olnmpc.name}, iteration {iter_count}...\n')
    
    solver.solve(olnmpc, tee=True, load_solutions=True)
    
    objective.append(value(olnmpc.obj))
    save_controls(olnmpc, controls_dict)
    
    # Set up Plant model
    set_initial_conditions(target_model=plant, source_model=plant)
    plant.fs.fix_initial_conditions()
    apply_control_actions(controller_model=olnmpc, plant_model=plant)
    assert degrees_of_freedom(plant) == 0

    print(f'\nSolving {plant.name}, iteration {iter_count}...\n')
    
    solver.solve(plant.fs, tee=True, load_solutions=True)
    
    save_CVs(plant, CVs_dict)
    save_states(plant, states_dict)
    save_h2_production_rate(plant)
    
    iter_count += 1

timer_stop = time.time()

print(f'Elapsed time, in seconds: {timer_stop-timer_start:.3f}')
###


def make_plots():
    for i in controls_dict.keys():
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(controls_dict[i],'b-')
        for j in alias_dict.keys():
            if i == j:
                ax.plot(var_targets[alias_dict[j]].values(),'r--')
                plt.title(alias_dict[j])
        plt.show()
    
    for i in CVs_dict.keys():
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(CVs_dict[i],'b-')
        for j in alias_dict.keys():
            if i == j:
                ax.plot(var_targets[alias_dict[j]].values(),'r--')
                fig.suptitle(alias_dict[j])
        plt.show()
    
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(h2_production_rate, 'b-')
    ax.plot(var_targets[alias_dict['fs.h2_mass_production']].values(), 'r--')
    plt.title('h2 production rate')
    
    return None



