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
    "fs.stack_core_temperature": "stack_core_temperature",
    "fs.feed_recycle_split.recycle_ratio": "fuel_recycle_ratio",
    "fs.sweep_recycle_split.recycle_ratio": "sweep_recycle_ratio",
    "fs.sweep_recycle_split.mixed_state.mole_frac_comp[O2]": "oxygen_out",
    "fs.feed_recycle_mix.mixed_state.mole_frac_comp[H2]": "hydrogen_in",
    "fs.condenser_split.recycle_ratio": "vgr_recycle_ratio",
    # "fs.condenser_flash.heat_duty": "condenser_heat_duty",
    "fs.condenser_flash.vap_outlet.temperature": "condenser_hot_outlet_temperature",
    "fs.makeup_mix.makeup_mole_frac_comp_H2": "makeup_mole_frac_comp_H2",
    "fs.makeup_mix.makeup_mole_frac_comp_H2O": "makeup_mole_frac_comp_H2O",
}

# Set up time discretization.
pred_step_num = 5
t_step = 150.0 #s
pred_horizon = pred_step_num*t_step

t_start = 0.5*60*60
t_ramp = 0.5*60*60
t_settle = 2*60*60
t_end = 1*60*60

sim_horizon = t_start + t_ramp + t_settle + t_ramp + t_end
critical_times = [np.array([t_start,t_start])/3600,
                  np.array([t_start+t_ramp,t_start+t_ramp])/3600,
                  np.array([t_start+t_ramp+t_settle,t_start+t_ramp+t_settle])/3600,
                  np.array([t_start+t_ramp+t_settle+t_ramp,t_start+t_ramp+t_settle+t_ramp])/3600]
sim_nfe = int((sim_horizon)/t_step)
sim_time_set = np.linspace(0, sim_nfe*t_step, sim_nfe+1)
traj_time_set = np.linspace(0, sim_nfe*t_step+pred_horizon, sim_nfe+pred_step_num+1)
    
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
def make_var_trajs(mode_, tset_, ts_, tr_, tl_, te_):
    df = pd.read_csv(
        "./../../soec_flowsheet_operating_conditions.csv",
        index_col=0,
    )
    
    def make_one_var_traj(alias, mode, tset, ts, tr, tl, te):
        mode1 = 'maximum_H2'
        mode2 = 'power'
        # mode1 = 'maximum_H2'
        # mode2 = 'minimum_H2'
        # mode1 = 'minimum_H2'
        # mode2 = 'maximum_H2'
        slope = (df[alias][mode1] - df[alias][mode2])/tr
        
        #Mode 1: up-down
        #Mode 2: down-up
        mode = int(mode)
        
        var_target = {t:0.0 for t in tset}
        
        for t in tset:
            if t < ts:
                if mode == 1:
                    var_target[t] = df[alias]["minimum_H2"]
                elif mode == 2:
                    var_target[t] = df[alias][mode1]
                else:
                    raise NotImplementedError("Invalid mode.")
            elif ts <= t < ts + tr:
                if mode == 1:
                    var_target[t] = df[alias]["minimum_H2"] + slope*(t - ts)
                elif mode == 2:
                    var_target[t] = df[alias][mode1] - slope*(t - ts)
                else:
                    raise NotImplementedError("Invalid mode.")
            elif ts + tr <= t < ts + tr + tl:
                if mode == 1:
                    var_target[t] = df[alias]["maximum_H2"]
                elif mode == 2:
                    var_target[t] = df[alias][mode2]
                else:
                    raise NotImplementedError("Invalid mode.")
            elif ts + tr + tl <= t < ts + tr + tl + tr:
                if mode == 1:
                    var_target[t] = df[alias]["maximum_H2"] - slope*(t - ts - tr - tl)
                elif mode == 2:
                    var_target[t] = df[alias][mode2] + slope*(t - ts - tr - tl)
                else:
                    raise NotImplementedError("Invalid mode.")
            else:
                if mode == 1:
                    var_target[t] = df[alias]["minimum_H2"]
                elif mode == 2:
                    var_target[t] = df[alias][mode1]
                else:
                    raise NotImplementedError("Invalid mode.")
        
        return var_target
    
    var_trajs = {alias:{} for alias in alias_dict.values()}
    for alias_ in var_trajs:
        var_trajs[alias_] = make_one_var_traj(alias_, mode_, tset_, ts_, tr_, tl_, te_)
    
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
        # m.fs.condenser_flash.heat_duty,
        m.fs.condenser_flash.vap_outlet.temperature,
        m.fs.feed_heater.electric_heat_duty,
        m.fs.sweep_heater.electric_heat_duty,
        m.fs.feed_recycle_split.recycle_ratio,
        m.fs.sweep_recycle_split.recycle_ratio,
        m.fs.makeup_mix.makeup_mole_frac_comp_H2,
        m.fs.makeup_mix.makeup_mole_frac_comp_H2O,
    ]

def set_indexed_variable_bounds(var, bounds):
    for idx, subvar in var.items():
        subvar.bounds = bounds

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
    
    # soec = m.fs.soc_module.solid_oxide_cell
    
    m.fs.p = pyo.Var(m.fs.time,
                      initialize=0,
                      domain=pyo.NonNegativeReals)
    m.fs.n = pyo.Var(m.fs.time,
                      initialize=0,
                      domain=pyo.NonNegativeReals)
    m.fs.q = pyo.Var(m.fs.time,
                      initialize=0,
                      domain=pyo.NonNegativeReals)
    m.fs.r = pyo.Var(m.fs.time,
                      initialize=0,
                      domain=pyo.NonNegativeReals)
    m.fs.p1 = pyo.Var(m.fs.time,
                      initialize=0,
                      domain=pyo.NonNegativeReals)
    m.fs.n1 = pyo.Var(m.fs.time,
                      initialize=0,
                      domain=pyo.NonNegativeReals)
    m.fs.n2 = pyo.Var(m.fs.time,
                      initialize=0,
                      domain=pyo.NonNegativeReals)
    m.fs.n3 = pyo.Var(m.fs.time,
                      initialize=0,
                      domain=pyo.NonNegativeReals)
    
    # soec.fuel_electrode.p = pyo.Var(m.fs.time,
    #                                 soec.fuel_electrode.ixnodes,
    #                                 soec.fuel_electrode.iznodes,
    #                                 initialize=0,
    #                                 domain=pyo.NonNegativeReals)
    # soec.fuel_electrode.n = pyo.Var(m.fs.time,
    #                                 soec.fuel_electrode.ixnodes,
    #                                 soec.fuel_electrode.iznodes,
    #                                 initialize=0,
    #                                 domain=pyo.NonNegativeReals)
    
    if not plant:
        # dTdz_electrode_lim = 675
        
        @m.fs.Constraint(m.fs.time)
        def makeup_mole_frac_eqn1(b, t):
            return b.makeup_mix.makeup_mole_frac_comp_H2[t] == 1e-14 + b.p[t]
        
        @m.fs.Constraint(m.fs.time)
        def makeup_mole_frac_eqn2(b, t):
            return b.makeup_mix.makeup_mole_frac_comp_H2O[t] == \
                0.999 - 1e-14 - b.n[t]
        
        @m.fs.Constraint(m.fs.time)
        def vgr_ratio_eqn(b, t):
            return b.condenser_split.recycle_ratio[t] == 1e-4 + b.q[t]
        
        @m.fs.Constraint(m.fs.time)
        def makeup_mole_frac_sum_eqn(b, t):
            return b.makeup_mix.makeup_mole_frac_comp_H2[t] + \
                b.makeup_mix.makeup_mole_frac_comp_H2O[t] == 0.999 - b.p[t]
        
        @m.fs.Constraint(m.fs.time)
        def condenser_outlet_temp_eqn(b, t):
            return b.condenser_flash.control_volume.properties_out[t] \
                .temperature == 273.15 + 50 + b.p1[t] - b.n1[t]

        @m.fs.Constraint(m.fs.time)
        def feed_recycle_ratio_eqn(b, t):
            return b.feed_recycle_split.recycle_ratio[t] == 0.999 - b.n2[t]
        
        @m.fs.Constraint(m.fs.time)
        def sweep_recycle_ratio_eqn(b, t):
            return b.sweep_recycle_split.recycle_ratio[t] == 0.999 - b.n3[t]

        
        # @soec.fuel_electrode.Constraint(m.fs.time, soec.fuel_electrode.ixnodes, soec.fuel_electrode.iznodes)
        # def dTdz_electrode_UB_rule(b, t, ix, iz):
        #     return b.dtemperature_dz[t, ix, iz] - dTdz_electrode_lim <= b.p[t, ix, iz]
        
        # @soec.fuel_electrode.Constraint(m.fs.time, soec.fuel_electrode.ixnodes, soec.fuel_electrode.iznodes)
        # def dTdz_electrode_LB_rule(b, t, ix, iz):
        #     return -b.dtemperature_dz[t, ix, iz] - dTdz_electrode_lim <= b.n[t, ix, iz]
        
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
                        v[t].set_value(v[m.fs.time.first()].value)
    
    # Fix initial conditions
    m.fs.fix_initial_conditions()
    
    for v in get_manipulated_variables(m):
        if plant:
            v[:].fix(v[m.fs.time.first()].value)
        else:
            v[:].set_value(v[m.fs.time.first()].value)
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
                v.set_value(state_var_source[tN_index].value)
        return None

    for state_var_target, state_var_source in zip(get_state_vars(target_model),
                                                  get_state_vars(source_model)):
        set_state_var_ics(state_var_target, state_var_source)
    
    return None

def apply_control_actions(controller_model, plant_model):
    for c, p in zip(get_manipulated_variables(controller_model),
                    get_manipulated_variables(plant_model)):
        p[:].fix(c[controller_model.fs.time.first()].value)
    
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
    oxygen_out = var_targets["oxygen_out"]
    hydrogen_in = var_targets["hydrogen_in"]
    vgr_recycle_ratio = var_targets['vgr_recycle_ratio']
    # condenser_heat_duty = var_targets['condenser_heat_duty']
    condenser_hot_outlet_temperature = var_targets['condenser_hot_outlet_temperature']
    makeup_mole_frac_comp_H2 = var_targets['makeup_mole_frac_comp_H2']
    makeup_mole_frac_comp_H2O = var_targets['makeup_mole_frac_comp_H2O']
    
    
    # soec = m.fs.soc_module.solid_oxide_cell
    
    expr = 0
    
    expr += 1e+00 * sum((m.fs.h2_mass_production[t] - h2_target[t_base + t])**2
                        for t in m.fs.time)
    
    # Penalties on manipulated variable deviations
    mv_multiplier = 1e-03
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.makeup_mix.makeup.flow_mol[t]
          - makeup_feed_rate[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.sweep_blower.inlet.flow_mol[t]
          - sweep_feed_rate[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e+00 * sum(
        (m.fs.soc_module.potential_cell[t]
          - potential[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e+01 * sum(
        (m.fs.feed_recycle_split.recycle_ratio[t]
          - fuel_recycle_ratio[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e+01 * sum(
        (m.fs.sweep_recycle_split.recycle_ratio[t]
          - sweep_recycle_ratio[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e-06 * sum(
        (m.fs.feed_heater.electric_heat_duty[t]
          - feed_heater_duty[t_base + t])**2 for t in m.fs.time) * 1e-5
    expr += mv_multiplier * 1e-07 * sum(
        (m.fs.sweep_heater.electric_heat_duty[t]
          - sweep_heater_duty[t_base + t])**2 for t in m.fs.time) * 1e-6
    expr += mv_multiplier * 1e+01 * sum(
        (m.fs.condenser_split.recycle_ratio[t]
          - vgr_recycle_ratio[t_base + t])**2 for t in m.fs.time)
    # expr += mv_multiplier * 1e-07 * sum(
    #     (m.fs.condenser_flash.heat_duty[t]
    #       - condenser_heat_duty[t_base + t])**2 for t in m.fs.time) * 1e-7
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.condenser_flash.vap_outlet.temperature[t]
          - condenser_hot_outlet_temperature[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e+01 * sum(
        (m.fs.makeup_mix.makeup_mole_frac_comp_H2[t]
          - makeup_mole_frac_comp_H2[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e+00 * sum(
        (m.fs.makeup_mix.makeup_mole_frac_comp_H2O[t]
          - makeup_mole_frac_comp_H2O[t_base + t])**2 for t in m.fs.time)
    
    expr += mv_multiplier * 1e-12 * sum((m.fs.feed_heater.electric_heat_duty[t] -
                              m.fs.feed_heater.electric_heat_duty[m.fs.time.prev(t)])**2
                            for t in m.fs.time if t != m.fs.time.first())
    expr += mv_multiplier * 1e-14 * sum((m.fs.sweep_heater.electric_heat_duty[t] -
                              m.fs.sweep_heater.electric_heat_duty[m.fs.time.prev(t)])**2
                            for t in m.fs.time if t != m.fs.time.first())
    # expr += mv_multiplier * 1e-14 * sum((m.fs.condenser_flash.heat_duty[t] -
    #                           m.fs.condenser_flash.heat_duty[m.fs.time.prev(t)])**2
    #                         for t in m.fs.time if t != m.fs.time.first())

    # mv_multiplier = 1e-02
    expr += mv_multiplier * 1e+00 * sum(
        (m.fs.soc_module.fuel_outlet_mole_frac_comp_H2[t]
          - soc_fuel_outlet_mole_frac_comp_H2[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.feed_heater.outlet.temperature[t]
          - feed_heater_outlet_temperature[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.sweep_heater.outlet.temperature[t]
          - sweep_heater_outlet_temperature[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.soc_module.fuel_outlet.temperature[t]
          - fuel_outlet_temperature[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.soc_module.oxygen_outlet.temperature[t]
          - sweep_outlet_temperature[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.stack_core_temperature[t]
          - stack_core_temperature[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e-03 * sum(
        (m.fs.stack_core_temperature[t]
          - stack_core_temperature[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e+01 * sum(
        (m.fs.sweep_recycle_split.mixed_state[t].mole_frac_comp['O2']
          - oxygen_out[t_base + t])**2 for t in m.fs.time)
    expr += mv_multiplier * 1e+01 * sum(
        (m.fs.feed_recycle_mix.mixed_state[t].mole_frac_comp['H2']
          - hydrogen_in[t_base + t])**2 for t in m.fs.time)
    
    m.fs.condenser_outlet_temp_eqn.activate()
    expr += 1e+03 * sum(m.fs.p1[t] + m.fs.n1[t] for t in m.fs.time)

    m.fs.feed_recycle_ratio_eqn.activate()
    expr += 1e+03 * sum(m.fs.n2[t] for t in m.fs.time)
    m.fs.sweep_recycle_ratio_eqn.activate()
    expr += 1e+03 * sum(m.fs.n3[t] for t in m.fs.time)

    if (t_base <= t_start) or (t_base >= t_start + t_ramp + t_settle + t_ramp):
        m.fs.makeup_mole_frac_eqn1.activate()
        m.fs.makeup_mole_frac_eqn2.activate()
        m.fs.vgr_ratio_eqn.activate()
        m.fs.makeup_mole_frac_sum_eqn.deactivate()
        expr += 1e+03 * sum(m.fs.p[t]
                            for t in m.fs.time)
        expr += 1e+03 * sum(m.fs.n[t]
                            for t in m.fs.time)
        expr += 1e+03 * sum(m.fs.q[t]
                            for t in m.fs.time)
    else:
        m.fs.makeup_mole_frac_eqn1.deactivate()
        m.fs.makeup_mole_frac_eqn2.deactivate()
        m.fs.vgr_ratio_eqn.deactivate()
        m.fs.makeup_mole_frac_sum_eqn.activate()
        expr += 1e+03 * sum(m.fs.p[t]
                            for t in m.fs.time)

    # expr += 1e+03 * sum(soec.fuel_electrode.p[t, ix, iz]
    #                     for t in m.fs.time
    #                     for ix in soec.fuel_electrode.ixnodes
    #                     for iz in soec.fuel_electrode.iznodes)
    # expr += 1e+03 * sum(soec.fuel_electrode.n[t, ix, iz]
    #                     for t in m.fs.time
    #                     for ix in soec.fuel_electrode.ixnodes
    #                     for iz in soec.fuel_electrode.iznodes)
    
    return expr

def apply_custom_scaling(m):
    sf = 410
    for _, c in m.fs.condenser_flash.control_volume.enthalpy_balances.items():
        sf_old = iscale.get_constraint_transform_applied_scaling_factor(c)
        sf_new = sf_old/sf
        iscale.constraint_scaling_transform(c, sf_new, overwrite=True)
    
    return None
###


###
# Set up ipopt.
use_idaes_solver_configuration_defaults()
idaes.cfg.ipopt.options.nlp_scaling_method = "user-scaling"
idaes.cfg.ipopt["options"]["linear_solver"] = "ma57"
idaes.cfg.ipopt.options.OF_ma57_automatic_scaling = "yes"
idaes.cfg.ipopt["options"]["max_iter"] = 300
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
# apply_custom_scaling(plant)
# apply_custom_scaling(olnmpc)
iscale.scale_time_discretization_equations(plant, plant.fs.time, 1/t_step)
iscale.scale_time_discretization_equations(olnmpc, olnmpc.fs.time, 1/t_step)

#Generate initial plant model results.
print('Solving initial plant model...\n')
init_petsc_results = petsc_initialize(plant)
init_plant_results = solver.solve(plant, tee=True)

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
                states_dict[c.name][tuple(idxs)].append(v.value)
    return None
states_dict = make_states_dict(plant)
save_states(plant, states_dict)

#Loosen ipopt tolerances before controller loop.
idaes.cfg.ipopt["options"]["tol"] = 1e-04
idaes.cfg.ipopt["options"]["constr_viol_tol"] = 1e-04
idaes.cfg.ipopt["options"]["acceptable_tol"] = 1e-04
idaes.cfg.ipopt["options"]["dual_inf_tol"] = 1e+01
solver = pyo.SolverFactory("ipopt")

h2_production_rate = []
def save_h2_production_rate(m, h2_production_rate):
    h2_production_rate.append(m.fs.h2_mass_production[m.fs.time.last()].value)
    return None

objective = []

controls_dict = {c.name: [] for c in get_manipulated_variables(olnmpc)}
def save_controls(m, controls_dict):
    for c in get_manipulated_variables(m):
        controls_dict[c.name].append(c[m.fs.time.first()].value)
    return None

CVs_dict = {c.name: [] for c in get_CVs(plant)}
CVs_dict.update({'fs.sweep_recycle_split.mixed_state.mole_frac_comp[O2]': []})
CVs_dict.update({'fs.feed_recycle_mix.mixed_state.mole_frac_comp[H2]': []})
other_states = [
    "fs.soc_module.fuel_inlet.flow_mol",
    "fs.soc_module.oxygen_inlet.flow_mol",
    "fs.soc_module.solid_oxide_cell.fuel_inlet.mole_frac_comp",  # H2O
    "fs.soc_module.solid_oxide_cell.fuel_channel.mole_frac_comp",  # H2O
    "fs.soc_module.solid_oxide_cell.oxygen_inlet.mole_frac_comp",  # O2
    "fs.soc_module.solid_oxide_cell.oxygen_channel.mole_frac_comp",  # O2
    "fs.condenser_split.inlet.mole_frac_comp",  # H2
    "fs.feed_medium_exchanger.tube_inlet.flow_mol",
    "fs.total_electric_power",
    "fs.soc_module.solid_oxide_cell.fuel_channel.temperature_inlet",
    "fs.soc_module.solid_oxide_cell.oxygen_channel.temperature_inlet",
    "fs.soc_module.solid_oxide_cell.soec.temperature_z",  # mean over iznodes
    "fs.condenser_split.inlet.mole_frac_comp",
]
for i in other_states:
    CVs_dict.update({i: []})


def save_CVs(m, CVs_dict):
    for c in get_CVs(m):
        CVs_dict[c.name].append(c[m.fs.time.last()].value)
    CVs_dict['fs.sweep_recycle_split.mixed_state.mole_frac_comp[O2]'].append(m.fs.sweep_recycle_split.mixed_state[m.fs.time.last()].mole_frac_comp['O2'].value)
    CVs_dict['fs.feed_recycle_mix.mixed_state.mole_frac_comp[H2]'].append(m.fs.feed_recycle_mix.mixed_state[m.fs.time.last()].mole_frac_comp['H2'].value)
    return None


# slack = []
# slack = {}
# slack.update({'p': []})
# slack.update({'n': []})
# def save_slack(m, slack):
#     slack['p'].append(m.fs.p[m.fs.time.first()].value)
#     slack['n'].append(m.fs.n[m.fs.time.first()].value)
#     return None

solver_time = []

dTdz_electrode_logbook = {}
for i in range(plant.fs.soc_module.solid_oxide_cell.fuel_electrode.iznodes.first(),
               plant.fs.soc_module.solid_oxide_cell.fuel_electrode.iznodes.last()+1):
    dTdz_electrode_logbook.update({i: []})
###


###
#Create variable setpoint trajectories.
global var_targets
var_targets = make_var_trajs(2, traj_time_set, t_start, t_ramp, t_settle, t_end)

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
    
    timer_start = time.time()
    olnmpc_results = solver.solve(olnmpc, tee=True, load_solutions=True)
    solver_time.append(time.time()-timer_start)
    # break
    objective.append(value(olnmpc.obj))
    save_controls(olnmpc, controls_dict)
    # save_slack(olnmpc, slack)
    
    # Set up Plant model
    set_initial_conditions(target_model=plant, source_model=plant)
    plant.fs.fix_initial_conditions()
    apply_control_actions(controller_model=olnmpc, plant_model=plant)
    assert degrees_of_freedom(plant) == 0

    print(f'\nSolving {plant.name}, iteration {iter_count}...\n')
    
    plant_results = solver.solve(plant.fs, tee=True, load_solutions=True)
    
    save_CVs(plant, CVs_dict)
    save_states(plant, states_dict)
    save_h2_production_rate(plant, h2_production_rate)
    for i in range(plant.fs.soc_module.solid_oxide_cell.fuel_electrode.iznodes.first(),
                   plant.fs.soc_module.solid_oxide_cell.fuel_electrode.iznodes.last()+1):
        dTdz_electrode_logbook[i].append(plant.fs.soc_module.solid_oxide_cell.fuel_electrode.dtemperature_dz[plant.fs.time.last(),1,i].value)
    
    iter_count += 1
###

for i in controls_dict.keys():
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(controls_dict[i],'b-')
    # ax.plot(sim_time_set/3600,controls_dict[i],'b-')
    for j in alias_dict.keys():
        if i == j:
            ax.plot(var_targets[alias_dict[j]].values(),'r--')
            # ax.plot(traj_time_set/3600,var_targets[alias_dict[j]].values(),'r--')
            # ax.set_xlabel('Time (hr)')
            plt.title(alias_dict[j])
    # if i == 'fs.makeup_mix._flow_mol_makeup_ref' or i == 'fs.sweep_blower._flow_mol_inlet_ref':
    #     ax.axhline(0, color='k', linestyle='--')
    #     ax.axhline(5e4, color='k', linestyle='--')
    # if i == 'fs.condenser_split.recycle_ratio' or i == 'fs.feed_recycle_split.recycle_ratio' or i == 'fs.sweep_recycle_split.recycle_ratio':
    #     ax.axhline(0, color='k', linestyle='--')
    # if i == 'fs.makeup_mix.makeup_mole_frac_comp_H2' or i == 'fs.makeup_mix.makeup_mole_frac_comp_H2O':
    #     ax.axhline(1e-20, color='k', linestyle='--')
    #     ax.axhline(1.001, color='k', linestyle='--')
    plt.show()

for i in CVs_dict.keys():
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(CVs_dict[i],'g-')
    # ax.plot(sim_time_set/3600,CVs_dict[i],'g-')
    for j in alias_dict.keys():
        if i == j:
            ax.plot(var_targets[alias_dict[j]].values(),'r--')
            # ax.plot(traj_time_set/3600,var_targets[alias_dict[j]].values(),'r--')
            # ax.set_xlabel('Time (hr)')
            fig.suptitle(alias_dict[j])
    # if i == 'fs.soc_module.fuel_outlet_mole_frac_comp_H2' or i == 'fs.feed_recycle_mix.mixed_state.mole_frac_comp[H2]' or i == 'fs.sweep_recycle_split.mixed_state.mole_frac_comp[O2]':
    #     ax.axhline(1e-20, color='k', linestyle='--')
    #     ax.axhline(1.001, color='k', linestyle='--')
    # if i == 'fs.feed_heater._temperature_outlet_ref' or i == 'fs.soc_module._temperature_fuel_outlet_ref' or i == 'fs.sweep_heater._temperature_outlet_ref' or i == 'fs.soc_module._temperature_oxygen_outlet_ref':
    #     ax.axhline(273.15, color='k', linestyle='--')
    #     ax.axhline(2500, color='k', linestyle='--')
    plt.show()

fig = plt.figure()
ax = fig.subplots()
ax.plot(h2_production_rate, 'g-')
ax.plot(var_targets[alias_dict['fs.h2_mass_production']].values(), 'r--')
# ax.plot(sim_time_set/3600,h2_production_rate, 'g-')
# ax.plot(traj_time_set/3600,var_targets[alias_dict['fs.h2_mass_production']].values(), 'r--')
# ax.set_xlabel('Time (hr)')
plt.title('$\mathrm{H_2}$ production rate')
plt.show()

fig = plt.figure()
ax = fig.subplots()
for iz in 1, 3, 5, 8, 10:
    ax.plot(
        # plot_time_set,
        dTdz_electrode_logbook[iz],
        label=f"node {iz}"
    )
ax.set_xlim((0, len(sim_time_set)-1))
ax.set_ylim((-1000, 1000))
# ax.set_xlabel("Time (hr)", fontsize=12)
ax.set_ylabel("$dT/dz$ ($K/m$)", fontsize=12)
ax.set_title("SOEC PEN temperature gradient, NMPC", fontsize=12)
# demarcate_ramps(ax)
ax.legend(loc='best')
plt.show()