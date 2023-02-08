import sys
sys.path.append('../')
sys.path.append('../../')
import time
import pandas as pd
import numpy as np

import pyomo.environ as pyo
from pyomo.environ import Objective
from pyomo.common.collections import ComponentMap
from pyomo.dae import DerivativeVar
# import pyomo.dae.flatten as flatten
# from generate_sliced_components import generate_sliced_components
# flatten.generate_sliced_components = generate_sliced_components
from pyomo.dae.flatten import generate_sliced_components
from pyomo.dae.flatten import flatten_dae_components

import idaes
import idaes.core.util.scaling as iscale
from idaes.core.solvers import petsc
from idaes.core.solvers import use_idaes_solver_configuration_defaults
import idaes.logger as idaeslog
import idaes.core.util.model_serializer as ms
from idaes.models.control.controller import (
    ControllerType,
    ControllerMVBoundType,
    ControllerAntiwindupType,
)
from soec_dynamic_flowsheet_mk2 import SoecStandaloneFlowsheet
from soec_dynamic_flowsheet_helper import (
    create_ramping_eqns,
    scale_indexed_constraint,
    OperatingScenario,
)
from save_results import save_results
from plot_results import plot_results
import matplotlib.pyplot as plt

# =============================================================================
def get_time_coordinates():
    t_start = 0.5 * 60 * 60
    t_ramp = 0.5 * 60 * 60
    t_settle = 3 * 60 * 60
    t_end = 3 * 60 * 60
    time_set_PI = [0,
                   t_start,
                   t_start + t_ramp,
                   t_start + t_ramp + t_settle,
                   t_start + 2 * t_ramp + t_settle,
                   t_start + 2 * t_ramp + t_settle + t_end]

    step = 150.0
    nsteps_horizon = 6
    horizon = nsteps_horizon * step
    ntfe = int(horizon / step)
    time_set_controller = np.linspace(0, horizon, num=ntfe+1)
    
    t_last = time_set_PI[-1]
    sim_horizon = t_last - horizon
    nsteps = int(t_last / step)
    time_set = np.linspace(0, t_last, num=nsteps+1)
    time_set_output = np.linspace(0, t_last + horizon, num=nsteps+1+nsteps_horizon)
    
    nmpc_params = {
        't_start': t_start,
        't_ramp': t_ramp,
        't_settle': t_settle,
        't_end': t_end,
        'time_set_PI': time_set_PI,
        'step': step,
        'nsteps_horizon': nsteps_horizon,
        'horizon': horizon,
        'ntfe': ntfe,
        'time_set_controller': time_set_controller,
        't_last': t_last,
        'sim_horizon': sim_horizon,
        'nsteps': nsteps,
        'time_set': time_set,
        'time_set_output': time_set_output,
    }
    
    return nmpc_params

nmpc_params = get_time_coordinates()


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
    # "fs.condenser_hx._flow_mol_cold_side_inlet_ref": "cooling_water_feed",
    "fs.condenser_hx._temperature_hot_side_outlet_ref": "condenser_hot_outlet_temperature",
    "fs.makeup_mix.makeup_mole_frac_comp_H2": "makeup_mole_frac_comp_H2",
    "fs.makeup_mix.makeup_mole_frac_comp_H2O": "makeup_mole_frac_comp_H2O",
}


def get_tracking_variables(m):
    tracking_variables = [
        m.fs.h2_mass_production,
        m.fs.soc_module.potential_cell,
        m.fs.soc_module.fuel_outlet_mole_frac_comp_H2,
        m.fs.makeup_mix._flow_mol_makeup_ref,
        m.fs.sweep_blower._flow_mol_inlet_ref,
        m.fs.feed_heater.electric_heat_duty,
        m.fs.feed_heater._temperature_outlet_ref,
        m.fs.soc_module._temperature_fuel_outlet_ref,
        m.fs.sweep_heater.electric_heat_duty,
        m.fs.sweep_heater._temperature_outlet_ref,
        m.fs.soc_module._temperature_oxygen_outlet_ref,
        m.fs.feed_recycle_split.recycle_ratio,
        m.fs.sweep_recycle_split.recycle_ratio,
        # m.fs.sweep_recycle_split.mixed_state[0].mole_frac_comp['O2'],
        # m.fs.feed_recycle_mix.mixed_state[0].mole_frac_comp['H2'],
        m.fs.condenser_split.recycle_ratio,
        m.fs.stack_core_temperature,
        m.fs.condenser_flash.heat_duty,
        # m.fs.condenser_hx._flow_mol_cold_side_inlet_ref,
        m.fs.condenser_hx._temperature_hot_side_outlet_ref,
        m.fs.makeup_mix.makeup_mole_frac_comp_H2,
        m.fs.makeup_mix.makeup_mole_frac_comp_H2O,
    ]
    return tracking_variables


def get_manipulated_variables(m):
    manipulated_variables = [
        m.fs.soc_module.potential_cell,
        m.fs.makeup_mix._flow_mol_makeup_ref,
        m.fs.sweep_blower._flow_mol_inlet_ref,
        m.fs.condenser_split.recycle_ratio,
        # m.fs.condenser_hx._flow_mol_cold_side_inlet_ref,
        m.fs.condenser_flash.heat_duty,
        m.fs.feed_heater.electric_heat_duty,
        m.fs.sweep_heater.electric_heat_duty,
        m.fs.feed_recycle_split.recycle_ratio,
        m.fs.sweep_recycle_split.recycle_ratio,
        m.fs.makeup_mix.makeup_mole_frac_comp_H2,
        m.fs.makeup_mix.makeup_mole_frac_comp_H2O,
    ]
    return manipulated_variables


# =============================================================================
def get_h2_production_target(m, reversible_mode=False):
    df = pd.read_csv(
        "./../../soec_flowsheet_operating_conditions.csv",
        index_col=0,
    )

    alias = "h2_production_rate"
    if reversible_mode:
        sp_hydrogen_hi = df[alias]["maximum_H2"]
        sp_hydrogen_lo = df[alias]["power"]
    else:
        sp_hydrogen_hi = df[alias]["maximum_H2"]
        sp_hydrogen_lo = df[alias]["minimum_H2"]
    
    def ramp_h2_production(t, ramp_down):
        slope = (sp_hydrogen_hi - sp_hydrogen_lo) / nmpc_params['t_ramp']
        if ramp_down:
            t_ramp_start = nmpc_params['time_set_PI'][1]
            return sp_hydrogen_hi - slope * (t - t_ramp_start)
        else:
            t_ramp_start = nmpc_params['time_set_PI'][3]
            return sp_hydrogen_lo + slope * (t - t_ramp_start)
    
    h2_target = {t: 0.0 for t in nmpc_params['time_set_output']}
    for t in nmpc_params['time_set_output']:
        if t <= nmpc_params['t_start']:
            h2_target[t] = sp_hydrogen_hi
        elif t > nmpc_params['time_set_PI'][1] and t <= nmpc_params['time_set_PI'][2]:
            h2_target[t] = ramp_h2_production(t, ramp_down=True)
        elif t > nmpc_params['time_set_PI'][2] and t <= nmpc_params['time_set_PI'][3]:
            h2_target[t] = sp_hydrogen_lo
        elif t > nmpc_params['time_set_PI'][3] and t <= nmpc_params['time_set_PI'][4]:
            h2_target[t] = ramp_h2_production(t, ramp_down=False)
        else:
            h2_target[t] = sp_hydrogen_hi

    return h2_target


def get_MV_targets(m):
    df = pd.read_csv(
        "./../../soec_flowsheet_operating_conditions.csv",
        index_col=0,
    )

    def get_setpoint_trajectory(var, reversible_mode=False):
        alias = alias_dict[var.name]
        if reversible_mode:
            sp_hydrogen_hi = df[alias]["maximum_H2"]
            sp_hydrogen_lo = df[alias]["power"]
        else:
            sp_hydrogen_hi = df[alias]["maximum_H2"]
            sp_hydrogen_lo = df[alias]["minimum_H2"]
        
        def ramp_var(t, ramp_down):
            slope = (sp_hydrogen_hi - sp_hydrogen_lo) / nmpc_params['t_ramp']
            if ramp_down:
                t_ramp_start = nmpc_params['time_set_PI'][1]
                return sp_hydrogen_hi - slope * (t - t_ramp_start)
            else:
                t_ramp_start = nmpc_params['time_set_PI'][3]
                return sp_hydrogen_lo + slope * (t - t_ramp_start)
        
        var_target = {t: 0.0 for t in nmpc_params['time_set_output']}
        for t in nmpc_params['time_set_output']:
            if t <= nmpc_params['time_set_PI'][1]:
                var_target[t] = sp_hydrogen_hi
            elif t > nmpc_params['time_set_PI'][1] and t <= nmpc_params['time_set_PI'][2]:
                var_target[t] = ramp_var(t, ramp_down=True)
            elif t > nmpc_params['time_set_PI'][2] and t <= nmpc_params['time_set_PI'][3]:
                var_target[t] = sp_hydrogen_lo
            elif t > nmpc_params['time_set_PI'][3] and t <= nmpc_params['time_set_PI'][4]:
                var_target[t] = ramp_var(t, ramp_down=False)
            else:
                var_target[t] = sp_hydrogen_hi
        
        return var_target
    
    MV_targets = {alias: {} for alias in alias_dict.values()}
    tracking_variables = get_tracking_variables(m)
    for var in tracking_variables:
        alias = alias_dict[var.name]
        target = get_setpoint_trajectory(var)
        MV_targets[alias] = target
    
    return MV_targets


def get_h2_target_for_control_horizon(m, iter):
    tset = m.fs.time
    h2_target = get_h2_production_target(m)
    h2_target_list = list(h2_target.values())
    h2_target_for_horizon = h2_target_list[iter:iter+len(tset)]
    h2_target_for_horizon = {t: h for t, h in zip(tset, h2_target_for_horizon)}
    return h2_target_for_horizon


def get_MV_targets_for_control_horizon(m, iter):
    tset = m.fs.time
    MV_targets = get_MV_targets(m)
    MV_targets_for_horizon = {alias: [] for alias in alias_dict.values()}
    for alias in alias_dict.values():
        target_list = list(MV_targets[alias].values())
        target_for_horizon = target_list[iter:iter+len(tset)]
        target_for_horizon = {t: mv for t, mv in zip(tset, target_for_horizon)}
        MV_targets_for_horizon[alias] = target_for_horizon
    return MV_targets_for_horizon
    

def tracking_objective_simple(m):
    h2_target = {t: 0.4 for t in m.fs.time}
    return sum((m.fs.h2_mass_production[t] - h2_target[t])**2 for t in m.fs.time)


def tracking_objective(m):
    h2_target = get_h2_production_target(m)
    return sum((m.fs.h2_mass_production[t] - h2_target[t])**2 for t in m.fs.time)


def make_tracking_objective_with_MVs(m, iter):
    h2_target = get_h2_target_for_control_horizon(m, iter)
    MV_targets = get_MV_targets_for_control_horizon(m, iter)
    
    potential = MV_targets['potential']
    soc_fuel_outlet_mole_frac_comp_H2 = MV_targets['soc_fuel_outlet_mole_frac_comp_H2']
    makeup_feed_rate = MV_targets['makeup_feed_rate']
    sweep_feed_rate = MV_targets['sweep_feed_rate']
    feed_heater_duty = MV_targets['feed_heater_duty']
    feed_heater_outlet_temperature = MV_targets['feed_heater_outlet_temperature']
    fuel_outlet_temperature = MV_targets['fuel_outlet_temperature']
    sweep_heater_duty = MV_targets['sweep_heater_duty']
    sweep_heater_outlet_temperature = MV_targets['sweep_heater_outlet_temperature']
    sweep_outlet_temperature = MV_targets['sweep_outlet_temperature']
    fuel_recycle_ratio = MV_targets['fuel_recycle_ratio']
    sweep_recycle_ratio = MV_targets['sweep_recycle_ratio']
    vgr_recycle_ratio = MV_targets['vgr_recycle_ratio']
    condenser_heat_duty = MV_targets['condenser_heat_duty']
    # cooling_water_feed = MV_targets['cooling_water_feed']
    stack_core_temperature = MV_targets['stack_core_temperature']
    condenser_hot_outlet_temperature = MV_targets['condenser_hot_outlet_temperature']
    makeup_mole_frac_comp_H2 = MV_targets['makeup_mole_frac_comp_H2']
    makeup_mole_frac_comp_H2O = MV_targets['makeup_mole_frac_comp_H2O']
    
    # plt.figure()
    # plt.plot(list(m.fs.time), h2_target.values())

    def tracking_objective_with_MVs(m):
        expr = 0
        expr += 1e+02 * sum(
            (m.fs.h2_mass_production[t] - h2_target[t])**2 for t in m.fs.time
            if t != m.fs.time.first())

        # Penalties on manipulated variable deviations
        mv_multiplier = 1e-03
        expr += mv_multiplier * 1e-03 * sum(
            (m.fs.makeup_mix.makeup.flow_mol[t]
             - makeup_feed_rate[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e-03 * sum(
            (m.fs.sweep_blower.inlet.flow_mol[t]
             - sweep_feed_rate[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e+00 * sum(
            (m.fs.soc_module.potential_cell[t]
             - potential[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e+00 * sum(
            (m.fs.feed_recycle_split.recycle_ratio[t]
             - fuel_recycle_ratio[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e+00 * sum(
            (m.fs.sweep_recycle_split.recycle_ratio[t]
             - sweep_recycle_ratio[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e-06 * sum(
            (m.fs.feed_heater.electric_heat_duty[t]
             - feed_heater_duty[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e-07 * sum(
            (m.fs.sweep_heater.electric_heat_duty[t]
             - sweep_heater_duty[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e+04 * sum(
            (m.fs.condenser_split.recycle_ratio[t]
             - vgr_recycle_ratio[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        # expr += mv_multiplier * 1e-04 * sum(
            # (m.fs.condenser_hx.cold_side_inlet.flow_mol[t]
             # - cooling_water_feed[t])**2 for t in m.fs.time
             # if t != m.fs.time.first())
        expr += mv_multiplier * 1e-07 * sum(
            (m.fs.condenser_flash.heat_duty[t]
              - condenser_heat_duty[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e+14 * sum(
            (m.fs.makeup_mix.makeup_mole_frac_comp_H2[t]
              - makeup_mole_frac_comp_H2[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e+00 * sum(
            (m.fs.makeup_mix.makeup_mole_frac_comp_H2O[t]
             - makeup_mole_frac_comp_H2O[t])**2 for t in m.fs.time
            if t != m.fs.time.first())

        mv_multiplier = 1e-06
        expr += mv_multiplier * 1e+00 * sum(
            (m.fs.soc_module.fuel_outlet_mole_frac_comp_H2[t]
             - soc_fuel_outlet_mole_frac_comp_H2[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e-03 * sum(
            (m.fs.feed_heater.outlet.temperature[t]
             - feed_heater_outlet_temperature[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e-03 * sum(
            (m.fs.sweep_heater.outlet.temperature[t]
             - sweep_heater_outlet_temperature[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e-03 * sum(
            (m.fs.soc_module.fuel_outlet.temperature[t]
             - fuel_outlet_temperature[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e-03 * sum(
            (m.fs.soc_module.oxygen_outlet.temperature[t]
             - sweep_outlet_temperature[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e-03 * sum(
            (m.fs.condenser_hx.hot_side_outlet.temperature[t]
             - condenser_hot_outlet_temperature[t])**2 for t in m.fs.time
            if t != m.fs.time.first())
        expr += mv_multiplier * 1e-03 * sum(
            (m.fs.stack_core_temperature[t]
             - stack_core_temperature[t])**2 for t in m.fs.time
            if t != m.fs.time.first())

        # Penalties on oscillations
        # expr += 1e+01 * sum(
        # expr += 0.1 * sum(
        #     (m.fs.sweep_recycle_split.recycle_split_fraction[t] -
        #       m.fs.sweep_recycle_split.recycle_split_fraction[m.fs.time.prev(t)])**2
        #     for t in m.fs.time if t != m.fs.time.first())

        # expr += 1e+01 * sum(
        #     (m.fs.h2_mass_production[t] -
        #      m.fs.h2_mass_production[m.fs.time.prev(t)])**2
        #     for t in m.fs.time if t != m.fs.time.first())
    
        return expr
    
    if hasattr(m, 'obj'):
        m.del_component('obj')
    m.obj = Objective(rule=tracking_objective_with_MVs, sense=pyo.minimize)

    return None


def shift_dae_vars_by_dt(m, tset, dt):
    scalar_vars, dae_vars = flatten_dae_components(m, tset, pyo.Var)
    
    seen = set()
    t0 = tset.first()
    tf = tset.last()
    for var in dae_vars:
        if id(var[t0]) in seen:
            continue
        else:
            seen.add(id(var[t0]))
        for t in tset:
            ts = t + dt
            idx = tset.find_nearest_index(ts)
            if idx is None:
                # ts is outside the controller's horizon
                var[t].set_value(var[tf].value)
            else:
                ts = tset.at(idx)
                var[t].set_value(var[ts].value)

    return None


def apply_custom_variable_scaling(m):
    sf = 1.39e+03
    for i, c in m.fs.soc_module.oxygen_properties_in.items():
        sf_old = iscale.get_scaling_factor(c.temperature)
        sf_new = sf * sf_old
        iscale.set_scaling_factor(c.temperature, 1 / sf_new)
    
    sf = 1.48e+06
    for i, c in m.fs.soc_module.solid_oxide_cell.fuel_electrode.int_energy_density_solid.items():
        sf_old = iscale.get_scaling_factor(c)
        sf_new = sf * sf_old
        iscale.set_scaling_factor(c, sf_new)
    
    sf = 1.16e+05
    for i, c in m.fs.soc_module.solid_oxide_cell.interconnect.int_energy_density_solid.items():
        sf_old = iscale.get_scaling_factor(c)
        sf_new = sf * sf_old
        iscale.set_scaling_factor(c, sf_new)

    sf = 6.67e+05
    for i, c in m.fs.sweep_exchanger.heat_holdup.items():
        sf_old = iscale.get_scaling_factor(c)
        sf_new = sf * sf_old
        iscale.set_scaling_factor(c, sf_new)

    sf = 6.67e+05
    for i, c in m.fs.feed_hot_exchanger.heat_holdup.items():
        sf_old = iscale.get_scaling_factor(c)
        sf_new = sf * sf_old
        iscale.set_scaling_factor(c, sf_new)

    sf = 6.67e+05
    for i, c in m.fs.feed_medium_exchanger.heat_holdup.items():
        sf_old = iscale.get_scaling_factor(c)
        sf_new = sf * sf_old
        iscale.set_scaling_factor(c, sf_new)

    sf = 6.67e+06
    for i, c in m.fs.feed_heater.heat_holdup.items():
        sf_old = iscale.get_scaling_factor(c)
        sf_new = sf * sf_old
        iscale.set_scaling_factor(c, sf_new)

    sf = 6.67e+06
    for i, c in m.fs.sweep_heater.heat_holdup.items():
        sf_old = iscale.get_scaling_factor(c)
        sf_new = sf * sf_old
        iscale.set_scaling_factor(c, sf_new)
    
    # sf = 1e+03
    # for i, c in m.fs.sweep_blower.d_flow_mol_inlet_refdt.items():
    #     sf_old = iscale.get_scaling_factor(c)
    #     sf_new = sf * sf_old
    #     iscale.set_scaling_factor(c, sf_new)

    # sf = 1e+04
    # for i, c in m.fs.condenser_hx.d_flow_mol_cold_side_inlet_refdt.items():
    #     sf_old = iscale.get_scaling_factor(c)
    #     sf_new = sf * sf_old
    #     iscale.set_scaling_factor(c, sf_new)

    # sf = 1e+03
    # for i, c in m.fs.makeup_mix.d_flow_mol_makeup_refdt.items():
    #     sf_old = iscale.get_scaling_factor(c)
    #     sf_new = sf * sf_old
    #     iscale.set_scaling_factor(c, sf_new)

    # sf = 3.29e-06
    # for i, c in m.fs.soc_module.solid_oxide_cell.fuel_electrode.dcedt_solid.items():
    #     iscale.set_scaling_factor(c, 1 / sf)
    
    # sf = 1.14e-05
    # for i, c in m.fs.soc_module.solid_oxide_cell.interconnect.dcedt_solid.items():
    #     iscale.set_scaling_factor(c, 1 / sf)

    # sf = 1.80e-06
    # for i, c in m.fs.sweep_exchanger.heat_accumulation.items():
    #     iscale.set_scaling_factor(c, 1 / sf)

    # sf = 1.80e-06
    # for i, c in m.fs.feed_hot_exchanger.heat_accumulation.items():
    #     iscale.set_scaling_factor(c, 1 / sf)

    # sf = 1.80e-06
    # for i, c in m.fs.feed_medium_exchanger.heat_accumulation.items():
    #     iscale.set_scaling_factor(c, 1 / sf)

    # sf = 1.01e-06
    # for i, c in m.fs.feed_heater.heat_accumulation.items():
    #     iscale.set_scaling_factor(c, 1 / sf)

    # sf = 1.01e-06
    # for i, c in m.fs.sweep_heater.heat_accumulation.items():
    #     iscale.set_scaling_factor(c, 1 / sf)

    return None


def apply_custom_constraint_scaling(m):
    sf = 1e-02
    for i, c in m.fs.soc_module.solid_oxide_cell.fuel_electrode.conc_mol_comp_eqn.items():
        sf_old = iscale.get_constraint_transform_applied_scaling_factor(c)
        sf_new = sf * sf_old
        iscale.constraint_scaling_transform(c, sf_new)

    sf = 410
    for _, c in m.fs.condenser_flash.control_volume.enthalpy_balances.items():
        sf_old = iscale.get_constraint_transform_applied_scaling_factor(c)
        sf_new = sf_old/sf
        iscale.constraint_scaling_transform(c, sf_new, overwrite=True)

    # sf = 6.76e-07
    # for i, c in m.fs.soc_module.solid_oxide_cell.fuel_electrode.int_energy_density_solid_eqn.items():
    #     iscale.constraint_scaling_transform(c, 1 / sf, overwrite=True)
    
    # sf = 6.76e-07
    # for i, c in m.fs.soc_module.solid_oxide_cell.interconnect.int_energy_density_solid_eqn.items():
    #     iscale.constraint_scaling_transform(c, 1 / sf, overwrite=True)

    # sf = 2.09e+06
    # for i, c in m.fs.soc_module.solid_oxide_cell.fuel_electrode.dcedt_solid_disc_eq.items():
    #     iscale.constraint_scaling_transform(c, 1 / sf, overwrite=True)
    
    # sf = 6.67e+06
    # for i, c in m.fs.feed_heater.heat_accumulation_disc_eq.items():
    #     iscale.constraint_scaling_transform(c, 1 / sf, overwrite=True)
    
    # sf = 6.67e+06
    # for i, c in m.fs.sweep_heater.heat_accumulation_disc_eq.items():
    #     iscale.constraint_scaling_transform(c, 1 / sf, overwrite=True)

    # sf = 6.67e+05
    # for i, c in m.fs.feed_medium_exchanger.heat_accumulation_disc_eq.items():
    #     iscale.constraint_scaling_transform(c, 1 / sf, overwrite=True)
        
    # sf = 6.67e+05
    # for i, c in m.fs.feed_hot_exchanger.heat_accumulation_disc_eq.items():
    #     iscale.constraint_scaling_transform(c, 1 / sf, overwrite=True)

    # sf = 6.67e+05
    # for i, c in m.fs.sweep_exchanger.heat_accumulation_disc_eq.items():
    #     iscale.constraint_scaling_transform(c, 1 / sf, overwrite=True)
    
    # sf = 1.65e+05
    # for i, c in m.fs.soc_module.solid_oxide_cell.interconnect.dcedt_solid_disc_eq.items():
    #     iscale.constraint_scaling_transform(c, 1 / sf, overwrite=True)
    
    # sf = 1.65e+05
    # for i, c in m.fs.soc_module.solid_oxide_cell.interconnect.dcedt_solid_disc_eq.items():
    #     iscale.constraint_scaling_transform(c, 1 / sf, overwrite=True)    

    return None


def check_scaling(m, large=1e+03, small=1e-03):
    jac, nlp = iscale.get_jacobian(m, scaled=True)
    jac_csc = jac.tocsc()
    djac = jac.toarray()
    # print("Extreme Jacobian entries:")
    # for i in iscale.extreme_jacobian_entries(jac=jac, nlp=nlp, large=1E3, small=0):
    #     print(f"    {i[0]:.2e}, [{i[1]}, {i[2]}]")
    print("Badly scaled variables:")
    for i in iscale.extreme_jacobian_columns(
            jac=jac, nlp=nlp, large=large, small=small):
        print(f"    {i[0]:.2e}, [{i[1]}]")
    print("\n\n"+"Badly scaled constraints:")
    for i in iscale.extreme_jacobian_rows(
            jac=jac, nlp=nlp, large=large, small=small):
        print(f"    {i[0]:.2e}, [{i[1]}]")
    print(f"Jacobian Condition Number: {iscale.jacobian_cond(jac=jac):.2e}")
    
    jac_conditioning = {idx: (min(row), max(row)) for idx, row in enumerate(djac)}
    solcons = nlp.get_pyomo_constraints()
    solvars = nlp.get_pyomo_variables()

    return jac, jac_conditioning, solcons, solvars


def hunt_degeneracy(m):
    from idaes.core.util.model_diagnostics import DegeneracyHunter
    import time
    
    if not hasattr(m, "obj"):
        m.obj = pyo.Objective(expr=0)
    
    t1 = time.time()
    dh = DegeneracyHunter(m, solver=pyo.SolverFactory('cbc'))
    dh.check_rank_equality_constraints(dense=True)
    variables = dh.nlp.get_pyomo_variables()
    ds = dh.find_candidate_equations()

    for i in np.where(abs(dh.v[:,-1]) > 0.1)[0]:
        print(str(i) + ": " + variables[i].name)
    for i in np.where(abs(dh.u[:,-1]) > 0.1)[0]:
        print(str(i) + ": " + dh.eq_con_list[i].name)

    t2 = time.time()
    print(f"Time spent on computing candidate degenerate set: {(t2 - t1) / 60} min")
    
    return ds


def check_incidence_analysis(m):
    from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
    igraph = IncidenceGraphInterface(m)
    var_dmp, con_dmp = igraph.dulmage_mendelsohn()

    print("Overconstrained variables:")
    for var in var_dmp.overconstrained:
        print("  %s" % var.name)
    print("Overconstraining equations:")
    for con in con_dmp.overconstrained:
        print("  %s" % con.name)
    print("Unmatched variables:")
    for var in var_dmp.unmatched:
        print("  %s" % var.name)
    print("Unmatched equations:")
    for con in con_dmp.unmatched:
        print("  %s" % con.name)
    print(f"Number of unmatched variables: {len(var_dmp.unmatched)}")
    
    from idaes.core.util.model_statistics import (
        unused_variables_set,
        fixed_unused_variables_set,
        variables_near_bounds_set,
    )
    print(f"Unused variables:")
    for var in unused_variables_set(m):
        if not var.fixed:
            if not isinstance(var.parent_component(), DerivativeVar):
                print(f" {var.name}")
    print(f"Variables near bounds:")
    for var in variables_near_bounds_set(m, tol=1e-6):
        print(f" {var.name}")
    
    return None


def get_large_duals(m, tol=1e+03, jacnz=False):
    jac, nlp = iscale.get_jacobian(m)
    # duals = nlp.get_duals()
    lam = nlp.get_duals_eq()
    # mu = nlp.get_duals_ineq()
    
    conlist = nlp.get_pyomo_constraints()
    varlist = nlp.get_pyomo_variables()
    assert len(conlist) == len(lam)
    
    large_lams = {
        con: val for con, val in zip(conlist, lam) if abs(val) >= tol
    }
    
    # print("Candidate degenerate constraints:")
    # for con, val in large_lams.items():
    #     print(f" {con.name}: {val}")
    
    # Nonzeros
    if jacnz:
        jacnz = {
            conlist[i].name: {
                varlist[j].name: val for j, val in enumerate(jac.toarray()[i]) if val != 0
            } for i, _ in enumerate(conlist)
        }
    
    # Candidate degenerate set
    cds = list(large_lams.keys())
    
    return cds


def get_large_residuals(m, tol=1e+04):
    from idaes.core.util.model_statistics import (
        large_residuals_set, number_large_residuals)
    rs = large_residuals_set(m, tol=tol)
    rn = number_large_residuals(m, tol=tol)


def add_penalty_formulation(m, tol=1e+03):
    # Adding penalty terms
    from pyomo.core.expr.relational_expr import EqualityExpression
    import cloudpickle as pickle
    # cons = [con for con in m.component_data_objects(
    #     ctype=Constraint, active=True)]
    # cons = [con for con in m.component_data_objects(
    #     ctype=pyo.Constraint) if isinstance(con.expr, EqualityExpression)]
    try:
        cnames = pickle.load(open('candidate_degenerate_equations.pkl', 'rb'))
        cons = [
            c for c in m.component_data_objects(ctype=pyo.Constraint)
            if c.name in cnames
        ]
    except:
        cons = get_large_duals(m, tol=tol)
    cidx = list(range(1, len(cons)+1))
    print(f'# constraints = {len(cidx)}')

    # Add artificial variables for candidate degenerate equations
    if len(cidx) == 0:
        print('No large duals above specified threshold. Use a tolerance '
              f'smaller than {tol}.')
        return None

    m.j = pyo.Set(initialize=cidx)
    m.p = pyo.Var(m.j, bounds=(0, None), initialize=0.01)
    m.n = pyo.Var(m.j, bounds=(0, None), initialize=0.01)
    
    exp_list = []
    for j, con in zip(cidx, cons):
        # old_body = con.body
        new_con_exp = con.body == m.p[j] - m.n[j]
        exp_list.append(new_con_exp)
    
    m.conlist = pyo.ConstraintList()
    for j, expr in enumerate(exp_list):
        cons[j].deactivate()
        m.conlist.add(expr)
    
    # def _p_lb(m, j):
    #     return m.p[j] >= 0
    # m.p_lb = Constraint(m.j, rule=_p_lb)
    
    # def _n_lb(m, j):
    #     return m.n[j] >= 0
    # m.n_lb = Constraint(m.j, rule=_n_lb)
    
     # Modify objective
    penalty_expr = 1e+02 * sum([m.p[j] + m.n[j] for j in m.j])
    m.obj.expr += penalty_expr

    return None
    

def initialize_model_with_petsc(m):
    idaeslog.solver_log.tee = True
    results = petsc.petsc_dae_by_time_element(
        m,
        time=m.fs.time,
        # timevar=m.fs.timevar,
        keepfiles=True,
        symbolic_solver_labels=True,
        ts_options={
            "--ts_type": "beuler",
            "--ts_dt": 1,
            "--ts_rtol": 1e-3,
            # "--ts_adapt_clip":"0.001,300",
            # "--ksp_monitor":"",
            "--ts_adapt_dt_min": 1e-6,
            # "--ts_adapt_dt_max": 300,
            "--ksp_rtol": 1e-10,
            "--snes_type": "newtontr",
            # "--ts_max_reject": 200,
            # "--snes_monitor":"",
            "--ts_monitor": "",
            "--ts_save_trajectory": 1,
            "--ts_trajectory_type": "visualization",
            "--ts_max_snes_failures": 1000,
            # "--show_cl":"",
        },
        skip_initial=False,
        initial_solver="ipopt",
        # vars_stub="soec_flowsheet_prototype",
        # trajectory_save_prefix="soec_flowsheet_prototype",
    )
    return results


def run_simulation(time_set=nmpc_params["time_set_PI"],
                   dynamic_simulation=True,
                   optimize_steady_state=True,
                   operating_scenario=OperatingScenario.maximum_production):

    def set_indexed_variable_bounds(var, bounds):
        for idx, subvar in var.items():
            subvar.bounds = bounds

    use_idaes_solver_configuration_defaults()
    idaes.cfg.ipopt.options.nlp_scaling_method = "user-scaling"
    idaes.cfg.ipopt["options"]["linear_solver"] = "ma57"
    idaes.cfg.ipopt.options.OF_ma57_automatic_scaling = "yes"
    idaes.cfg.ipopt["options"]["max_iter"] = 250
    idaes.cfg.ipopt["options"]["halt_on_ampl_error"] = "no"

    m = pyo.ConcreteModel()
    if dynamic_simulation:
        setpoints = ["maximum_H2", "maximum_H2",
                     "minimum_H2", "minimum_H2",
                     "maximum_H2", "maximum_H2"]

        m.fs = SoecStandaloneFlowsheet(
            dynamic=True,
            time_set=time_set,
            time_units=pyo.units.s,
            thin_electrolyte_and_oxygen_electrode=True,
            has_gas_holdup=False,
            include_interconnect=True,
        )
    else:
        m.fs = SoecStandaloneFlowsheet(
            dynamic=False,
            thin_electrolyte_and_oxygen_electrode=True,
            has_gas_holdup=False,
            include_interconnect=True,
        )

    iscale.calculate_scaling_factors(m)
    solver = pyo.SolverFactory("ipopt")

    if dynamic_simulation:
        m.fs.deactivate_shortcut()

        antiwindup = ControllerAntiwindupType.BACK_CALCULATION
        inner_controller_pairs = ComponentMap()
        inner_controller_pairs[m.fs.feed_heater.electric_heat_duty] = (
            "feed_heater_inner_controller",
            m.fs.soc_module.fuel_inlet.temperature,
            ControllerType.PI,
            ControllerMVBoundType.SMOOTH_BOUND,
            antiwindup,
        )
        inner_controller_pairs[m.fs.sweep_heater.electric_heat_duty] = (
            "sweep_heater_inner_controller",
            m.fs.soc_module.oxygen_inlet.temperature,
            ControllerType.PI,
            ControllerMVBoundType.SMOOTH_BOUND,
            antiwindup,
        )
        m.fs.add_controllers(inner_controller_pairs)

        variable_pairs = ComponentMap()
        variable_pairs[m.fs.feed_heater_inner_controller.setpoint] = (
            "feed_heater_outer_controller",
            m.fs.soc_module.fuel_outlet.temperature,
            ControllerType.P,
            ControllerMVBoundType.NONE,
            ControllerAntiwindupType.NONE,
        )
        variable_pairs[m.fs.sweep_heater_inner_controller.setpoint] = (
            "sweep_heater_outer_controller",
            m.fs.soc_module.oxygen_outlet.temperature,
            ControllerType.P,
            ControllerMVBoundType.NONE,
            ControllerAntiwindupType.NONE,
        )
        variable_pairs[m.fs.soc_module.potential_cell] = (
            "voltage_controller",
            m.fs.soc_module.fuel_outlet_mole_frac_comp_H2,
            ControllerType.PI,
            ControllerMVBoundType.SMOOTH_BOUND,
            antiwindup,
        )
        # variable_pairs[m.fs.feed_translator.inlet.flow_mol] = (
        #     "h2_production_rate_controller",
        #     m.fs.h2_mass_production,
        #     ControllerType.PI,
        #     ControllerMVBoundType.SMOOTH_BOUND,
        # )
        variable_pairs[m.fs.sweep_recycle_split.recycle_ratio] = (
            "sweep_recycle_controller",
            m.fs.soc_module.oxygen_outlet.temperature,
            ControllerType.P,
            ControllerMVBoundType.SMOOTH_BOUND,
            ControllerAntiwindupType.NONE,
        )
        # variable_pairs[m.fs.condenser_hx.cold_side_inlet.flow_mol] = (
        #     "condenser_controller",
        #     m.fs.condenser_hx.hot_side_outlet.temperature,
        #     ControllerType.P,
        #     ControllerMVBoundType.SMOOTH_BOUND,
        # )
        m.fs.add_controllers(variable_pairs)

        # K = 0
        K = 0.25 * 10e4
        tau_I = 30*60
        m.fs.feed_heater_inner_controller.gain_p.fix(K)
        m.fs.feed_heater_inner_controller.gain_i.fix(K/tau_I)
        m.fs.feed_heater_inner_controller.mv_lb = 0
        m.fs.feed_heater_inner_controller.mv_ub = 10e6
        m.fs.feed_heater_inner_controller.smooth_eps = 1000
        if antiwindup == ControllerAntiwindupType.BACK_CALCULATION:
            m.fs.feed_heater_inner_controller.gain_b.fix(0.5/tau_I)

        # K = 0
        K = 0.25 * 20e4
        tau_I = 30*60
        m.fs.sweep_heater_inner_controller.gain_p.fix(K)
        m.fs.sweep_heater_inner_controller.gain_i.fix(K/tau_I)
        m.fs.sweep_heater_inner_controller.mv_lb = 0
        m.fs.sweep_heater_inner_controller.mv_ub = 10e6
        m.fs.sweep_heater_inner_controller.smooth_eps = 1000
        if antiwindup == ControllerAntiwindupType.BACK_CALCULATION:
            m.fs.sweep_heater_inner_controller.gain_b.fix(0.5/tau_I)
        
        # K = 0.75
        K = 0
        tau_I = 60*60
        m.fs.feed_heater_outer_controller.gain_p.fix(K)
        # m.fs.feed_heater_outer_controller.gain_i.fix(K/tau_I)
        # m.fs.feed_heater_outer_controller.mv_lb = 0
        # m.fs.feed_heater_outer_controller.mv_ub = 4e6
        # m.fs.feed_heater_outer_controller.smooth_eps = 0.1

        # K = 0.75
        K = 0
        tau_I = 60*60
        m.fs.sweep_heater_outer_controller.gain_p.fix(K)
        # m.fs.sweep_heater_outer_controller.gain_i.fix(K/tau_I)
        # m.fs.sweep_heater_outer_controller.mv_lb = 0
        # m.fs.sweep_heater_outer_controller.mv_ub = 12e6
        # m.fs.sweep_heater_outer_controller.smooth_eps = 0.01

        # K = 0
        K = -2
        tau_I = 240
        m.fs.voltage_controller.gain_p.fix(K)
        m.fs.voltage_controller.gain_i.fix(K/tau_I)
        m.fs.voltage_controller.mv_lb = 0.7
        m.fs.voltage_controller.mv_ub = 1.6
        m.fs.voltage_controller.smooth_eps = 0.01
        if antiwindup == ControllerAntiwindupType.BACK_CALCULATION:
            m.fs.voltage_controller.gain_b.fix(0.5/tau_I)

        K = 0
        # K = -0.5 * 0.025
        # tau_I = 1200
        m.fs.sweep_recycle_controller.gain_p.fix(K)
        #m.fs.sweep_recycle_controller.gain_i.fix(K / tau_I)
        m.fs.sweep_recycle_controller.mv_lb = 0.01
        m.fs.sweep_recycle_controller.mv_ub = 2
        m.fs.sweep_recycle_controller.smooth_eps = 1e-3

        # K = 200
        # tau_I = 20 * 60
        # m.fs.h2_production_rate_controller.gain_p.fix(K)
        # m.fs.h2_production_rate_controller.gain_i.fix(K / tau_I)
        # m.fs.h2_production_rate_controller.mv_lb = 0
        # m.fs.h2_production_rate_controller.mv_ub = 1500
        # m.fs.h2_production_rate_controller.smooth_eps = 1

        create_ramping_eqns(m.fs, m.fs.manipulated_variables, 1)

        time_nfe = len(m.fs.time) - 1
        pyo.TransformationFactory("dae.finite_difference").apply_to(
            m.fs, nfe=time_nfe, wrt=m.fs.time, scheme="BACKWARD"
        )
        apply_custom_variable_scaling(m)
        apply_custom_constraint_scaling(m)

        if operating_scenario == OperatingScenario.minimum_production:
            ms.from_json(
                m,
                fname="../../min_production.json.gz",
                wts=ms.StoreSpec.value(),
            )
        elif operating_scenario == OperatingScenario.maximum_production:
            ms.from_json(
                m,
                fname="../../max_production.json.gz",
                wts=ms.StoreSpec.value(),
            )
        elif operating_scenario == OperatingScenario.power_mode:
            ms.from_json(
                m,
                fname="../../power_mode.json.gz",
                wts=ms.StoreSpec.value(),
            )
        elif operating_scenario == OperatingScenario.neutral:
            ms.from_json(
                m,
                fname="../../neutral.json.gz",
                wts=ms.StoreSpec.value(),
            )

        m.fs.feed_heater.electric_heat_duty.unfix()
        m.fs.sweep_heater.electric_heat_duty.unfix()

        # Copy initial conditions to rest of model for initialization
        m.fs.fix_initial_conditions()

        alias_dict = ComponentMap()
        alias_dict[m.fs.voltage_controller.mv_ref] = "potential"
        alias_dict[m.fs.voltage_controller.setpoint] = "soc_fuel_outlet_mole_frac_comp_H2"
        alias_dict[m.fs.soc_module.fuel_outlet_mole_frac_comp_H2] = "soc_fuel_outlet_mole_frac_comp_H2"
        alias_dict[m.fs.makeup_mix.makeup.flow_mol] = "makeup_feed_rate"
        alias_dict[m.fs.sweep_blower.inlet.flow_mol] = "sweep_feed_rate"
        alias_dict[m.fs.feed_heater_inner_controller.mv_ref] = "feed_heater_duty"
        alias_dict[m.fs.feed_heater_outer_controller.mv_ref] = "feed_heater_outlet_temperature"
        alias_dict[m.fs.feed_heater_outer_controller.setpoint] = "fuel_outlet_temperature"
        alias_dict[m.fs.sweep_heater_inner_controller.mv_ref] = "sweep_heater_duty"
        alias_dict[m.fs.sweep_heater_outer_controller.mv_ref] = "sweep_heater_outlet_temperature"
        alias_dict[m.fs.sweep_heater_outer_controller.setpoint] = "sweep_outlet_temperature"
        alias_dict[m.fs.makeup_mix.makeup_mole_frac_comp_H2] = "makeup_mole_frac_comp_H2"
        alias_dict[m.fs.makeup_mix.makeup_mole_frac_comp_H2O] = "makeup_mole_frac_comp_H2O"
        alias_dict[m.fs.condenser_hx.cold_side_inlet.flow_mol] = "cooling_water_feed"

        alias_dict[m.fs.sweep_recycle_controller.mv_ref] = "sweep_recycle_ratio"
        alias_dict[m.fs.sweep_recycle_controller.setpoint] = "sweep_outlet_temperature"

        alias_dict[m.fs.feed_recycle_split.recycle_ratio] = "fuel_recycle_ratio"
        alias_dict[m.fs.condenser_split.recycle_ratio] = "vgr_recycle_ratio"

        df = pd.read_csv("./../../soec_flowsheet_operating_conditions.csv", index_col=0)
        t0 = m.fs.time.first()
        for var in m.fs.manipulated_variables:
            shortname = var.name.split(".")[-1]
            alias = alias_dict[var]
            blk = var.parent_block()
            v_ramp = getattr(blk, shortname + "_ramp_rate")
            var[t0].fix(float(df[alias][setpoints[0]]))
            for i, t in enumerate(time_set):
                v_ramp[t].fix(float(
                    (df[alias][setpoints[i]] - df[alias][setpoints[i-1]])
                    / (time_set[i] - time_set[i-1])
                ))

        # Need to initialize the setpoint for the inner controller or else it starts with the default value 0.5.
        m.fs.feed_heater_inner_controller.setpoint[0].value = m.fs.feed_heater_outer_controller.mv_ref[0].value
        m.fs.sweep_heater_inner_controller.setpoint[0].value = m.fs.sweep_heater_outer_controller.mv_ref[0].value

        idaeslog.solver_log.tee = True
        results = initialize_model_with_petsc(m)
        # traj = results.trajectory

        save_results(m, np.array(m.fs.time)[1:], results.trajectory, "PI_ramping")

        # Make plots
        # plot_results("PI_ramping", include_PI=True)
        
        # # Save dynamic results
        # if operating_scenario == OperatingScenario.minimum_production:
        #     ms.to_json(m, fname="min_production_dynamic.json.gz")
        # elif operating_scenario == OperatingScenario.maximum_production:
        #     ms.to_json(m, fname="max_production_dynamic.json.gz")

    else:
        t0 = time.time()
        if operating_scenario == OperatingScenario.power_mode:
            m.fs.initialize_build(fuel_cell_mode=True, outlvl=idaeslog.DEBUG)
        else:
            m.fs.initialize_build(fuel_cell_mode=False, outlvl=idaeslog.DEBUG)
        # assert False
        optimization_constraints = []
        if optimize_steady_state:
            for var in m.fs.manipulated_variables:
                var.unfix()
            m.fs.set_performance_bounds()
            m.fs.make_performance_constraints()
            m.fs.cooling_water_penalty = pyo.Param(initialize=1e-5, mutable=True)
            if operating_scenario == OperatingScenario.power_mode:
                m.fs.makeup_mix.makeup.mole_frac_comp[0, "H2O"].fix(0.03)
                m.fs.makeup_mix.makeup.mole_frac_comp[0, "H2"].fix(0.969)
                set_indexed_variable_bounds(m.fs.condenser_flash.heat_duty, (None, 0))
                set_indexed_variable_bounds(m.fs.condenser_split.split_fraction, (0.00001, 1))
                m.fs.feed_recycle_split.out.mole_frac_comp[0, "H2"].bounds = (0.1, 0.75)
                set_indexed_variable_bounds(m.fs.condenser_split.recycle_ratio, (0.000001, 50))
                @m.fs.Constraint(m.fs.time)
                def current_density_average_limit_eqn(b, t):
                    return sum([b.soc_module.solid_oxide_cell.current_density[t, iz]
                                for iz in b.soc_module.solid_oxide_cell.iznodes]
                               ) / 10 == 4e3

                optimization_constraints.append(m.fs.current_density_average_limit_eqn)
                scale_indexed_constraint(m.fs.current_density_average_limit_eqn, 1e-3)

                # Make sure the steady-state value is nonzero so we don't have trouble with antiwindup
                m.fs.feed_heater.electric_heat_duty.fix(1e5)
                m.fs.sweep_heater.electric_heat_duty.fix(1e5)
                set_indexed_variable_bounds(m.fs.soc_module.solid_oxide_cell.current_density, (-1.3e4, 5.2e3))

                m.fs.condenser_flash.control_volume.properties_out[:].temperature.fix(273.15+50)
                m.fs.obj = pyo.Objective(
                    expr=(
                            1e-8 * m.fs.total_electric_power[0] + m.fs.h2_mass_consumption[0]
                    )
                )
            elif operating_scenario == OperatingScenario.neutral:
                m.fs.makeup_mix.makeup.mole_frac_comp[0, "H2O"].fix(0.5)
                m.fs.makeup_mix.makeup.mole_frac_comp[0, "H2"].fix(0.499)
                @m.fs.Constraint(m.fs.time)
                def current_density_average_limit_eqn(b, t):
                    return sum([b.soc_module.solid_oxide_cell.current_density[t, iz]
                                for iz in b.soc_module.solid_oxide_cell.iznodes]
                               ) / 10 == 0

                optimization_constraints.append(m.fs.current_density_average_limit_eqn)
                scale_indexed_constraint(m.fs.current_density_average_limit_eqn, 1e-3)

                m.fs.feed_recycle_split.recycle_ratio.fix(1)
                m.fs.sweep_recycle_split.recycle_ratio.fix(1)
                m.fs.condenser_split.recycle_ratio.fix(1)
                m.fs.sweep_blower.inlet.flow_mol.fix(2261.016)
                m.fs.feed_medium_exchanger.tube_inlet.flow_mol.fix(1508.836)
                # set_indexed_variable_bounds(m.fs.condenser_split.recycle_ratio, (0.000001, 50))

                m.fs.condenser_flash.control_volume.properties_out[:].temperature.fix(273.15 + 50)

                m.fs.obj = pyo.Objective(
                    expr=(
                            1e-8 * m.fs.total_electric_power[0] + m.fs.h2_mass_consumption[0]
                    )
                )

            else:
                m.fs.makeup_mix.makeup.mole_frac_comp[0, "H2O"].fix(0.999 - 1e-14)
                m.fs.makeup_mix.makeup.mole_frac_comp[0, "H2"].fix(1e-14)
                if operating_scenario == OperatingScenario.minimum_production:
                    m.fs.h2_mass_production.fix(0.4)
                elif operating_scenario == OperatingScenario.maximum_production:
                    m.fs.h2_mass_production.fix(2)
                m.fs.feed_recycle_split.out.mole_frac_comp[0, "H2O"].bounds = (0.25, 0.4)
                set_indexed_variable_bounds(m.fs.soc_module.solid_oxide_cell.current_density, (-1.3e4, 5.2e3))

                m.fs.condenser_split.split_fraction[:, "recycle"].fix(0.0001)
                m.fs.condenser_split.split_fraction[:, "out"].value = 0.9999
                m.fs.condenser_flash.control_volume.properties_out[:].temperature.fix(273.15 + 50)
                m.fs.obj = pyo.Objective(
                    expr=(
                        1e-8 * m.fs.total_electric_power[0]
                    )
                )

            @m.fs.Constraint(m.fs.time)
            def sweep_concentration_eqn(b, t):
                return b.sweep_recycle_split.mixed_state[t].mole_frac_comp["O2"] <= 0.35


            @m.fs.Constraint(m.fs.time)
            def min_h2_feed_eqn(b, t):
                return b.feed_recycle_mix.mixed_state[t].mole_frac_comp["H2"] >= 0.05


            optimization_constraints.append(m.fs.sweep_concentration_eqn)
            optimization_constraints.append(m.fs.min_h2_feed_eqn)

        else:
            alias_dict = ComponentMap()
            alias_dict[m.fs.soc_module.potential_cell] = "potential"
            alias_dict[m.fs.feed_translator.inlet.flow_mol] = "steam_feed_rate"
            alias_dict[m.fs.sweep_blower.inlet.flow_mol] = "sweep_feed_rate"
            alias_dict[m.fs.feed_heater.electric_heat_duty] = "feed_heater_duty"
            alias_dict[m.fs.sweep_heater.electric_heat_duty] = "sweep_heater_duty"
            alias_dict[m.fs.feed_recycle_split.recycle_ratio] = "fuel_recycle_ratio"
            alias_dict[m.fs.sweep_recycle_split.recycle_ratio] = "sweep_recycle_ratio"
            alias_dict[m.fs.feed_hydrogen_water_ratio] = "feed_hydrogen_water_ratio"

            df = pd.read_csv("soec_flowsheet_operating_conditions_full_control.csv", index_col=0)

            for var in m.fs.manipulated_variables:
                shortname = var.name.split(".")[-1]
                alias = alias_dict[var]
                var[t0].fix(float(df[alias][setpoints[0]]))

        optimization_constraints.append(m.fs.thermal_gradient_eqn_1)
        optimization_constraints.append(m.fs.thermal_gradient_eqn_2)
        optimization_constraints.append(m.fs.thermal_gradient_eqn_3)
        optimization_constraints.append(m.fs.thermal_gradient_eqn_4)

        jac_unscaled, jac_scaled, nlp = iscale.constraint_autoscale_large_jac(m)
        results = solver.solve(m, tee=True)
        pyo.assert_optimal_termination(results)

        m.fs.h2_mass_production.unfix()
        m.fs.condenser_split.split_fraction.unfix()
        m.fs.feed_medium_exchanger.tube_inlet.flow_mol.unfix()
        m.fs.condenser_flash.control_volume.properties_out[:].temperature.unfix()

        set_indexed_variable_bounds(m.fs.feed_recycle_split.split_fraction, (0, 1))
        set_indexed_variable_bounds(m.fs.sweep_recycle_split.split_fraction, (0, 1))
        set_indexed_variable_bounds(m.fs.condenser_split.split_fraction, (0, 1))

        m.fs.initialize_condenser_hx(outlvl=idaeslog.DEBUG)
        for con in optimization_constraints:
            con.deactivate()
        for var in m.fs.manipulated_variables:
            var.fix()
        results = solver.solve(m, tee=True)
        pyo.assert_optimal_termination(results)



        m.fs.write_pfd(fname="soec_dynamic_flowsheet.svg")

        if operating_scenario == OperatingScenario.minimum_production:
            ms.to_json(m, fname="min_production.json.gz")
        elif operating_scenario == OperatingScenario.maximum_production:
            ms.to_json(m, fname="max_production.json.gz")
        elif operating_scenario == OperatingScenario.power_mode:
            ms.to_json(m, fname="power_mode.json.gz")
        elif operating_scenario == OperatingScenario.neutral:
            ms.to_json(m, fname="neutral.json.gz")

        # for var in m.fs.manipulated_variables:
        #    print(var.name + f": {var.value}")
        #    var.pprint()
        t1 = time.time()
        print(f"Finished in {t1-t0} seconds")
        print(f"Hydrogen production rate: {pyo.value(m.fs.h2_mass_production[0])} kg/s")
        print(f"Cell potential: {pyo.value(m.fs.soc_module.potential_cell[0])} V")
        print(f"SOC fuel outlet H2 mole frac: {pyo.value(m.fs.soc_module.fuel_outlet_mole_frac_comp_H2[0])}")
        print(f"Makeup feed rate: {pyo.value(m.fs.makeup_mix.makeup.flow_mol[0])} mol/s")
        #print(f"Hydrogen/water ratio: {pyo.value(m.fs.feed_hydrogen_water_ratio[0])}")
        print(f"Sweep feed rate: {pyo.value(m.fs.sweep_blower.inlet.flow_mol[0])} mol/s")
        print(f"Fuel-side heat duty: {pyo.value(m.fs.feed_heater.electric_heat_duty[0])} W")
        print(f"Fuel-side inlet temperature: {pyo.value(m.fs.soc_module.fuel_inlet.temperature[0])} K")
        # print(f"Fuel-side inlet temperature: {pyo.value(m.fs.stack_fuel_inlet_temperature[0])} K")
        print(f"Fuel side outlet temperature: {pyo.value(m.fs.soc_module.fuel_outlet.temperature[0])} K")
        print(f"Sweep-side heat duty: {pyo.value(m.fs.sweep_heater.electric_heat_duty[0])} W")
        print(f"Sweep-side inlet temperature: {pyo.value(m.fs.soc_module.oxygen_inlet.temperature[0])} K")
        # print(f"Sweep-side inlet temperature: {pyo.value(m.fs.stack_sweep_inlet_temperature[0])} K")
        print(f"Oxygen side outlet temperature: {pyo.value(m.fs.soc_module.oxygen_outlet.temperature[0])} K")
        print(f"Stack core temperature: {pyo.value(m.fs.stack_core_temperature[0])} K")
        print(f"Fuel recycle ratio: {pyo.value(m.fs.feed_recycle_split.recycle_ratio[0])}")
        print(f"Sweep recycle ratio: {pyo.value(m.fs.sweep_recycle_split.recycle_ratio[0])}")
        print(f"Sweep oxygen outlet: {pyo.value(m.fs.sweep_recycle_split.mixed_state[0].mole_frac_comp['O2'])}")
        print(f"Feed hydrogen inlet: {pyo.value(m.fs.feed_recycle_mix.mixed_state[0].mole_frac_comp['H2'])}")

        print(f"Vent gas recirculation recycle ratio {pyo.value(m.fs.condenser_split.recycle_ratio[0])}")
        print(f"Condenser cooling water feed rate: {pyo.value(m.fs.condenser_hx.cold_side_inlet.flow_mol[0])} mol/s")
        print(f"Condenser hydrogen outlet temperature: {pyo.value(m.fs.condenser_hx.hot_side_outlet.temperature[0])} K")
        # print(f"Condenser heat duty: {pyo.value(m.fs.condenser_flash.heat_duty[0])} W")
        # print(f"Condenser hydrogen outlet temperature: {pyo.value(m.fs.condenser_flash.vap_outlet.temperature[0])} K")


        soc = m.fs.soc_module.solid_oxide_cell

        plt.figure()
        plt.plot(np.array(soc.iznodes), np.array(
            [pyo.value(soc.current_density[0, i]) for i in
             soc.iznodes]) / 10, label="Current Density")
        plt.plot(np.array(soc.iznodes), np.ones((10, 1)) * sum(
            [pyo.value(soc.current_density[0, i]) for i in
             soc.iznodes]) / 100, "--", label="Average Current Density")

        plt.xlabel("z node", fontsize=14)
        plt.ylabel(r"Current Density ($mA/cm^2$)", fontsize=14)
        plt.title("Current Density", fontsize=16)
        
        plt.figure()
        plt.plot(np.array(soc.iznodes), np.array(
            [pyo.value(soc.fuel_triple_phase_boundary.potential_nernst[0, iz]
                       + soc.oxygen_triple_phase_boundary.potential_nernst[0, iz])
             for iz in soc.iznodes]),
            label="Nernst Potential"
         )

        plt.xlabel("z node", fontsize=14)
        plt.ylabel(r"Nernst Potential (V)", fontsize=14)
        plt.title("Nernst Potential", fontsize=16)

        plt.figure()
        plt.plot(np.array(soc.iznodes), np.array(
            [pyo.value(soc.fuel_electrode.temperature[0, 1, i]) for i in
             soc.iznodes]), label="Temperature")
        plt.plot(np.array(soc.iznodes), np.ones((10, 1)) * sum(
            [pyo.value(soc.fuel_electrode.temperature[0, 1, i]) for i in
             soc.iznodes]) / 10, "--", label="Average Temperature")

        plt.legend()
        plt.xlabel("z node", fontsize=14)
        plt.ylabel("Temperature (K)", fontsize=14)
        plt.title("PEN Temperature", fontsize=16)

        plt.figure()
        plt.plot(np.array(soc.iznodes), np.array(
            [pyo.value(soc.fuel_electrode.dtemperature_dz[0, 1, i]) for i in
             soc.iznodes])/100)
        plt.xlabel("z node", fontsize=14)
        plt.ylabel(r"$\frac{dT}{dz}$ (K/cm)", fontsize=14)
        plt.title("PEN Temperature Gradient", fontsize=16)

        plt.show()

        results = None
        
    return m, results















