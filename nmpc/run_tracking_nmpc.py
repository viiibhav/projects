import sys
sys.path.append('../')
sys.path.append('../../')
import pandas as pd
import numpy as np

import pyomo.environ as pyo
from pyomo.environ import (ConcreteModel, Var, Constraint, Param, Expression,
                           value, Objective, Suffix)
from pyomo.opt import SolverStatus, TerminationCondition, ProblemFormat
# import pyomo.dae.flatten as flatten
# from generate_sliced_components import generate_sliced_components
# flatten.generate_sliced_components = generate_sliced_components
from pyomo.dae.flatten import flatten_dae_components, generate_sliced_components
# from scale_discretization_equations import scale_discretization_equations
from pyomo.common.config import ConfigBlock
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
import idaes
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
idaeslog.getLogger("idaes.core.util.scaling").setLevel(idaeslog.ERROR)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import petsc
from idaes.core.solvers import use_idaes_solver_configuration_defaults
import idaes.core.util.model_serializer as ms
from pyomo.dae import ContinuousSet, DerivativeVar
from soec_dynamic_flowsheet_mk2 import SoecStandaloneFlowsheet
import matplotlib.pyplot as plt
import cloudpickle as pickle
from nmpc_helper import (
    make_tracking_objective,
    apply_custom_variable_scaling,
    apply_custom_constraint_scaling,
    initialize_model_with_petsc,
    get_time_coordinates,
    get_h2_production_target,
    get_tracking_targets,
    get_tracking_variables,
    get_manipulated_variables,
    get_controlled_variables,
    check_scaling,
    hunt_degeneracy,
    shift_dae_vars_by_dt,
    check_incidence_analysis,
    get_large_duals,
    get_large_residuals,
    add_penalty_formulation,
    alias_dict,
)


if __name__ == "__main__":

    # Set solver options
    use_idaes_solver_configuration_defaults()
    idaes.cfg.ipopt.options.nlp_scaling_method = "user-scaling"
    idaes.cfg.ipopt.options.linear_solver = "ma57"
    idaes.cfg.ipopt.options.OF_ma57_pivot_order = 5  # 2, 3 or 5 (default)
    idaes.cfg.ipopt.options.OF_ma57_pivtol = 1e-06  # default = 1e-08
    idaes.cfg.ipopt.options.OF_ma57_automatic_scaling = "yes"
    idaes.cfg.ipopt.options.OF_print_info_string = "yes"
    idaes.cfg.ipopt.options.OF_warm_start_init_point = "yes"
    idaes.cfg.ipopt.options.OF_warm_start_mult_bound_push = 1e-06
    idaes.cfg.ipopt.options.OF_warm_start_bound_push = 1e-06
    idaes.cfg.ipopt.options.max_iter = 200
    idaes.cfg.ipopt.options.halt_on_ampl_error = "no"
    idaes.cfg.ipopt.options.bound_relax_factor = 1e-08  # default = 1e-08
    idaes.cfg.ipopt.options.bound_push = 1e-06  # default = 1e-02
    idaes.cfg.ipopt.options.mu_init = 1e-06  # default = 1e-01
    idaes.cfg.ipopt.options.tol = 1e-08  # default = 1e-08
    # idaes.cfg.declare("ipopt_l1", ConfigBlock(implicit=True))
    # idaes.cfg.ipopt_l1.declare("options", ConfigBlock(implicit=True))
    # idaes.cfg.ipopt_l1.options.nlp_scaling_method = "user-scaling"
    # idaes.cfg.ipopt_l1.options.linear_solver = "ma57"
    # idaes.cfg.ipopt_l1.options.max_iter = 250    
    # idaes.cfg.ipopt_l1.options.bound_relax_factor = 1e-08  # default = 1e-08
    # idaes.cfg.ipopt_l1.options.bound_push = 1e-06  # default = 1e-02
    # idaes.cfg.ipopt_l1.options.mu_init = 1e-06  # default = 1e-01
    # idaes.cfg.ipopt_l1.options.tol = 1e-08  # default = 1e-08
    solver = pyo.SolverFactory("ipopt")
    
    # Get time coordinates
    nmpc_params = get_time_coordinates()


    def fix_manipulated_variables(m):
        manipulated_variables = get_manipulated_variables(m)
        for var, alias in manipulated_variables.items():
            var[:].fix()


    def unfix_manipulated_variables(m):
        manipulated_variables = get_manipulated_variables(m)
        for var, alias in manipulated_variables.items():
            var[:].unfix()


    # def rectify_initialization(m):
    #     m.fs.condenser_hx.hot_side.properties[:]\
    #         .temperature_dew['Liq', 'Vap'].set_value(700)


    def create_model(time_set, nfe, plant=True):
        m = ConcreteModel()
        m.fs = SoecStandaloneFlowsheet(
            dynamic=True,
            time_set=time_set,
            time_units=pyo.units.s,
            thin_electrolyte_and_oxygen_electrode=True,
            has_gas_holdup=False,
            include_interconnect=True,
        )
        
        m.fs.p = pyo.Var(m.fs.time,
                          initialize=0,
                          domain=pyo.NonNegativeReals)
        # m.fs.n = pyo.Var(m.fs.time,
        #                   initialize=0,
        #                   domain=pyo.NonNegativeReals)
        m.fs.q = pyo.Var(m.fs.time,
                          initialize=0,
                          domain=pyo.NonNegativeReals)
        # m.fs.r = pyo.Var(m.fs.time,
        #                   initialize=0,
        #                   domain=pyo.NonNegativeReals)
        # m.fs.p1 = pyo.Var(m.fs.time,
        #                   initialize=0,
        #                   domain=pyo.NonNegativeReals)
        # m.fs.n1 = pyo.Var(m.fs.time,
        #                   initialize=0,
        #                   domain=pyo.NonNegativeReals)
        # m.fs.n2 = pyo.Var(m.fs.time,
        #                   initialize=0,
        #                   domain=pyo.NonNegativeReals)
        # m.fs.n3 = pyo.Var(m.fs.time,
        #                   initialize=0,
        #                   domain=pyo.NonNegativeReals)

        if not plant:
            # dTdz_electrode_lim = 675
            
            # @m.fs.Constraint(m.fs.time)
            # def makeup_mole_frac_eqn1(b, t):
            #     return b.makeup_mix.makeup_mole_frac_comp_H2[t] == 1e-14 + b.p[t]
            
            # @m.fs.Constraint(m.fs.time)
            # def makeup_mole_frac_eqn2(b, t):
            #     return b.makeup_mix.makeup_mole_frac_comp_H2O[t] == \
            #         0.999 - 1e-14 - b.n[t]
            
            @m.fs.Constraint(m.fs.time)
            def vgr_ratio_eqn(b, t):
                return b.condenser_split.recycle_ratio[t] == 1e-4 + b.q[t]
            
            @m.fs.Constraint(m.fs.time)
            def makeup_mole_frac_sum_eqn(b, t):
                return b.makeup_mix.makeup_mole_frac_comp_H2[t] + \
                    b.makeup_mix.makeup_mole_frac_comp_H2O[t] == 0.999 - b.p[t]
            
            # @m.fs.Constraint(m.fs.time)
            # def condenser_outlet_temp_eqn(b, t):
            #     return b.condenser_flash.control_volume.properties_out[t] \
            #         .temperature == 273.15 + 50 + b.p1[t] - b.n1[t]
    
            # @m.fs.Constraint(m.fs.time)
            # def feed_recycle_ratio_eqn(b, t):
            #     return b.feed_recycle_split.recycle_ratio[t] == 0.999 - b.n2[t]
            
            # @m.fs.Constraint(m.fs.time)
            # def sweep_recycle_ratio_eqn(b, t):
            #     return b.sweep_recycle_split.recycle_ratio[t] == 0.999 - b.n3[t]
    
            # # @soec.fuel_electrode.Constraint(
            #     # m.fs.time, soec.fuel_electrode.ixnodes, soec.fuel_electrode.iznodes)
            # # def dTdz_electrode_UB_rule(b, t, ix, iz):
            # #     return b.dtemperature_dz[t, ix, iz] - dTdz_electrode_lim <= b.p[t, ix, iz]
            
            # # @soec.fuel_electrode.Constraint(
            # #     m.fs.time, soec.fuel_electrode.ixnodes, soec.fuel_electrode.iznodes)
            # # def dTdz_electrode_LB_rule(b, t, ix, iz):
            # #     return -b.dtemperature_dz[t, ix, iz] - dTdz_electrode_lim <= b.n[t, ix, iz]

        iscale.calculate_scaling_factors(m)

        pyo.TransformationFactory("dae.finite_difference").apply_to(
            m.fs, nfe=nfe, wrt=m.fs.time, scheme="BACKWARD"
        )

        # Initialize model
        ms.from_json(m,
                     fname="../../max_production.json.gz",
                     wts=ms.StoreSpec.value())

        # Copy initial conditions to rest of model for initialization
        if not plant:
            regular_vars, time_vars = flatten_dae_components(m, m.fs.time, pyo.Var)
            for t in m.fs.time:
                for v in time_vars:
                    if not v[t].fixed:
                        if v[m.fs.time.first()].value is None:
                            v[t].set_value(0.0)
                        else:
                            v[t].set_value(pyo.value(v[m.fs.time.first()]))

        # Fix initial conditions
        m.fs.fix_initial_conditions()

        # Fix DOF issue
        # m.fs.condenser_hx.hot_side.properties[:, 1].temperature.fix()

        manipulated_variables = get_manipulated_variables(m)        
        # import pdb; pdb.set_trace()
        for v, _ in manipulated_variables.items():
            if plant:
                v[:].fix(v[0].value)
            else:
                v[:].set_value(v[0].value)
                v[:].unfix()

        if plant:
            assert degrees_of_freedom(m) == 0

        print(f"degrees of freedom = {degrees_of_freedom(m)}")
        return m


    print("Building plant model...")
    plant = create_model(time_set=[0, nmpc_params['step']], nfe=1, plant=True)
    plant.name = "Plant"
    # Apply scaling and initialize model
    # apply_custom_variable_scaling(plant)
    # apply_custom_constraint_scaling(plant)

    print("Building controller model...")
    controller = create_model(
        time_set=nmpc_params['time_set_controller'],
        nfe=nmpc_params['ntfe'],
        plant=False,
    )
    controller.name = "Controller"
    make_tracking_objective(controller, iter=0)
    
    # Apply scaling and initialize model
    # apply_custom_variable_scaling(controller)
    # apply_custom_constraint_scaling(controller)
    ms.from_json(controller,
                 fname="../../max_production.json.gz",
                 wts=ms.StoreSpec.value())
    # controller.write(filename='controller.nl',
    #                   format=ProblemFormat.nl,
    #                   io_options={'symbolic_solver_labels': True})


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


    def deactivate_state_constraints(m):
        soec = m.fs.soc_module.solid_oxide_cell
        t0 = m.fs.time.first()
        if m.fs.config.has_gas_holdup:
            soec.fuel_channel.mole_frac_comp_eqn[t0,:].deactivate()
            soec.oxygen_channel.mole_frac_comp_eqn[t0,:].deactivate()
        m.fs.sweep_exchanger.temp_wall_center_eqn[t0, :].deactivate()
        m.fs.feed_medium_exchanger.temp_wall_center_eqn[t0, :].deactivate()
        m.fs.feed_hot_exchanger.temp_wall_center_eqn[t0, :].deactivate()
        m.fs.feed_heater.temp_wall_eqn[t0, :].deactivate()
        m.fs.sweep_heater.temp_wall_eqn[t0, :].deactivate()
        return None


    def set_initial_conditions(target_model, source_model):
        tN = source_model.fs.time.last()
        t0 = target_model.fs.time.first()
        target_model_state_vars = get_state_vars(target_model)
        source_model_state_vars = get_state_vars(source_model)

        def set_state_var_ics(state_var_target, state_var_source):
            for (t, *idxs), v in state_var_target.items():
                if t == t0:
                    tN_index = tuple([tN, *idxs])
                    v.set_value(value(state_var_source[tN_index]))
            return None

        for state_var_target, state_var_source in zip(target_model_state_vars,
                                                      source_model_state_vars):
            set_state_var_ics(state_var_target, state_var_source)

        return None


    def apply_control_actions():
        controller_MVs = get_manipulated_variables(controller).keys()
        plant_MVs = get_manipulated_variables(plant).keys()
        # controller_MVs = controller.fs.manipulated_variables
        # plant_MVs = plant.fs.manipulated_variables

        for c, p in zip(controller_MVs, plant_MVs):
            t0 = controller.fs.time.first()
            t1 = controller.fs.time.next(t0)
            for t, v in c.items():
                if t == t1:
                    control_input = value(c[t])
                    p[t].set_value(control_input)
                    p[t].fix()
                    # p[:].set_value(control_input_0)
                    # p[:].fix()
            
        return None


    # Save steady state from the nominal value of oxygen feed flowrate
    def make_states_dict():
        plant_state_vars = get_state_vars(plant)
        states_dict = {c.name: {} for c in plant_state_vars}
        for c in plant_state_vars:
            for (t, *idxs), _ in c.items():
                # Save the last state
                if t == plant.fs.time.last():
                    states_dict[c.name][tuple(idxs)] = []
        return states_dict
    states_dict = make_states_dict()


    def save_states(states_dict):
        plant_state_vars = get_state_vars(plant)
        for c in plant_state_vars:
            for (t, *idxs), v in c.items():
                if t == plant.fs.time.last():
                    states_dict[c.name][tuple(idxs)].append(value(v))
        return None


    controls_dict = {
        alias: [] for c, alias in get_manipulated_variables(controller).items()
    }
    # controls_dict = {c.name: [] for c in controller.fs.manipulated_variables}
    def save_controls(controls_dict):
        t0 = controller.fs.time.first()
        t1 = controller.fs.time.next(t0)  # controller.fs.time.at(2)
        for c, alias in get_manipulated_variables(controller).items():
        # for c in controller.fs.manipulated_variables:
            controls_dict[alias].append(value(c[t1]))
        return None


    controlled_vars_dict = {
        alias: [] for alias in get_controlled_variables(plant).values()
    }
    def save_controlled_vars(controlled_vars_dict):
        # t0 = plant.fs.time.first()
        # tf = plant.fs.time.last()
        for var, alias in get_controlled_variables(plant).items():
            if alias == "cell_average_temperature":
                val = np.mean([value(temp) for temp in var])
                controlled_vars_dict[alias].append(val)
            else:
                controlled_vars_dict[alias].append(value(var))
        return None


    h2_production_rate = []
    def save_h2_production_rate():
        tN = plant.fs.time.last()
        rate = value(plant.fs.h2_mass_production[tN])
        h2_production_rate.append(rate)
        return None


    objective = []
    
    
    def dump_results(filepath):
        # TODO: Add results and plotting for performance variables like power,
        # temperature profiles, efficiency etc.
        """
        filepath is something like './raw_results_dump/yyyy_mm_dd/'
        """
        pickle.dump(controls_dict, open(filepath + 'controls_dict.pkl', 'wb'))
        pickle.dump(states_dict, open(filepath + 'states_dict.pkl', 'wb'))
        pickle.dump(h2_production_rate, open(filepath + 'h2_production_rate.pkl', 'wb'))
        pickle.dump(objective, open(filepath + 'objective.pkl', 'wb'))
        
        plt.figure()
        plt.plot(time_set_nmpc[:len(h2_production_rate)] / 3600, h2_production_rate)
        plt.title('h2_production_rate')
        plt.xlabel('time [hrs]')
        plt.savefig(filepath + 'h2_production_rate.png')
        
        for var, values in controls_dict.items():
            plt.figure()
            plt.plot(time_set_nmpc[:len(h2_production_rate)] / 3600, values)
            plt.title(var)
            plt.xlabel('time [hrs]')
            plt.savefig(filepath + var + '.png')
        
        return None
    
    
    def write_nl(m, file_prefix='tracking'):
        m.write(
            filename=file_prefix + '.nl',
            format=ProblemFormat.nl,
            io_options={'symbolic_solver_labels': True}
        )
        return None
    

    def get_extra_dofs(m):
        # write nl file
        write_nl(m, file_prefix='tracking')
        
        # get variables from pyomo
        with open('tracking.col', 'r') as model_variables:
            varlist = model_variables.readlines()
            print(len(varlist))
            varlist = [var.strip('\n') for var in varlist]
        
        # get variables from incidence graph
        from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
        igraph = IncidenceGraphInterface(m)
        varlist_igraph = [var.name for var in igraph.variables]
        
        extra_dofs = [
            varname for varname in varlist if varname not in varlist_igraph
        ]
            
        return extra_dofs
    
    
    def make_plots(savefig=False):
        tracking_targets = get_tracking_targets(plant)
        
        def demarcate_ramps(ax):
            ramp_list = nmpc_params['time_set_PI'][1:-1]
            for tpoint in np.squeeze(ramp_list):
                ax.plot(
                    np.array([tpoint, tpoint]) / 60**2,
                    [-1e+08, 1e+08],
                    color="gray",
                    linestyle='--',
                )
        
        def make_subplots(var_dict, aliases):
            nplots = len(aliases)
            if nplots == 1:
                subplots = (1, 1)
            elif nplots == 2:
                subplots = (2, 1)
            elif nplots == 3:
                subplots = (3, 1)

            trajectories = [var_dict[alias] for alias in aliases]
            fig, axs = plt.subplots(*subplots)
            axs = np.array(axs).reshape(-1)
            for ax, alias, values in zip(axs, aliases, trajectories):
                ax.plot(
                    time_set_nmpc[:len(h2_production_rate)] / 3600,
                    values,
                    color='blue',
                    linewidth=2,
                )
                
                try:
                    target = np.array(list(tracking_targets[alias].values()))
                    ax.plot(
                        nmpc_params['time_set_output'] / 3600,
                        target,
                        'r--',
                    )
                except:
                    pass
                
                demarcate_ramps(ax)
                try:
                    ymin = min(list(target) + list(values))
                    ymax = max(list(target) + list(values))
                except:
                    ymin = min(values)
                    ymax = max(values)
                ylim = [
                    ymin - 0.05 * (ymax - ymin),
                    ymax + 0.05 * (ymax - ymin),
                ]
                ax.set_ylim(ylim)
                ax.set_title(alias)
                ax.set_xlabel('time [hrs]')
            fig.tight_layout()
            
            return None
        
        # hydrogen production rate
        h2_target = np.array(list(get_h2_production_target(plant).values()))
        fig, ax = plt.subplots()
        ax.plot(
            time_set_nmpc[:len(h2_production_rate)] / 3600,
            h2_production_rate,
            color="blue",
            linewidth=2,
        )
        ax.plot(nmpc_params['time_set_output'] / 3600, h2_target, 'r--')
        demarcate_ramps(ax)
        ax.set_ylim([-1.1, 2.1])
        ax.set_title('h2_production_rate')
        ax.set_xlabel('time [hrs]')
        fig.tight_layout()
        
        # controls
        var_dict = controls_dict
        
        # potential
        aliases = ["potential"]
        make_subplots(var_dict, aliases)
        
        # feed rates
        aliases = ["makeup_feed_rate", "sweep_feed_rate"]
        make_subplots(var_dict, aliases)
        
        # condenser
        aliases = ["vgr_recycle_ratio", "condenser_hot_outlet_temperature"]
        make_subplots(var_dict, aliases)
        
        # trim heater duties
        aliases = ["feed_heater_duty", "sweep_heater_duty"]
        make_subplots(var_dict, aliases)

        # recycle ratios
        aliases = ["fuel_recycle_ratio", "sweep_recycle_ratio"]
        make_subplots(var_dict, aliases)

        # makeup mole fractions
        aliases = ["makeup_mole_frac_comp_H2", "makeup_mole_frac_comp_H2O"]
        make_subplots(var_dict, aliases)

        # controlled variables and other output variables
        var_dict = controlled_vars_dict
        
        # trim heater tempeatures
        aliases = ["feed_heater_outlet_temperature", "sweep_heater_outlet_temperature"]
        make_subplots(var_dict, aliases)

        aliases = ["fuel_outlet_temperature",
                   "sweep_outlet_temperature",
                   "stack_core_temperature"]
        make_subplots(var_dict, aliases)
 
        aliases = ["oxygen_out", "hydrogen_in"]
        make_subplots(var_dict, aliases)
               
        aliases = ["fuel_inlet_temperature",
                   "sweep_inlet_temperature",
                   "cell_average_temperature"]
        make_subplots(var_dict, aliases)

        return None


    iscale.scale_time_discretization_equations(
        controller, controller.fs.time, 1 / nmpc_params['step'])
    iscale.scale_time_discretization_equations(
        plant, plant.fs.time, 1 / nmpc_params['step'])

    # Solve initial plant model
    print('Solving initial plant model...\n')
    res = initialize_model_with_petsc(plant)
    solver.solve(plant, tee=True)
    save_states(states_dict)
    # raise Exception()

    idaes.cfg.ipopt.options.tol = 1e-05  # default = 1e-08
    idaes.cfg.ipopt.options.acceptable_tol = 1e-04  # default = 1e-06
    idaes.cfg.ipopt.options.constr_viol_tol = 1e-04  # default = 1e-04
    idaes.cfg.ipopt.options.dual_inf_tol = 1e+01  # default = 1e+00 (unscaled)
    idaes.cfg.ipopt.options.compl_inf_tol = 1e+00  # default = 1e+00 (unscaled)
    # idaes.cfg.ipopt.options.acceptable_obj_change_tol = 1e-01  # default = 1e+20
    idaes.cfg.ipopt.options.bound_relax_factor = 1e-06 # default = 1e-08
    idaes.cfg.ipopt.options.max_iter = 250
    idaes.cfg.ipopt.options.mu_init = 1e-05
    # idaes.cfg.ipopt_l1.options.bound_relax_factor = 1e-06  # default = 1e-08
    # idaes.cfg.ipopt_l1.options.tol = 1e-04  # default = 1e-08
    # idaes.cfg.ipopt_l1.options.constr_viol_tol = 1e-04  # default = 1e-04
    # idaes.cfg.ipopt_l1.options.acceptable_tol = 1e-06  # default = 1e-06
    # idaes.cfg.ipopt_l1.options.dual_inf_tol = 1e+00  # default = 1e+00 (unscaled)
    solver = pyo.SolverFactory('ipopt')

    # Set up Suffixes for duals
    # with open("ipopt.opt", "w") as f:
    #     f.write("warm_start_init_point yes \n"
    #             "warm_start_mult_bound_push 1e-06 \n"
    #             "warm_start_bound_push 1e-06 \n")

    controller.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    controller.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    controller.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    controller.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    controller.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
    

    # import pdb; pdb.set_trace()
    # =============================================================================
    # NMPC
    # =============================================================================
    time_set_nmpc = np.linspace(
        0.0,
        nmpc_params['sim_horizon'],
        num=nmpc_params['nsteps'] - nmpc_params['nsteps_horizon'] + 1,
    )
    
    for iter, t in enumerate(time_set_nmpc):
        print(f"""
        ===============================================
        Iteration {iter}, Time horizon = {t} to {t+nmpc_params['horizon']}
        ===============================================""")

        # Set up the control problem
        # if iter > 0:
        #     shift_dae_vars_by_dt(
        #         controller, controller.fs.time, nmpc_params['step']
        #     )
        
        # Fix initial conditions for states
        set_initial_conditions(target_model=controller, source_model=plant)
        controller.fs.fix_initial_conditions()

        print(f'\nSolving {controller.name}...\n')
        make_tracking_objective(controller, iter)
        # add_penalty_formulation(controller)
        # controller.solutions.load_from(results)
        
        results = solver.solve(controller, tee=True, load_solutions=True)
        termination_condition = results.solver.termination_condition
        if (termination_condition == TerminationCondition.infeasible or
            termination_condition == TerminationCondition.maxIterations):
            # cds = get_large_duals(controller, tol=1e+03)
            break
            
        objective.append(value(controller.obj))
        save_controls(controls_dict)

        # update parameters between runs
        # controller.ipopt_zL_in.update(controller.ipopt_zL_out)
        # controller.ipopt_zU_in.update(controller.ipopt_zU_out)

        # Set up Plant model
        set_initial_conditions(target_model=plant, source_model=plant)
        plant.fs.fix_initial_conditions()
        apply_control_actions()
        assert degrees_of_freedom(plant) == 0

        print(f'\nSolving {plant.name}...\n')
        solver.solve(plant, tee=True, load_solutions=True)
        # res = initialize_model_with_petsc(plant)

        save_states(states_dict)
        save_h2_production_rate()
        save_controlled_vars(controlled_vars_dict)

        # break











