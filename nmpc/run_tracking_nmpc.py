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
from plotting import make_plots
from nmpc_helper import (
    make_tracking_objective_with_MVs,
    apply_custom_variable_scaling,
    apply_custom_constraint_scaling,
    initialize_model_with_petsc,
    get_time_coordinates,
    get_tracking_variables,
    get_manipulated_variables,
    check_scaling,
    hunt_degeneracy,
    shift_dae_vars_by_dt,
    check_incidence_analysis,
    get_large_duals,
    get_large_residuals,
    add_penalty_formulation,
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
        for var in manipulated_variables:
            var[:].fix()


    def unfix_manipulated_variables(m):
        manipulated_variables = get_manipulated_variables(m)
        for var in manipulated_variables:
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
        
        # rectify_initialization(m)        
        # m.fs.deactivate_shortcut()
        iscale.calculate_scaling_factors(m)

        pyo.TransformationFactory("dae.finite_difference").apply_to(
            m.fs, nfe=nfe, wrt=m.fs.time, scheme="BACKWARD"
        )

        # Initialize model
        ms.from_json(m,
                     fname="../../max_production.json.gz",
                     wts=ms.StoreSpec.value())
        
        # import pdb; pdb.set_trace()

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
        m.fs.condenser_hx.hot_side.properties[:, 1].temperature.fix()
        
        manipulated_variables = get_manipulated_variables(m)        
        # import pdb; pdb.set_trace()
        for v in manipulated_variables:
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
    apply_custom_variable_scaling(plant)
    apply_custom_constraint_scaling(plant)

    print("Building controller model...")
    controller = create_model(
        time_set=nmpc_params['time_set_controller'],
        nfe=nmpc_params['ntfe'],
        plant=False,
    )
    controller.name = "Controller"
    make_tracking_objective_with_MVs(controller, iter=0)
    
    # Apply scaling and initialize model
    apply_custom_variable_scaling(controller)
    apply_custom_constraint_scaling(controller)
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
        controller_MVs = get_manipulated_variables(controller)
        plant_MVs = get_manipulated_variables(plant)
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


    controls_dict = {c.name: [] for c in get_manipulated_variables(controller)}
    # controls_dict = {c.name: [] for c in controller.fs.manipulated_variables}
    def save_controls(controls_dict):
        t0 = controller.fs.time.first()
        t1 = controller.fs.time.next(t0)  # controller.fs.time.at(2)
        for c in get_manipulated_variables(controller):
        # for c in controller.fs.manipulated_variables:
            controls_dict[c.name].append(value(c[t1]))
        return None


    # controlled_vars_dict = {
    #     c.name: [] for c in [controller.fs.feed_heater.outlet.temperature,
    #                           controller.fs.sweep_heater.outlet.temperature]
    # }
    # def save_controlled_vars(controlled_vars_dict):
    #     t0 = controller.fs.time.first()
    #     for c in [controller.fs.feed_heater.outlet.temperature,
    #               controller.fs.sweep_heater.outlet.temperature]:
    #         controlled_vars_dict[c.name].append(value(c[t0]))
    #     return None


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

    idaes.cfg.ipopt.options.tol = 1e-04  # default = 1e-08
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
        make_tracking_objective_with_MVs(controller, iter)
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
        # save_controlled_vars(controlled_vars_dict)

        # update parameters between runs
        # controller.ipopt_zL_in.update(controller.ipopt_zL_out)
        # controller.ipopt_zU_in.update(controller.ipopt_zU_out)

        # Set up Plant model
        set_initial_conditions(target_model=plant, source_model=plant)
        plant.fs.fix_initial_conditions()
        apply_control_actions()
        assert degrees_of_freedom(plant) == 0

        print(f'\nSolving {plant.name}...\n')
        # solver.solve(plant, tee=True, load_solutions=True)
        res = initialize_model_with_petsc(plant)

        save_states(states_dict)
        save_h2_production_rate()

        # break











