import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.io import loadmat
import cloudpickle as pickle
from nmpc_helper import alias_dict, get_time_coordinates


def get_nmpc_results(filepath):
    controls_dict = pickle.load(open(filepath + 'controls_dict.pkl', 'rb'))
    CVs_dict = pickle.load(open(filepath + 'cvs_dict.pkl', 'rb'))
    states_dict = pickle.load(open(filepath + 'states_dict.pkl', 'rb'))
    h2_production_rate = pickle.load(
        open(filepath + 'h2_production_rate.pkl', 'rb')
    )
    dTdz_electrode_logbook = pickle.load(
        open(filepath + 'temperature_gradients.pkl', 'rb')
    )
    return [controls_dict,
            CVs_dict,
            states_dict,
            h2_production_rate,
            dTdz_electrode_logbook]

def _demarcate_ramps(ax, results_dict):
    for tpoint in np.squeeze(results_dict["ramp_list"])[:-1]:
        ax.plot(
            np.array([tpoint, tpoint]) / 60 ** 2,
            [-1e6, 1e6],
            color="darkgray",
            linestyle="--",
        )

def plot_results(filename, nmpc_filepath, include_PI):
    results_dict = loadmat(filename)
    controls_dict, CVs_dict, states_dict, h2_production_rate, \
        dTdz_electrode_logbook = get_nmpc_results(nmpc_filepath)

    for key, value in results_dict.items():
        # Turn n by 1 arrays in into vectors
        results_dict[key] = np.squeeze(value)

    demarcate_ramps = lambda ax: _demarcate_ramps(ax, results_dict)
    time = results_dict["time"] / 60 ** 2
    nmpc_params = get_time_coordinates()
    time_nmpc = np.array(
        [i * nmpc_params["step"] / 3600 for i in range(nmpc_params["nsteps"])]
    )

    ax_fontsize = 14
    title_fontsize = 16
    iz_plot = [1, 3, 5, 8, 10]

    fig = plt.figure()
    ax = fig.subplots()

    ax.plot(time, results_dict["potential"], label="PI")
    if include_PI:
        ax.plot(
            time,
            results_dict["voltage_controller_mv_ref"],
            color="darkblue",
            linestyle="dotted",
        )

    key, = [k for k, v in alias_dict.items() if v == "potential"]
    potential = controls_dict[key]
    ax.plot(
        time_nmpc[:len(potential)],
        potential,
        color="red",
        label="NMPC"
    )
    
    demarcate_ramps(ax)
    
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0.65, 1.45))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Cell potential (V)", fontsize=ax_fontsize)
    ax.set_title("SOEC Voltage", fontsize=title_fontsize)
    ax.legend(loc="best")

    fig = plt.figure()
    ax = fig.subplots()

    ax.plot(time, results_dict["soec_fuel_inlet_flow"], color="blue", label="Fuel")
    ax.plot(time, results_dict["soec_oxygen_inlet_flow"], color="darkblue", label="Sweep")
    
    # key_H2, = [k for k, v in alias_dict.items() if v == "hydrogen_in"]
    # key_O2, = [k for k, v in alias_dict.items() if v == "oxygen_out"]
    # H2_in = CVs_dict[key_H2]
    # O2_in = CVs_
    # ax.plot(
    #     time_nmpc[]
    # )
    
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0, 15000))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("SOEC inlet molar flow (mol/s)", fontsize=ax_fontsize)
    ax.set_title("Inlet molar flow rates", fontsize=title_fontsize)
    ax.legend()

    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(time, 1e-6 * results_dict["fuel_heater_duty"], label="Fuel PI", color="navy")
    ax.plot(time, 1e-6 * results_dict["sweep_heater_duty"], label="Sweep PI", color="deepskyblue")
    if include_PI:
        ax.plot(
            time,
            1e-6 * results_dict["feed_heater_inner_controller_mv_ref"],
            label="Fuel reference",
            color="navy",
            linestyle="dotted"
        )
        ax.plot(
            time,
            1e-6 * results_dict["sweep_heater_inner_controller_mv_ref"],
            label="Sweep reference",
            color="blue",
            linestyle="dotted"
        )
    
    key_fuel, = [k for k, v in alias_dict.items() if v == "feed_heater_duty"]
    key_sweep, = [k for k, v in alias_dict.items() if v == "sweep_heater_duty"]
    feed_heater_duty = controls_dict[key_fuel]
    sweep_heater_duty = controls_dict[key_sweep]
    ax.plot(
        time_nmpc[:len(feed_heater_duty)],
        1e-06 * np.array(feed_heater_duty),
        color="firebrick",
        label="Fuel NMPC",
    )
    ax.plot(
        time_nmpc[:len(sweep_heater_duty)],
        1e-06 * np.array(sweep_heater_duty),
        color="darkorange",
        label="Sweep NMPC",
    )
    
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0, 13))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Heater duty (MW)", fontsize=ax_fontsize)
    ax.set_title("Trim heater duties", fontsize=title_fontsize)
    ax.legend()


    fig = plt.figure()
    ax = fig.subplots()
    if include_PI:
        ax.plot(
            time,
            1e-6 * results_dict["feed_heater_inner_controller_mv_ref"],
            label="Fuel reference",
            color="navy",
            linestyle="dotted"
        )
        ax.plot(
            time,
            1e-6 * results_dict["sweep_heater_inner_controller_mv_ref"],
            label="Sweep reference",
            color="blue",
            linestyle="dotted"
        )
    
    key_fuel, = [k for k, v in alias_dict.items() if v == "feed_heater_duty"]
    key_sweep, = [k for k, v in alias_dict.items() if v == "sweep_heater_duty"]
    feed_heater_duty = controls_dict[key_fuel]
    sweep_heater_duty = controls_dict[key_sweep]
    ax.plot(
        time_nmpc[:len(feed_heater_duty)],
        1e-06 * np.array(feed_heater_duty),
        color="firebrick",
        label="Fuel NMPC",
    )
    ax.plot(
        time_nmpc[:len(sweep_heater_duty)],
        1e-06 * np.array(sweep_heater_duty),
        color="darkorange",
        label="Sweep NMPC",
    )
    
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0, 2))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Heater duty (MW)", fontsize=ax_fontsize)
    ax.set_title("Trim heater duties", fontsize=title_fontsize)
    ax.legend()

    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(time, results_dict["fuel_inlet_H2O"], label="Inlet $H_2O$")
    ax.plot(time, results_dict["fuel_outlet_H2O"], label="Outlet $H_2O$")
    ax.plot(time, results_dict["sweep_inlet_O2"], label="Inlet $O_2$")
    ax.plot(time, results_dict["sweep_outlet_O2"], label="Outlet $O_2$")
    ax.plot(time, results_dict["product_mole_frac_H2"], label="Product $H_2$")
    ax.plot(time, 0.35 * np.ones(time.shape), '--')
    ax.plot(time, 0.25 * np.ones(time.shape), '--')
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0, 1))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Mole fraction", fontsize=ax_fontsize)
    ax.set_title("Reactor feed and effluent concentrations", fontsize=title_fontsize)
    ax.legend()

    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(time, results_dict["H2_production"], label="PI")
    ax.plot(time, 0.4 * np.ones(time.shape), 'r--')
    ax.plot(time, 2 * np.ones(time.shape), 'r--')
    
    ax.plot(
        time_nmpc[:len(h2_production_rate)],
        np.array(h2_production_rate),
        color="red",
        label="NMPC",
    )
    
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((-1.25, 2.5))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Hydrogen Production Rate (kg/s)", fontsize=ax_fontsize)
    ax.set_title("Instantaneous $H_2$ production rate", fontsize=title_fontsize)

    # if include_PI:
    #     ax.plot(
    #         time,
    #         results_dict["h2_production_rate_controller_setpoint"],
    #         label="Target",
    #         color="darkblue",
    #         linestyle="dotted"
    #     )
    ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()
#     ax.plot(time, results_dict["steam_feed_rate"])
#     # if include_PI:
#     #     ax.plot(time,
#     #             results_dict["h2_production_rate_controller_mv_ref"],
#     #             label="Target",
#     #             color="darkblue",
#     #             linestyle="dotted"
#     #     )
#     demarcate_ramps(ax)
#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((0, 7500))
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Steam feed rate (mol/s)", fontsize=ax_fontsize)
#     ax.set_title("Steam feed rate", fontsize=title_fontsize)
#     ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()
#     # ax2 = ax.twinx()
#     ax.plot(time, 1e-6 * results_dict["total_electric_power"], 'b', label="Total power")
#     # ax2.plot(time, results_dict["efficiency_lhv"], 'r', label="Efficiency (LHV)")
#     demarcate_ramps(ax)
#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((-125, 350))
#     # ax2.set_ylim((0, 1.4))
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Power usage (MW)", color="blue", fontsize=ax_fontsize)
#     # ax2.set_ylabel("Energy per H2 mass (MJ/kg)", color="red",
#     #                fontsize=ax_fontsize)
#     # ax2.set_ylabel("Efficiency (LHV)", color="red",
#     #                fontsize=ax_fontsize)
#     ax.set_title("Power usage and efficiency", fontsize=title_fontsize)
#     # ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()

#     ax.plot(time, results_dict["fuel_inlet_temperature"], label="Fuel", color="tab:blue")
#     ax.plot(time, results_dict["sweep_inlet_temperature"], label="Sweep", color="tab:orange")
#     ax.plot(time, results_dict["cell_average_temperature"], label="Cell average", color="darkgreen")

#     if include_PI:
#         ax.plot(
#             time,
#             results_dict["feed_heater_inner_controller_setpoint"],
#             label="Fuel target",
#             color="darkblue",
#             linestyle="dotted"
#         )
#         ax.plot(
#             time,
#             results_dict["sweep_heater_inner_controller_setpoint"],
#             label="Sweep target",
#             color="saddlebrown",
#             linestyle="dotted"
#         )

#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((850, 1150))
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
#     demarcate_ramps(ax)
#     ax.set_title("SOEC temperature", fontsize=title_fontsize)
#     ax.legend()

    fig = plt.figure()
    ax = fig.subplots()

    ax.plot(time, results_dict["fuel_outlet_temperature"], label="Fuel PI", color="navy")
    ax.plot(time, results_dict["sweep_outlet_temperature"], label="Sweep PI", color="deepskyblue")
    if include_PI:
        ax.plot(
            time,
            results_dict["feed_heater_outer_controller_setpoint"],
            label="Fuel target",
            color="navy",
            linestyle="dotted"
        )
        ax.plot(
            time,
            results_dict["sweep_heater_outer_controller_setpoint"],
            label="Sweep target",
            color="blue",
            linestyle="dotted"
        )

    key_fuel, = [k for k, v in alias_dict.items() if v == "fuel_outlet_temperature"]
    key_sweep, = [k for k, v in alias_dict.items() if v == "sweep_outlet_temperature"]
    fuel_outlet_temperature = CVs_dict[key_fuel]
    sweep_outlet_temperature = CVs_dict[key_sweep]
    ax.plot(
        time_nmpc[:len(fuel_outlet_temperature)],
        np.array(fuel_outlet_temperature),
        color="firebrick",
        label="Fuel NMPC",
    )
    ax.plot(
        time_nmpc[:len(sweep_outlet_temperature)],
        np.array(sweep_outlet_temperature),
        color="darkorange",
        label="Sweep NMPC",
    )

    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((890, 1050))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
    demarcate_ramps(ax)
    ax.set_title("SOEC outlet temperature", fontsize=title_fontsize)
    ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()

#     for iz in iz_plot:
#         ax.plot(time, results_dict["temperature_z"][iz-1, :], label=f"z node {iz}")

#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((890, 1050))
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
#     demarcate_ramps(ax)
#     ax.set_title("SOEC temperature profile", fontsize=title_fontsize)
#     ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()

#     for iz in iz_plot:
#         ax.plot(
#             time,
#             results_dict["temperature_z"][iz-1, :] + results_dict["fuel_electrode_temperature_deviation_x"][iz-1, :],
#             label=f"z node {iz}"
#         )

#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((890, 1050))
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
#     demarcate_ramps(ax)
#     ax.set_title("Temperature electrode", fontsize=title_fontsize)
#     ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()

#     for iz in iz_plot:
#         ax.plot(
#             time,
#             results_dict["temperature_z"][iz-1, :] + results_dict["interconnect_temperature_deviation_x"][iz-1, :],
#             label=f"z node {iz}"
#         )

#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((890, 1050))
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
#     demarcate_ramps(ax)
#     ax.set_title("Temperature interconnect", fontsize=title_fontsize)
#     ax.legend()

#     # fig = plt.figure()
#     # ax = fig.subplots()
#     #
#     # for iz in iz_plot:
#     #     ax.plot(
#     #         time,
#     #         results_dict["temperature_z_gradient"][iz-1, :],
#     #         label=f"node {iz}"
#     #     )
#     #
#     # ax.set_xlim(time[0], time[-1])
#     # ax.set_ylim((-550, 550))
#     # # ax.set_ylim((-75,75))
#     # ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     # ax.set_ylabel("$dT/dz$ ($K/m$)", fontsize=ax_fontsize)
#     # ax.set_title("SOEC temperature z gradient", fontsize=title_fontsize)
#     # demarcate_ramps(ax)
#     # ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()

#     for iz in iz_plot:
#         ax.plot(
#             time,
#             results_dict["fuel_electrode_gradient"][iz-1, :],
#             label=f"node {iz}"
#         )

#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((-1000, 1000))
#     # ax.set_ylim((-75,75))
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("$dT/dz$ ($K/m$)", fontsize=ax_fontsize)
#     ax.set_title("SOEC PEN temperature gradient", fontsize=title_fontsize)
#     demarcate_ramps(ax)
#     ax.legend()

#     # fig = plt.figure()
#     # ax = fig.subplots()
#     #
#     # for iz in iz_plot:
#     #     ax.plot(
#     #         time,
#     #         results_dict["interconnect_gradient"][iz-1, :],
#     #         label=f"node {iz}"
#     #     )
#     #
#     # ax.set_xlim(time[0], time[-1])
#     # ax.set_ylim((-550, 550))
#     # # ax.set_ylim((-75,75))
#     # ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     # ax.set_ylabel("$dT/dz$ ($K/m$)", fontsize=ax_fontsize)
#     # ax.set_title("SOEC interconnect temperature gradient", fontsize=title_fontsize)
#     # demarcate_ramps(ax)
#     # ax.legend()

#     # fig = plt.figure()
#     # ax = fig.subplots()
#     #
#     # for iz in [2, 4, 6, 9]:
#     #     ax.plot(
#     #         time,
#     #         (
#     #                 np.array(traj.vecs[str(temp_z[tf, iz + 1])])
#     #                 - 2 * np.array(traj.vecs[str(temp_z[tf, iz])])
#     #                 + np.array(traj.vecs[str(temp_z[tf, iz - 1])])
#     #         ) / pyo.value(m.fs.soc_module.solid_oxide_cell.length_z / 10) ** 2,
#     #         label=f"node {iz}"
#     #     )
#     #
#     # ax.set_xlim(time[0], time[-1])
#     # ax.set_ylim((-6500, 14500))
#     # # ax.set_ylim((-75,75))
#     # ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     # ax.set_ylabel("$d^2T/dz^2$ ($K/m^2$)", fontsize=ax_fontsize)
#     # ax.set_title("SOEC temperature curvature", fontsize=title_fontsize)
#     # demarcate_ramps(ax)
#     # ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()

#     for iz in iz_plot:
#         ax.plot(time, results_dict["current_density"][iz-1, :] / 10, label=f"z node {iz}")

#     ax.set_xlim(time[0], time[-1])
#     # ax.set_ylim((575,875))
#     ax.set_ylim((-1000, 500))
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Current density ($mA/cm^2$)", fontsize=ax_fontsize)
#     ax.set_title("SOEC current density", fontsize=title_fontsize)
#     demarcate_ramps(ax)
#     ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
#     for z in range(results_dict["feed_heater_temperature"].shape[0]):
#         ax.plot(time, results_dict["feed_heater_temperature"][z, :], label=f"Feed wall node {z+1}")
#         ax.plot(time, results_dict["sweep_heater_temperature"][z, :], label=f"Sweep wall node {z+1}")
#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((870, 1175))
#     demarcate_ramps(ax)
#     ax.set_title("Trim heater wall temperature", fontsize=title_fontsize)
#     ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
#     for z in range(results_dict["feed_medium_exchanger_temperature"].shape[0]):
#         ax.plot(time,
#                 results_dict["feed_medium_exchanger_temperature"][z, :],
#                 label=f"Node {z + 1}")
#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((370, 520))
#     demarcate_ramps(ax)
#     ax.set_title("Medium exchanger wall temperature", fontsize=title_fontsize)
#     ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)

#     for z in range(results_dict["feed_hot_exchanger_temperature"].shape[0]):
#         ax.plot(time,
#                 results_dict["feed_hot_exchanger_temperature"][z, :],
#                 label=f"Node {z + 1}")
#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((420, 1070))
#     demarcate_ramps(ax)
#     ax.set_title("Hot exchanger wall temperature", fontsize=title_fontsize)
#     ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
#     for z in range(results_dict["sweep_exchanger_temperature"].shape[0]):
#         ax.plot(time,
#                 results_dict["sweep_exchanger_temperature"][z, :],
#                 label=f"Node {z}")
#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((350, 1020))
#     demarcate_ramps(ax)
#     ax.set_title("Sweep exchanger wall temperature", fontsize=title_fontsize)
#     ax.legend()

#     fig = plt.figure()
#     ax = fig.subplots()
#     ax2 = ax.twinx()

#     ax.plot(time, results_dict["condenser_outlet_temperature"], label="Temperature", color="tab:blue")
#     ax2.plot(time, results_dict["product_mole_frac_H2"], label="H2 mole fraction", color="tab:orange")

#     # if include_PI:
#     #     ax.plot(
#     #         time,
#     #         results_dict["feed_heater_inner_controller_setpoint"],
#     #         label="Fuel target",
#     #         color="darkblue",
#     #         linestyle="dotted"
#     #     )
#     #     ax.plot(
#     #         time,
#     #         results_dict["sweep_heater_inner_controller_setpoint"],
#     #         label="Sweep target",
#     #         color="saddlebrown",
#     #         linestyle="dotted"
#     #     )

#     ax.set_xlim(time[0], time[-1])
#     ax.set_ylim((273.15, 373.15))
#     ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
#     ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize, color="tab:blue")
#     ax2.set_ylim((0,1))
#     ax2.set_ylabel("Mole fraction $H_2$", fontsize=ax_fontsize,  color="tab:orange")

#     demarcate_ramps(ax)
#     ax.set_title("Condenser Vapor Outlet", fontsize=title_fontsize)
#     # ax.legend()

#     plt.show()


def make_paper_figures_old(filename, include_PI):
    results_dict = loadmat(filename)
    for key, value in results_dict.items():
        # Turn n by 1 arrays in into vectors
        results_dict[key] = np.squeeze(value)

    demarcate_ramps = lambda ax: _demarcate_ramps(ax, results_dict)

    time = results_dict["time"] / 60 ** 2

    fig = plt.figure(figsize=(7.5, 9.5))
    axes = fig.subplots(nrows=3, ncols=2, squeeze=False)
    ax = axes[2, 0]

    ax_fontsize = 10
    title_fontsize = 12
    legend_fontsize = 8
    tick_label_fontsize = 7

    ax.plot(time, results_dict["fuel_heater_duty"] * 1e-6, label="Fuel", color="steelblue")
    ax.plot(time, results_dict["sweep_heater_duty"] * 1e-6, label="Sweep", color="orangered")
    if include_PI:
        ax.plot(
            time,
            results_dict["feed_heater_inner_controller_mv_ref"] * 1e-6,
            label="Fuel reference",
            color="darkblue",
            linestyle="dotted"
        )
        ax.plot(
            time,
            results_dict["sweep_heater_inner_controller_mv_ref"] * 1e-6,
            label="Sweep reference",
            color="saddlebrown",
            linestyle="dotted"
        )
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0, 13))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Heater duty (MW)", fontsize=ax_fontsize)
    ax.set_title("Trim heater duties", fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
    ax.legend(fontsize=legend_fontsize)

    ax = axes[2, 1]
    ax2 = ax.twinx()
    ax.plot(
        time,
        1 - results_dict["fuel_inlet_H2O"],
        label="Inlet $H_2$",
        color="steelblue"
    )
    ax.plot(
        time,
        0.05 * np.ones(time.shape),
        color="darkblue",
        linestyle="dotted"
    )
    # ax.plot(time,
    #         traj.vecs[str(soec.oxygen_inlet.mole_frac_comp[tf, "O2"])],
    #         label="Inlet $O_2$")
    ax.plot(
        time,
        results_dict["sweep_outlet_O2"],
        label="Outlet $O_2$",
        color="orangered"
    )
    ax.plot(
        time,
        0.35 * np.ones(time.shape),
        color="saddlebrown",
        linestyle="dotted"
    )

    ax2.plot(
        time,
        100 * (1 - results_dict["fuel_outlet_H2O"]),
        color="forestgreen"
        # label="Overall water conversion"
    )
    ax2.plot(
        time,
        75 * np.ones(time.shape),
        color="darkgreen",
        linestyle="dotted"
    )
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0, 1))
    ax2.set_ylim((0, 100))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Mole fraction", fontsize=ax_fontsize)
    ax2.set_ylabel("Overall water conversion", fontsize=ax_fontsize, color="forestgreen")
    ax.set_title("Reactor feed and effluent concentrations", fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.tick_params(axis="y", which="major", labelsize=tick_label_fontsize)
    ax2.tick_params(axis="y", which="minor", labelsize=tick_label_fontsize)
    ax.legend(fontsize=legend_fontsize)

    ax = axes[1, 1]
    ax.plot(
        time,
        results_dict["H2_production"],
        color="steelblue"
    )
    if include_PI:
        ax.plot(
            time,
            results_dict["h2_production_rate_controller_setpoint"],
            label="Target",
            color="darkblue",
            linestyle="dotted"
        )
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((-0.8, 2.5))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Hydrogen Production Rate (kg/s)", fontsize=ax_fontsize)
    ax.set_title("Instantaneous $H_2$ production rate", fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
    ax.legend(fontsize=legend_fontsize)

    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.plot(time, results_dict["total_electric_power"] * 1e-6, color='steelblue', label="Total power")
    ax2.plot(time, results_dict["efficiency_lhv"], color='orangered', label="Efficiency (LHV)")
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0, 350))
    ax2.set_ylim((0, 1.4))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Power usage (MW)", color="steelblue", fontsize=ax_fontsize)
    # ax2.set_ylabel("Energy per H2 mass (MJ/kg)", color="red",
    #                fontsize=ax_fontsize)
    ax2.set_ylabel("Efficiency (LHV)", color="orangered",
                    fontsize=ax_fontsize)
    ax.set_title("Power usage and efficiency", fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)

    ax = axes[0, 0]

    ax.plot(time, results_dict["fuel_inlet_temperature"], label="Fuel", color="steelblue")
    ax.plot(time, results_dict["sweep_inlet_temperature"], label="Sweep", color="orangered")
    ax.plot(time, results_dict["cell_average_temperature"], label="Cell average", color="forestgreen")
    if include_PI:
        ax.plot(
            time,
            results_dict["feed_heater_inner_controller_setpoint"],
            label="Fuel target",
            color="darkblue",
            linestyle="dotted"
        )
        ax.plot(
            time,
            results_dict["sweep_heater_inner_controller_setpoint"],
            label="Sweep target",
            color="saddlebrown",
            linestyle="dotted"
        )

    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((890, 1140))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
    demarcate_ramps(ax)
    ax.set_title("SOEC temperature", fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
    ax.legend(fontsize=legend_fontsize)

    ax = axes[1, 0]

    ax.plot(time, results_dict["fuel_outlet_temperature"], label="Fuel", color="steelblue")
    ax.plot(time, results_dict["sweep_outlet_temperature"], label="Sweep", color="orangered")
    if include_PI:
        ax.plot(
            time,
            results_dict["feed_heater_outer_controller_setpoint"],
            label="Fuel target",
            color="darkblue",
            linestyle="dotted"
        )
        ax.plot(
            time,
            results_dict["sweep_heater_outer_controller_setpoint"],
            label="Sweep target",
            color="saddlebrown",
            linestyle="dotted"
        )

    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((890, 1120))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
    demarcate_ramps(ax)
    ax.set_title("SOEC outlet temperature", fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
    ax.legend(fontsize=legend_fontsize)

    fig.tight_layout()
    fig.savefig("plot_grid.png", dpi=360)


def make_paper_figures(filename_PI, filename_NMPC, third_row=False):
    results_PI = loadmat(filename_PI)
    results_NMPC = loadmat(filename_NMPC)

    demarcate_ramps = lambda ax: _demarcate_ramps(ax, results_PI)

    if third_row:
        fig = plt.figure(figsize=(7.5, 9.5))
        axes = fig.subplots(nrows=3, ncols=2, squeeze=False)
    else:
        fig = plt.figure(figsize=(7.5, 6.5))
        axes = fig.subplots(nrows=2, ncols=2, squeeze=False)


    ax_fontsize = 10
    title_fontsize = 12
    legend_fontsize = 8
    tick_label_fontsize = 7
    iz_plot = [1, 10]
    iz_colors = ["steelblue", "orangered"]

    for results_dict, run, linestyle in zip((results_PI, results_NMPC), ("PI", "NMPC"), ("dashdot", "solid")):
        for key, value in results_dict.items():
            # Turn n by 1 arrays in into vectors
            results_dict[key] = np.squeeze(value)
        time = results_dict["time"] / 60 ** 2
        if third_row:
            ax = axes[2, 0]
        else:
            ax = axes[1, 1]
        for iz, color in zip(iz_plot, iz_colors):
            ax.plot(
                time,
                results_dict["fuel_electrode_gradient"][iz - 1, :],
                label=f"{run} Node {iz}",
                color=color,
                linestyle=linestyle,
            )

        ax.set_xlim(time[0], time[-1])
        ax.set_ylim((-550, 550))
        # ax.set_ylim((-75,75))
        ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
        ax.set_ylabel("$dT/dz$ ($K/m$)", fontsize=ax_fontsize)
        ax.set_title("SOEC PEN Temperature Gradient", fontsize=title_fontsize)
        demarcate_ramps(ax)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
        ax.legend(fontsize=legend_fontsize, ncol=2)

        ax = axes[0, 1]
        ax2 = ax.twinx()
        ax.plot(
            time,
            results_dict["H2_production"],
            color="steelblue",
            linestyle=linestyle,
            label=f"{run} H2 Production",
        )
        ax2.plot(time, results_dict["efficiency_lhv"], color='orangered', linestyle=linestyle, label=f"{run} Efficiency (LHV)")
        demarcate_ramps(ax)
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim((0, 2.5))
        ax2.set_ylim((0, 1.4))
        ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
        ax.set_ylabel("Hydrogen Production (kg/s)", color="steelblue", fontsize=ax_fontsize)
        # ax2.set_ylabel("Energy per H2 mass (MJ/kg)", color="red",
        #                fontsize=ax_fontsize)
        ax2.set_ylabel("Efficiency (LHV)", color="orangered", fontsize=ax_fontsize)
        ax.set_title("Hydrogen Production and Efficiency", fontsize=title_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax2.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
        # ax.legend(fontsize=legend_fontsize)

        ax = axes[0, 0]

        ax.plot(time, results_dict["fuel_inlet_temperature"], label=f"{run} Fuel", color="steelblue", linestyle=linestyle)
        ax.plot(time, results_dict["sweep_inlet_temperature"], label=f"{run} Sweep", color="orangered", linestyle=linestyle)
        ax.plot(time, results_dict["cell_average_temperature"], label=f"{run} Average", color="forestgreen", linestyle=linestyle)

        ax.set_xlim(time[0], time[-1])
        ax.set_ylim((900, 1100))
        ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
        ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
        demarcate_ramps(ax)
        ax.set_title("SOEC Inlet and Stack Temperature", fontsize=title_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
        ax.legend(fontsize=legend_fontsize, ncol=2)

        ax = axes[1, 0]

        ax.plot(time, results_dict["fuel_outlet_temperature"], label=f"{run} Fuel", color="steelblue", linestyle=linestyle)
        ax.plot(time, results_dict["sweep_outlet_temperature"], label=f"{run} Sweep", color="orangered", linestyle=linestyle)

        ax.set_xlim(time[0], time[-1])
        ax.set_ylim((900, 1100))
        ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
        ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
        demarcate_ramps(ax)
        ax.set_title("SOEC Outlet Temperature", fontsize=title_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
        ax.legend(fontsize=legend_fontsize, ncol=2)

        if third_row:
            ax = axes[1, 1]
            ax2 = ax.twinx()
            ax.plot(
                time,
                results_dict["fuel_inlet_H2"],
                label=f"{run} Inlet $H_2$",
                color="steelblue",
                linestyle=linestyle,
            )
            ax.plot(
                time,
                0.05 * np.ones(time.shape),
                color="darkblue",
                linestyle="dotted"
            )
            # ax.plot(time,
            #         traj.vecs[str(soec.oxygen_inlet.mole_frac_comp[tf, "O2"])],
            #         label="Inlet $O_2$")
            ax.plot(
                time,
                results_dict["sweep_outlet_O2"],
                label=f"{run} Outlet $O_2$",
                color="orangered",
                linestyle=linestyle,
            )
            ax.plot(
                time,
                0.35 * np.ones(time.shape),
                color="saddlebrown",
                linestyle="dotted"
            )

            ax.plot(
                time,
                results_dict["soc_fuel_outlet_mole_frac_comp_H2"],
                color="forestgreen",
                label="Outlet H2",
                linestyle=linestyle,
            )
            ax.plot(
                time,
                np.ones(time.shape),
                color="darkgreen",
                linestyle="dotted"
            )
            demarcate_ramps(ax)
            ax.set_xlim(time[0], time[-1])
            ax.set_ylim((0, 1))
            #ax2.set_ylim((0, 100))
            ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
            ax.set_ylabel("Mole fraction", fontsize=ax_fontsize)
            #ax2.set_ylabel("Overall water conversion", fontsize=ax_fontsize, color="forestgreen")
            ax.set_title("Reactor feed and effluent concentrations", fontsize=title_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
            ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
            #ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
            #ax2.tick_params(axis="y", which="major", labelsize=tick_label_fontsize)
            #ax2.tick_params(axis="y", which="minor", labelsize=tick_label_fontsize)
            ax.legend(fontsize=legend_fontsize, ncol=2)

            ax = axes[2, 1]

            ax.plot(time, results_dict["fuel_heater_duty"] * 1e-6, label=f"{run} Fuel", color="steelblue", linestyle=linestyle)
            ax.plot(time, results_dict["sweep_heater_duty"] * 1e-6, label=f"{run} Sweep", color="orangered", linestyle=linestyle)
            if run == "PI":
                ax.plot(
                    time,
                    results_dict["feed_heater_inner_controller_mv_ref"] * 1e-6,
                    #label="Fuel reference",
                    color="darkblue",
                    linestyle="dotted"
                )
                ax.plot(
                    time,
                    results_dict["sweep_heater_inner_controller_mv_ref"] * 1e-6,
                    #label="Sweep reference",
                    color="saddlebrown",
                    linestyle="dotted"
                )

            demarcate_ramps(ax)
            ax.set_xlim(time[0], time[-1])
            ax.set_ylim((0, 13))
            ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
            ax.set_ylabel("Heater duty (MW)", fontsize=ax_fontsize)
            ax.set_title("Trim heater duties", fontsize=title_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
            ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)
            ax.legend(fontsize=legend_fontsize, ncol=2)

        fig.tight_layout()
        fig.savefig("plot_grid.png", dpi=360)
    # ax = axes[0, 0]
    # ax.legend(fontsize=legend_fontsize,)