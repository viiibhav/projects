import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.io import loadmat
import cloudpickle as pickle
from nmpc_helper import alias_dict, get_time_coordinates


def get_nmpc_results(filepath):
    controls_dict = pickle.load(open(filepath + 'controls_dict.pkl', 'rb'))
    CVs_dict = pickle.load(open(filepath + 'cvs_dict.pkl', 'rb'))
    # states_dict = pickle.load(open(filepath + 'states_dict.pkl', 'rb'))
    h2_production_rate = pickle.load(
        open(filepath + 'h2_production_rate.pkl', 'rb')
    )
    power_dict = pickle.load(open(filepath + 'power_dict.pkl', 'rb'))
    setpoint_dict = pickle.load(open(filepath + 'setpoint_dict.pkl', 'rb'))
    sim_time_set = pickle.load(open(filepath + 'sim_time_set.pkl', 'rb'))
    try:
        dTdz_electrode_logbook = pickle.load(
            open(filepath + 'temperature_gradients.pkl', 'rb')
        )
    except:
        dTdz_electrode_logbook = pickle.load(
            open(filepath + 'dTdz_electrode_dict.pkl', 'rb')
        )

    # out = [
    #     controls_dict,
    #     CVs_dict,
    #     states_dict,
    #     h2_production_rate,
    #     dTdz_electrode_logbook,
    # ]
    out = [
        controls_dict,
        CVs_dict,
        h2_production_rate,
        power_dict,
        setpoint_dict,
        sim_time_set,
        dTdz_electrode_logbook,
    ]
    return out

def _demarcate_ramps(ax, results_dict):
    for tpoint in np.squeeze(results_dict["ramp_list"])[:-1]:
        ax.plot(
            np.array([tpoint, tpoint]) / 60 ** 2,
            [-1e9, 1e9],
            color="gray",
            linestyle="--",
        )

def plot_results(filename, nmpc_filepath, include_PI, savefig=False):
    results_dict = loadmat(filename)
    controls_dict, CVs_dict, h2_production_rate, power_dict, \
        setpoint_dict, sim_time_set, dTdz_electrode_logbook \
        = get_nmpc_results(nmpc_filepath)

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

    ax.plot(time, results_dict["potential"], color="darkorange", linewidth=2, linestyle="-.", label="PI")
    if include_PI:
        ax.plot(
            time,
            results_dict["voltage_controller_mv_ref"],
            color="black",
            linestyle="dotted",
        )

    key, = [k for k, v in alias_dict.items() if v == "potential"]
    potential = controls_dict[key]
    ax.plot(
        time_nmpc[:len(potential)],
        potential[:len(time_nmpc)],
        color="blue",
        linewidth=2,
        label="NMPC"
    )
    
    demarcate_ramps(ax)
    
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0.75, 1.45))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Cell potential (V)", fontsize=ax_fontsize)
    ax.set_title("SOEC Voltage", fontsize=title_fontsize)
    ax.legend(loc="best")
    if savefig:
        plt.savefig(nmpc_filepath + 'potential.png')
        plt.savefig(nmpc_filepath + 'potential.pdf')

    fig = plt.figure()
    ax = fig.subplots()

    ax.plot(
        time,
        results_dict["soec_fuel_inlet_flow"],
        color="blue",
        linestyle="-.",
        linewidth=2,
        label="Fuel PI",
    )
    ax.plot(
        time,
        results_dict["soec_oxygen_inlet_flow"],
        color="darkorange",
        linestyle="-.",
        linewidth=2,
        label="Sweep PI",
    )
    
    H2_in = CVs_dict["soec_fuel_inlet_flow"]
    O2_in = CVs_dict["soec_oxygen_inlet_flow"]
    ax.plot(
        time_nmpc,
        H2_in[:len(time_nmpc)],
        color="blue",
        linewidth=2,
        label="Fuel NMPC",
    )
    ax.plot(
        time_nmpc,
        O2_in[:len(time_nmpc)],
        color="darkorange",
        linewidth=2,
        label="Sweep NMPC",
    )
    
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0, 15000))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("SOEC inlet molar flow (mol/s)", fontsize=ax_fontsize)
    ax.set_title("Inlet molar flow rates", fontsize=title_fontsize)
    ax.legend()
    if savefig:
        plt.savefig(nmpc_filepath + 'inlet_flow_rates.png')
        plt.savefig(nmpc_filepath + 'inlet_flow_rates.pdf')

    fig = plt.figure()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(
        time,
        1e-6 * results_dict["fuel_heater_duty"],
        color="blue",
        linestyle="-.",
        linewidth=2,
        label="Fuel PI",
    )
    ax2.plot(
        time,
        1e-6 * results_dict["sweep_heater_duty"],
        color="darkorange",
        linestyle="-.",
        linewidth=2,
        label="Sweep PI",
    )
    if include_PI:
        ax1.plot(
            time,
            1e-6 * results_dict["feed_heater_inner_controller_mv_ref"],
            label="Fuel reference",
            color="blue",
            linestyle="dotted"
        )
        ax2.plot(
            time,
            1e-6 * results_dict["sweep_heater_inner_controller_mv_ref"],
            label="Sweep reference",
            color="darkorange",
            linestyle="dotted"
        )
    
    key_fuel, = [k for k, v in alias_dict.items() if v == "feed_heater_duty"]
    key_sweep, = [k for k, v in alias_dict.items() if v == "sweep_heater_duty"]
    feed_heater_duty = controls_dict[key_fuel]
    sweep_heater_duty = controls_dict[key_sweep]
    ax1.plot(
        time_nmpc[:len(feed_heater_duty)],
        1e-06 * np.array(feed_heater_duty[:len(time_nmpc)]),
        color="blue",
        linewidth=2,
        label="Fuel NMPC",
    )
    ax2.plot(
        time_nmpc[:len(sweep_heater_duty)],
        1e-06 * np.array(sweep_heater_duty[:len(time_nmpc)]),
        color="darkorange",
        linewidth=2,
        label="Sweep NMPC",
    )
    
    demarcate_ramps(ax1)
    demarcate_ramps(ax2)
    ax1.set_xlim(time[0], time[-1])
    ax2.set_xlim(time[0], time[-1])
    ax1.set_ylim((0, 7))
    ax2.set_ylim((0, 12))
    ax2.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax1.set_ylabel("Heater duty (MW)", fontsize=ax_fontsize)
    ax2.set_ylabel("Heater duty (MW)", fontsize=ax_fontsize)
    ax1.set_title("Fuel", fontsize=ax_fontsize)
    ax2.set_title("Sweep", fontsize=ax_fontsize)
    fig.suptitle("Trim heater duties", fontsize=title_fontsize)
    ax1.legend()
    ax2.legend()
    if savefig:
        plt.savefig(nmpc_filepath + 'trim_heater_duties.png')
        plt.savefig(nmpc_filepath + 'trim_heater_duties.pdf')


    fig = plt.figure()
    ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
    ax1.plot(
        time,
        results_dict["fuel_inlet_H2O"],
        color="blue",
        linestyle="-.",
        linewidth=2,
        label="Inlet H$_2$O PI",
    )
    ax1.plot(
        time,
        results_dict["fuel_outlet_H2O"],
        color="darkorange",
        linestyle="-.",
        linewidth=2,
        label="Outlet H$_2$O PI",
    )
    water_in = CVs_dict["fuel_inlet_H2O"]
    water_out = CVs_dict["fuel_outlet_H2O"]
    ax1.plot(
        time_nmpc,
        water_in[:len(time_nmpc)],
        color="blue",
        linewidth=2,
        label="Inlet H$_2$O NMPC",
    )
    ax1.plot(
        time_nmpc,
        water_out[:len(time_nmpc)],
        color="darkorange",
        linewidth=2,
        label="Outlet H$_2$O NMPC",
    )
    ax1.plot(time, 0.25 * np.ones(time.shape), color="gray", linestyle='--')
    demarcate_ramps(ax1)
    ax1.set_xlim(time[0], time[-1])
    ax1.set_ylim((0, 1))
    ax1.set_ylabel("Mole fraction", fontsize=ax_fontsize)
    ax1.legend(ncols=2)
    
    ax2.plot(
        time,
        results_dict["sweep_inlet_O2"],
        color="blue",
        linestyle="-.",
        linewidth=2,
        label="Inlet O$_2$ PI",
    )
    ax2.plot(
        time,
        results_dict["sweep_outlet_O2"],
        color="darkorange",
        linestyle="-.",
        linewidth=2,
        label="Outlet O$_2$ PI",
    )
    oxygen_in = CVs_dict["sweep_inlet_O2"]
    oxygen_out = CVs_dict["sweep_outlet_O2"]
    ax2.plot(
        time_nmpc,
        oxygen_in[:len(time_nmpc)],
        color="blue",
        linewidth=2,
        label="Inlet O$_2$ NMPC",
    )
    ax2.plot(
        time_nmpc,
        oxygen_out[:len(time_nmpc)],
        color="darkorange",
        linewidth=2,
        label="Outlet O$_2$ NMPC",
    )
    ax2.plot(time, 0.35 * np.ones(time.shape), color="gray", linestyle='--')
    demarcate_ramps(ax2)
    ax2.set_xlim(time[0], time[-1])
    ax2.set_ylim((0, 1))
    ax2.set_ylabel("Mole fraction", fontsize=ax_fontsize)
    ax2.legend(ncols=2)

    ax3.plot(
        time,
        results_dict["product_mole_frac_H2"],
        linewidth=2,
        color="darkorange",
        linestyle="",
        marker="o",
        label="Product H$_2$ PI",
    )
    product_h2 = CVs_dict["product_mole_frac_H2"]
    ax3.plot(
        time_nmpc,
        product_h2[:len(time_nmpc)],
        color="blue",
        linewidth=2,
        label="Product H$_2$ NMPC",
    )
    demarcate_ramps(ax3)
    ax3.set_xlim(time[0], time[-1])
    ax3.set_ylim((0.5, 1))
    ax3.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax3.set_ylabel("Mole fraction", fontsize=ax_fontsize)
    ax3.legend()
    fig.suptitle("Reactor feed and effluent concentrations", fontsize=title_fontsize)
    if savefig:
        plt.savefig(nmpc_filepath + 'feed_effluent_concentrations.png')
        plt.savefig(nmpc_filepath + 'feed_effluent_concentrations.pdf')

    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(
        time,
        results_dict["H2_production"],
        color="darkorange",
        linestyle="-.",
        linewidth=2,
        label="PI",
    )
    ax.plot(time, -0.9192 * np.ones(time.shape), 'r:')
    ax.plot(time, 2 * np.ones(time.shape), 'r:')
    
    ax.plot(
        time_nmpc[:len(h2_production_rate)],
        np.array(h2_production_rate[:len(time_nmpc)]),
        color="blue",
        linewidth=2,
        label="NMPC",
    )
    
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((-1.25, 2.5))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Hydrogen Production Rate (kg/s)", fontsize=ax_fontsize)
    ax.set_title("Instantaneous H$_2$ production rate", fontsize=title_fontsize)
    ax.legend()
    if savefig:
        plt.savefig(nmpc_filepath + 'hydrogen_production_rate.png')
        plt.savefig(nmpc_filepath + 'hydrogen_production_rate.pdf')

    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(
        time,
        results_dict["steam_feed_rate"],
        color="darkorange",
        linewidth=2,
        linestyle="-.",
        label="PI",
    )
    steam_feed_rate = CVs_dict["steam_feed_rate"]
    ax.plot(
        time_nmpc,
        steam_feed_rate[:len(time_nmpc)],
        color="blue",
        linewidth=2,
        label="NMPCs",
    )
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((0, 5000))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Steam feed rate (mol/s)", fontsize=ax_fontsize)
    ax.set_title("Steam feed rate", fontsize=title_fontsize)
    ax.legend()
    if savefig:
        plt.savefig(nmpc_filepath + 'steam_feed_rate.png')
        plt.savefig(nmpc_filepath + 'steam_feed_rate.pdf')

    fig = plt.figure()
    ax = fig.subplots()
    # ax2 = ax.twinx()
    ax.plot(
        time,
        1e-06 * results_dict["total_electric_power"],
        color='darkorange',
        linewidth=2,
        linestyle="-.",
        label="PI",
    )
    total_electric_power = CVs_dict["total_electric_power"]
    # total_electric_power = power_dict["total_power"]
    ax.plot(
        time_nmpc,
        1e-06 * np.array(total_electric_power[:len(time_nmpc)]),
        color="blue",
        linewidth=2,
        label="NMPC",
    )
    # ax2.plot(time, results_dict["efficiency_lhv"], 'r', label="Efficiency (LHV)")
    demarcate_ramps(ax)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((-125, 350))
    # ax2.set_ylim((0, 1.4))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Power usage (MW)", color="blue", fontsize=ax_fontsize)
    # ax2.set_ylabel("Energy per H2 mass (MJ/kg)", color="red",
    #                fontsize=ax_fontsize)
    # ax2.set_ylabel("Efficiency (LHV)", color="red",
    #                fontsize=ax_fontsize)
    ax.set_title("Power usage", fontsize=title_fontsize)
    ax.legend()
    if savefig:
        plt.savefig(nmpc_filepath + 'power_usage.png')
        plt.savefig(nmpc_filepath + 'power_usage.pdf')

    fig = plt.figure()
    ax = fig.subplots()

    ax.plot(time, results_dict["fuel_inlet_temperature"], linewidth=2, linestyle="-.", label="Fuel PI", color="darkorange")
    ax.plot(time, results_dict["sweep_inlet_temperature"], linewidth=2, linestyle="-.", label="Sweep PI", color="blue")
    ax.plot(time, results_dict["cell_average_temperature"], linewidth=2, linestyle="-.", label="Cell average PI", color="black")

    if include_PI:
        ax.plot(
            time,
            results_dict["feed_heater_inner_controller_setpoint"],
            label="Fuel target",
            color="red",
            linestyle="dotted"
        )
        ax.plot(
            time,
            results_dict["sweep_heater_inner_controller_setpoint"],
            label="Sweep target",
            color="blue",
            linestyle="dotted"
        )
    
    fuel_inlet_temperature = CVs_dict["fuel_inlet_temperature"]
    sweep_inlet_temperature = CVs_dict["sweep_inlet_temperature"]
    cell_average_temperature = CVs_dict["cell_average_temperature"]
    ax.plot(
        time_nmpc,
        fuel_inlet_temperature[:len(time_nmpc)],
        color="darkorange",
        linewidth=2,
        label="Fuel NMPC",
    )
    ax.plot(
        time_nmpc,
        sweep_inlet_temperature[:len(time_nmpc)],
        color="blue",
        linewidth=2,
        label="Sweep NMPC",
    )
    ax.plot(
        time_nmpc,
        cell_average_temperature[:len(time_nmpc)],
        color="gray",
        linewidth=2,
        label="Cell average NMPC",
    )

    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((850, 1150))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
    demarcate_ramps(ax)
    ax.set_title("SOEC inlet temperature", fontsize=title_fontsize)
    ax.legend()
    if savefig:
        plt.savefig(nmpc_filepath + 'SOFC_inlet_temperatures.png')
        plt.savefig(nmpc_filepath + 'SOFC_inlet_temperatures.pdf')

    fig = plt.figure()
    ax = fig.subplots()

    ax.plot(time, results_dict["fuel_outlet_temperature"], label="Fuel PI", color="darkorange", linestyle="-.", linewidth=2)
    ax.plot(time, results_dict["sweep_outlet_temperature"], label="Sweep PI", color="blue", linestyle="-.", linewidth=2)
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
        np.array(fuel_outlet_temperature[:len(time_nmpc)]),
        color="darkorange",
        linewidth=2,
        label="Fuel NMPC",
    )
    ax.plot(
        time_nmpc[:len(sweep_outlet_temperature)],
        np.array(sweep_outlet_temperature[:len(time_nmpc)]),
        color="blue",
        linewidth=2,
        label="Sweep NMPC",
    )

    ax.set_xlim(time[0], time[-1])
    ax.set_ylim((890, 1050))
    ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
    ax.set_ylabel("Temperature (K)", fontsize=ax_fontsize)
    demarcate_ramps(ax)
    ax.set_title("SOEC outlet temperature", fontsize=title_fontsize)
    ax.legend()
    if savefig:
        plt.savefig(nmpc_filepath + 'SOFC_outlet_temperatures.png')
        plt.savefig(nmpc_filepath + 'SOFC_outlet_temperatures.pdf')

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


def plot_manipulated_variables(filename, nmpc_filepath, include_PI=True):
    results_dict = loadmat(filename)
    controls_dict, CVs_dict, h2_production_rate, power_dict, \
        setpoint_dict, sim_time_set, dTdz_electrode_logbook \
        = get_nmpc_results(nmpc_filepath)

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

    for i in controls_dict.keys():
        fig = plt.figure()
        ax = fig.subplots()
        values = controls_dict[i][:len(time_nmpc)]
        target = list(setpoint_dict[alias_dict[i]].values())[:len(time_nmpc)]
        ax.plot(
            time_nmpc,
            values,
            color='blue',
            linewidth=2,
        )
        ax.plot(
            time_nmpc,
            target,
            color='black',
            linestyle=':',
        )
        demarcate_ramps(ax)
        ax.set_xlim(time_nmpc[0], time_nmpc[-1])
        if alias_dict[i] == "condenser_hot_outlet_temperature":
            ymin = 300
            ymax = 350
        else:
            ymin = min(list(target) + list(values))
            ymax = max(list(target) + list(values))
            # ymin = min(values)
            # ymax = min(values)
        ylim = [
            ymin - 0.1 * (ymax - ymin),
            ymax + 0.1 * (ymax - ymin),
        ]
        ax.set_ylim(ylim)
        ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
        ax.set_title(alias_dict[i], fontsize=title_fontsize)


def plot_controlled_variables(filename, nmpc_filepath, include_PI=True):
    results_dict = loadmat(filename)
    controls_dict, CVs_dict, h2_production_rate, power_dict, \
        setpoint_dict, sim_time_set, dTdz_electrode_logbook \
        = get_nmpc_results(nmpc_filepath)

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

    for i in CVs_dict.keys():
        fig = plt.figure()
        ax = fig.subplots()
        values = CVs_dict[i][:len(time_nmpc)]
        try:
            target = list(setpoint_dict[alias_dict[i]].values())[:len(time_nmpc)]
        except:
            target = None
        ax.plot(
            time_nmpc,
            values,
            color='blue',
            linewidth=2,
        )
        try:
            ax.plot(
                time_nmpc,
                target,
                color='black',
                linestyle=':',
            )
        except:
            pass
        demarcate_ramps(ax)
        ax.set_xlim(time_nmpc[0], time_nmpc[-1])
        try:
            ymin = min(list(target) + list(values))
            ymax = max(list(target) + list(values))
        except:
            ymin = min(values)
            ymax = min(values)
        ylim = [
            ymin - 0.1 * (ymax - ymin),
            ymax + 0.1 * (ymax - ymin),
        ]
        ax.set_ylim(ylim)
        ax.set_xlabel("Time (hr)", fontsize=ax_fontsize)
        ax.set_title(i, fontsize=title_fontsize)
    
    fig, ax = plt.subplots()
    for iz in iz_plot:
        ax.plot(
            time_nmpc,
            dTdz_electrode_logbook[iz][:len(time_nmpc)],
            linewidth=2,
            label=f"node {iz}"
        )
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim((-1000, 1000))
    # ax.set_xlabel("Time (hr)", fontsize=12)
    ax.set_ylabel("$dT/dz$ ($K/m$)", fontsize=12)
    ax.set_title("SOEC PEN temperature gradient, NMPC", fontsize=12)
    # demarcate_ramps(ax)
    ax.legend(loc='best')



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