import argparse
from simulations import SumoParallelSimulationHandler

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type = str, default = 'demo.sumocfg', help = 'sumo configuration path')
    parser.add_argument('--data',       type = str, default = 'simulation', help = 'sensor xml data path')
    parser.add_argument('--flow',       type = str, default = 'demo.rou.xml', help = 'demand xml data path')
    parser.add_argument('--fcd_output', type = str, default = 'fcd.xml', help = 'floating car data path')
    parser.add_argument('--duration',   type = int, default = 7200, help = 'simulation time')
    parser.add_argument('--period',     type = int, default = 360, help = 'interval time')
    parser.add_argument('--seed',       type = int, default = 2025, help = 'for simulation reproductive')    
    parser.add_argument('--mute_warnings',  action='store_true', default=False, help='mute sumo warnings')
    parser.add_argument('--mute_step_logs', action='store_true', default=False, help='mute step logs')
    args = parser.parse_args()
    # consider 10 \times 9 = 90 tazs
    taz_n = 10 # TODO: check number of taz
    matrix_dim = (taz_n, taz_n, (args.duration // args.period))
    # configs
    simulation_handler = SumoParallelSimulationHandler(sumo_sim_args=args, simulation_scenario='scenario_0', matrix_dim=matrix_dim)
    simulation_handler.parallel_simulations(thread_n=10, simulation_n=1000)