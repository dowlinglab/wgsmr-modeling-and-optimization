# import packages
import pyomo.environ as pyo
import math
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from matplotlib import cm
from pint import UnitRegistry

# WGS-MR model
def create_model(
    temp_retentate=553,
    temp_membrane=553,
    CO_comp_feed=0.2,
    H2O_comp_feed=0.2,
    CO2_comp_feed=0.1,
    H2_comp_feed=0.5,
    CH4_comp_feed=0,
    N2_comp_feed=0,
    feed_flow=1.256e-3,
    feed_ghsv=2500,
    feed_pressure=1e6,
    pressure_drop_retentate=35000,
    CO_comp_sweep=0,
    H2O_comp_sweep=0,
    CO2_comp_sweep=0,
    H2_comp_sweep=0,
    CH4_comp_sweep=0,
    N2_comp_sweep=1,
    sweep_flow=0,
    sweep_pressure=1e5,
    pressure_drop_permeate=0,
    pre_exponent=0.0162,
    E_R=3098,
    pressure_exponent=0.5,
    vol_reactor=3.927e-5,
    area_membrane=0.0157,
    rho_catalyst=1.38e3,
    num_elements=20,
    T_lb=300,
    T_ub=800,
    GHSV_lb=500,
    GHSV_ub=4000,
    GHSV_sweep=0,
    with_reaction=True,
    discretize_temperature=False,
    separate_temperatures=False,
    initialize_pressure_strategy="constant",
    feed_input_mode="flowrate",
    no_permeation=False,
    permeance_scaling=1.0,
    reaction_scaling=1.0,
    max_h2=True,
):
    """
    Creates a cooncrete pyomo model for the water-gas shift membrane reactor.

    Arguments
    ---------
    temp_retentate: float, default 553
        temperature of membrane reactor or retentate-side in [K]
    temp_membrane: float, default 553
        temperature of membrane-side in [K]
    CO_comp_feed: float, default 0.2
        molar composition of CO in the feed [unitless]
    H2O_comp_feed: float, default 0.2
        molar composition of water in the feed [unitless]
    CO2_comp_feed: float, default 0.1
        molar composition of CO2 in the feed [unitless]
    H2_comp_feed: float, default 0.5
        molar composition of H2 in the feed [unitless]
    CH4_comp_feed: float, default 0
        molar composition of CH4 in the feed [unitless]
    N2_comp_feed: float, default 0
        molar composition of N2 in the feed [unitless]
    feed_flow: float, default 1.256e-3
        molar flow rate of feed in [mol/s]
    feed_ghsv: float, default 2500
        feed GHSV [hr-1]
    feed_pressure: float, default 1e6
        feed pressure in [Pa]
    pressure_drop_retentate: float, default35000
        total pressure drop on retentate-side [Pa]. We assumed 3.5% of feed pressure as default.
    CO_comp_sweep: float, default 0
        molar composition of CO in the sweep [unitless]
    H2O_comp_sweep: float, default 0
        molar composition of water in the sweep [unitless]
    CO2_comp_sweep: float, default 0
        molar composition of CO2 in the sweep [unitless]
    H2_comp_sweep: float, default 0
        molar composition of H2 in the sweep [unitless]
    CH4_comp_sweep: float, default 0
        molar composition of CH4 in the sweep [unitless]
    N2_comp_sweep: float, default 1
        molar composition of N2 in the sweep [unitless]
    sweep_flow: float, default 0
        molar flow rate of sweep in [mol/s]
    sweep_pressure: float, default 1e5
        sweep pressure in [Pa]
    pressure_drop_permeate: float, default 0
        total pressure drop on permeate-side [Pa]. We assumed 3.5% of sweep pressure.
    pre_exponent: float, default 0.0162
        pre-exponential factorfor hydrogen permeation in temp_reactor
    E_R: float, default 3098
        activation energy of hydrogen diffusion, E over universal gas consant, R, in [K]
    pressure_exponent: float, default 0.5
        partial pressure exponent
    vol_reactor: float, default 3.927e-5
        volume of catalyst packing/bed in [m3]
    area_membrane: float, default 0.0157
        permeation area of membrane in [m2]
    rho_catalyst: float, default 1.38e3
        density of catalyst in [kg/m3]
    num_elements: int, default 20
        number of finite elements [unitless]
    T_lb: float, default 300
        temperature lower bound [K]
    T_ub: float, default 800
        temperature upper bound [K]
    GHSV_lb: float, default 500
        GHSV lower bound [hr-1]
    GHSV_ub: float, default 4000
        GHSV upper bound [hr-1]
    GHSV_sweep: float, default 0
        GHSV of the sweep gas [hr-1]
    with_reaction: bool, default True, 
        toggle between WGS-MR (True) and membrane separator (False: without WGS reaction)
    discretize_temperature: bool, default False
        if True, index reactor temperature by finite volumes along the length of the reactor
    separate_temperatures: bool, default False
        if True, model reaction-side and membrane temperature separately
    initialize_pressure_strategy: str, default "constant"
        "constant" or "linear", toggle to specify mode of pressure initialization
    feed_input_mode: str, default "flowarte"
        "flowrate" or ["GHSV" or "ghsv"], specify if feed in flowrate or GHSV.
    no_permeation: bool, default False
        toggle between WGS-MR (False) or conventional WGS (True: without membrane separation)
    permeance_scaling: float, default 1
        scale permeance of membrane material
    reaction_scaling: float, default 1
        scale rate of WGS reaction
    max_h2: bool, default True
        if True, maximize hyrogen in permeate, otherwise constant objective (ideal for simulation mode)


    Returns
    -------
    m: concreete pyomo model
        
    Raises
    -----
    ValueError
        if feed composition does not sum to 1, OR
        sweep composition does not sum to 1, OR
        `feed_input_mode` is none of "flowrate", "GHSV", OR "ghsv"]
    """
    #################################################################
                  # MODEL SET-UP AND PRELIMINARY DATA
    #################################################################
    
    # input checks
    # check that feed composition sums to 1
    sum_feed = (CO_comp_feed + H2O_comp_feed + CO2_comp_feed + H2_comp_feed + CH4_comp_feed + N2_comp_feed)
    if abs(sum_feed - 1) > 0.001:
        raise ValueError ("Feed composition must sum to 1. The provided feed composition sums to {}".format(sum_feed))
        
    # check that the sweep composition sums to 1
    sum_sweep = (CO_comp_sweep+H2O_comp_sweep+CO2_comp_sweep+H2_comp_sweep+CH4_comp_sweep+N2_comp_sweep)
    if abs(sum_sweep - 1) > 0.001:
        raise ValueError ("Sweep composition must sum to 1. The sweep cmposition provided sums to {}".format(sum_sweep))
        
    # check that the feed_input_mode is correct
    if feed_input_mode not in ["flowrate", "GHSV", "ghsv"]:
        raise ValueError ("feed_input_mode must be one of 'flowrate' or 'GHSV' or 'ghsv'")
    
    # Create instance of a concrete pyomo model
    m = pyo.ConcreteModel()

    # Specify number of finite elelments
    m.N = num_elements

    # specify tolerance factor for partial pressure
    epsilon = 1e-8

    # Declare set of gas components
    m.COMPONENTS = pyo.Set(initialize=["CO", "H2O", "CO2", "H2", "CH4", "N2"])

    # Declare set of finite elements
    m.ELEMENTS = pyo.Set(initialize=range(0, m.N))

    # Specify feed component flow rates (F_i = x_i*F)
    feed = {
        "CO": CO_comp_feed * feed_flow,
        "H2O": H2O_comp_feed * feed_flow,
        "CO2": CO2_comp_feed * feed_flow,
        "H2": H2_comp_feed * feed_flow,
        "CH4": CH4_comp_feed * feed_flow,
        "N2": N2_comp_feed * feed_flow,
    }  # component flow rates in [mol/s].

    # Specify sweep compoent flow rates
    sweep = {
        "CO": CO_comp_sweep * sweep_flow,
        "H2O": H2O_comp_sweep * sweep_flow,
        "CO2": CO2_comp_sweep * sweep_flow,
        "H2": H2_comp_sweep * sweep_flow,
        "CH4": CH4_comp_sweep * sweep_flow,
        "N2": N2_comp_sweep * sweep_flow,
    }  # component flow rates in [mol/s]. 
    
    # assemble feed composition
    feed_comp = {
        "CO": CO_comp_feed,
        "H2O": H2O_comp_feed,
        "CO2": CO2_comp_feed,
        "H2": H2_comp_feed,
        "CH4": CH4_comp_feed, 
        "N2": N2_comp_feed 
    }  # component flow rates in [mol/s].

    # assemble sweep composition
    sweep_comp = {
        "CO": CO_comp_sweep,
        "H2O": H2O_comp_sweep,
        "CO2": CO2_comp_sweep,
        "H2": H2_comp_sweep,
        "CH4": CH4_comp_sweep,
        "N2": N2_comp_sweep 
    }  # component flow rates in [mol/s].
    
    # Specify stoichiometric coefficient
    stoichiometric_ceofficients = {
        "CO": -1,
        "H2O": -1,
        "CO2": 1,
        "H2": 1,
        "CH4": 0,
        "N2": 0,
    }  # stoichiometric coefficients in water-gas shift reaction

    # Generate data for initialize pressure
    p_retentate_list = np.linspace(
        feed_pressure, feed_pressure - pressure_drop_retentate, num_elements
    )

    # Generate data for initializing permeate-side pressure
    p_permeate_list = np.linspace(
        sweep_pressure - pressure_drop_permeate, sweep_pressure, num_elements
    )

    #############################################################
                       # MODEL PARAMETERS 
    #############################################################

    # add permeance scaling parameter
    m.permeance_scaling = pyo.Param(initialize=permeance_scaling, mutable=True)
    
    # add reaction scaling parameter
    m.reaction_scaling = pyo.Param(initialize=reaction_scaling, mutable=True)
    

    # pre-exponential factor for permeance in hydrogen flux [mol/m2-s-Pa^(0.5)]
    m.pre_exponential_factor = pyo.Param(initialize=pre_exponent)

    # ratio of activation energy for hydrogen permeation to universal gas constant [K]
    m.E_R = pyo.Param(
        initialize=E_R
    )  # Reformulate to take only value of E, value of R will be specified as a constant under inputs

    # partial pressure exponent
    m.n = pyo.Param(initialize=pressure_exponent)

    # reaction volume per finite element [m3]
    m.volume = pyo.Param(initialize=(vol_reactor / m.N))

    # permeation area per finite element [m2]
    m.area = pyo.Param(initialize=(area_membrane / m.N))

    # catalyst density [kg/m3]
    m.rho_catalyst = pyo.Param(initialize=rho_catalyst)

    # feed pressure [Pa]
    m.pressure_feed = pyo.Param(initialize=feed_pressure,mutable=True)

    # sweep pressure [Pa]
    m.pressure_sweep = pyo.Param(initialize=sweep_pressure)

    # pressure drop in retentate/shell [Pa]
    m.delta_p_retentate = pyo.Param(initialize=pressure_drop_retentate)

    # pressure drop in permeate/tube [Pa]
    m.delta_p_permeate = pyo.Param(initialize=pressure_drop_permeate)

    if feed_input_mode == "flowrate":
        # component flow rates for feed [mol/s]
        m.flow_feed = pyo.Param(m.COMPONENTS, initialize=feed, mutable=True)

        # component flow rates for sweep [mol/s]
        m.flow_sweep = pyo.Param(m.COMPONENTS, initialize=sweep, mutable=True)

    # stoichiometric coefficients for water-gas shift reaction
    m.coeff = pyo.Param(m.COMPONENTS, initialize=stoichiometric_ceofficients)
    
    # universal gas constant 
    m.R = pyo.Param(initialize=8.3145) # J/mol/K

    m.T_stp = pyo.Param(initialize=273)
    m.P_stp = pyo.Param(initialize=101325)
    
    # feed composition
    m.y_feed = pyo.Param(m.COMPONENTS, initialize=feed_comp)

    # feed composition
    m.y_sweep = pyo.Param(m.COMPONENTS, initialize=sweep_comp)
    
    if feed_input_mode == "GHSV" or feed_input_mode == "ghsv":
        # sweep ghsv
        m.ghsv_sweep = pyo.Param(initialize=GHSV_sweep)
    
    #########################################################
                       # MODEL VARIABLES
    #########################################################

    #max_flow_rate = 1.256e-1
    max_flow_rate = 1.0
    
    # reactor temperature
    if separate_temperatures:
        if discretize_temperature:
            m.temp_retentate = pyo.Var(m.ELEMENTS,initialize=temp_retentate, bounds=(T_lb,T_ub))
            m.temp_membrane = pyo.Var(m.ELEMENTS,initialize=temp_membrane, bounds=(T_lb,T_ub))
        else:
            m.temp_retentate = pyo.Var(initialize=temp_retentate, bounds=(T_lb,T_ub))
            m.temp_membrane = pyo.Var(initialize=temp_membrane, bounds=(T_lb,T_ub))
    else:
        if discretize_temperature:
            m.T = pyo.Var(m.ELEMENTS,initialize=temp_retentate, bounds=(T_lb,T_ub))
        else:
            m.T = pyo.Var(initialize=temp_retentate, bounds=(T_lb,T_ub))
        m.temp_retentate = pyo.Expression(expr=m.T)
        m.temp_membrane = pyo.Expression(expr=m.T)
    
    if feed_input_mode == "GHSV" or feed_input_mode == "ghsv":
        # gas hourly space velocity, GHSV (hr-1)
        m.ghsv = pyo.Var(initialize=feed_ghsv, bounds=(GHSV_lb,GHSV_ub))
    
    # retentate-side component molar flow rate [mol/s]
    m.flow_retentate = pyo.Var(
        m.ELEMENTS, m.COMPONENTS, initialize=3e-4, bounds=(0, max_flow_rate)
    )

    # permeate-side component molar flow rate [mol/s]
    m.flow_permeate = pyo.Var(
        m.ELEMENTS, m.COMPONENTS, initialize=3e-4, bounds=(0, max_flow_rate)
    )

    # retentate-side molar composition
    m.composition_retentate = pyo.Var(
        m.ELEMENTS, m.COMPONENTS, initialize=0.5, bounds=(0, 1)
    )

    # permeate-side molar composition
    m.composition_permeate = pyo.Var(
        m.ELEMENTS, m.COMPONENTS, initialize=0.5, bounds=(0, 1)
    )

    pressure_lower_bound = 1e3  # in Pa
    pressure_upper_bound = 1e7  # in Pa
    if initialize_pressure_strategy == "linear":

        # initialize retentate pressure
        m.retentate_pressure_init = {}
        for i, v in enumerate(m.ELEMENTS):
            m.retentate_pressure_init[v] = p_retentate_list[i]

        # retentate pressure [Pa]
        m.pressure_retentate = pyo.Var(
            m.ELEMENTS,
            initialize=m.retentate_pressure_init,
            bounds=(pressure_lower_bound, pressure_upper_bound),
        )

        # initialize permeate pressure
        m.permeate_pressure_init = {}
        for i, v in enumerate(m.ELEMENTS):
            m.permeate_pressure_init[v] = p_permeate_list[i]

        # permeate pressure [Pa]
        m.pressure_permeate = pyo.Var(
            m.ELEMENTS,
            initialize=m.permeate_pressure_init,
            bounds=(pressure_lower_bound, pressure_upper_bound),
        )
    elif initialize_pressure_strategy == "constant":

        # retentate pressure [Pa]
        m.pressure_retentate = pyo.Var(
            m.ELEMENTS,
            initialize=1e5,
            bounds=(pressure_lower_bound, pressure_upper_bound),
        )

        # permeate pressure [Pa]
        m.pressure_permeate = pyo.Var(
            m.ELEMENTS,
            initialize=1e5,
            bounds=(pressure_lower_bound, pressure_upper_bound),
        )
    else:
        raise ValueError(
            "initialize_pressure_strategy must be either 'linear' or 'constant'"
        )

        
    # max_reaction_rate = 1e-1
    max_reaction_rate = 1.0
    
    # reaction rate [mol/g/min]
    m.reaction_rate = pyo.Var(
        m.ELEMENTS, initialize=0.00509365, bounds=(0, max_reaction_rate)
    )  #Initial value obtained by manual calculation of expected reaction rate. BEST PRACTICE: initialize reaction rate using the 
    # init_reaction_rate function, thus this initial value does not matter.

    ###########################################################
                       # MODEL CONSTRAINTS 
    ###########################################################

    # retentate-side pressure drop constraint
    def RetentatePressureDropRule(m, j):
        '''Add retentate-side pressure drop expression.'''
        if j == m.ELEMENTS.first():
            return (
                m.pressure_retentate[j] == m.pressure_feed
            )  # feed pressure boundary condition
        return (
            m.pressure_retentate[j]
            == m.pressure_retentate[j - 1] - m.delta_p_retentate / m.N
        )

    m.retentate_pressure_constraint = pyo.Constraint(
        m.ELEMENTS, rule=RetentatePressureDropRule
    )

    # permeate-side pressure drop constraint
    def PermeatePressureDropRule(m, j):
        '''Add permeate-side pressure drop expression.'''
        if j == m.ELEMENTS.last():
            return (
                m.pressure_permeate[j] == m.pressure_sweep
            )  # sweep pressure boundary condition
        return (
            m.pressure_permeate[j]
            == m.pressure_permeate[j + 1] - m.delta_p_permeate / m.N
        )

    m.permeate_pressure_constraint = pyo.Constraint(
        m.ELEMENTS, rule=PermeatePressureDropRule
    )

    # flux expression: 
    def MembraneFluxRule(m, j, i):
        '''Add flux expression.'''
        if no_permeation:
            return 0
        else:
            if i == "H2":
                if discretize_temperature:
                    return m.permeance_scaling*m.pre_exponential_factor*pyo.exp(-m.E_R / m.temp_membrane[j]) * (
                        (
                            abs(
                                m.pressure_retentate[j] * m.composition_retentate[j, i]
                                + epsilon
                            )
                        )
                        ** m.n
                        - (
                            abs(
                                m.pressure_permeate[j] * m.composition_permeate[j, i]
                                + epsilon
                            )
                        )
                        ** m.n
                    )
                else:
                    return m.permeance_scaling*m.pre_exponential_factor*pyo.exp(-m.E_R / m.temp_membrane) * (
                        (
                            abs(
                                m.pressure_retentate[j] * m.composition_retentate[j, i]
                                + epsilon
                            )
                        )
                        ** m.n
                        - (
                            abs(
                                m.pressure_permeate[j] * m.composition_permeate[j, i]
                                + epsilon
                            )
                        )
                        ** m.n
                    )
            return 0  # infinite selectivity for hydrogen

    m.N_flux = pyo.Expression(m.ELEMENTS, m.COMPONENTS, rule=MembraneFluxRule)

    # reaction rate constraint
    def ReactionRateRule(m, j):
        '''Add reaction rate expression.'''
        if not with_reaction:
            return m.reaction_rate[j] == 0
        elif discretize_temperature:
            return m.reaction_rate[j] * (0.0126 * pyo.exp(4639 / m.temp_retentate[j])) * (
                1
                + 1.0197e-5
                * m.pressure_retentate[j]
                * (
                    2.2
                    * pyo.exp(101.5 / m.temp_retentate[j])
                    * m.composition_retentate[j, "CO"]
                    + 0.4
                    * pyo.exp(158.3 / m.temp_retentate[j])
                    * m.composition_retentate[j, "H2O"]
                    + 0.047
                    * pyo.exp(2737.9 / m.temp_retentate[j])
                    * m.composition_retentate[j, "CO2"]
                    + 0.05
                    * pyo.exp(596.1 / m.temp_retentate[j])
                    * m.composition_retentate[j, "H2"]
                )
            ) ** 2 == m.reaction_scaling.value*0.92 * pyo.exp(
                -454.3 / m.temp_retentate[j]
            ) * 1.03982e-10 * m.pressure_retentate[
                j
            ] ** 2 * (
                m.composition_retentate[j, "CO"]
                * m.composition_retentate[j, "H2O"]
                * (0.0126 * pyo.exp(4639 / m.temp_retentate[j]))
                - m.composition_retentate[j, "CO2"] * m.composition_retentate[j, "H2"]
            )
        else:
            return m.reaction_rate[j] * (0.0126 * pyo.exp(4639 / m.temp_retentate)) * (
                1
                + 1.0197e-5
                * m.pressure_retentate[j]
                * (
                    2.2
                    * pyo.exp(101.5 / m.temp_retentate)
                    * m.composition_retentate[j, "CO"]
                    + 0.4
                    * pyo.exp(158.3 / m.temp_retentate)
                    * m.composition_retentate[j, "H2O"]
                    + 0.047
                    * pyo.exp(2737.9 / m.temp_retentate)
                    * m.composition_retentate[j, "CO2"]
                    + 0.05
                    * pyo.exp(596.1 / m.temp_retentate)
                    * m.composition_retentate[j, "H2"]
                )
            ) ** 2 == m.reaction_scaling.value*0.92 * pyo.exp(
                -454.3 / m.temp_retentate
            ) * 1.03982e-10 * m.pressure_retentate[
                j
            ] ** 2 * (
                m.composition_retentate[j, "CO"]
                * m.composition_retentate[j, "H2O"]
                * (0.0126 * pyo.exp(4639 / m.temp_retentate))
                - m.composition_retentate[j, "CO2"] * m.composition_retentate[j, "H2"]
            )
        
    m.reaction_rate_constraint = pyo.Constraint(m.ELEMENTS, rule=ReactionRateRule)
    
    # add expression for total feed flow - serves as F0 in nondimensionalized model
    def TotalFeedRule(m):
        '''Calculate the total feed flowrate.'''
        if feed_input_mode == "flowrate":
            return sum(m.flow_feed[i] for i in m.COMPONENTS)
        else:
            return 1
    m.total_feed = pyo.Expression(rule=TotalFeedRule)
    
    # expression for feed factor based on feed input mode specified
    def FeedFactorRule(m):
        '''Sepcify factor for converting from dimensionless flowrate to mol/s.'''
        if feed_input_mode == "flowrate":
            return 1/m.total_feed
        elif feed_input_mode == "GHSV" or feed_input_mode == "ghsv":
            return 3600*m.R*m.T_stp/(m.N*m.P_stp*m.ghsv*m.volume) #### REMOVE m.N
        else:
            raise ValueError("feed_input_mode must be 'flowrate','GHSV' or 'ghsv'")
    m.input_factor = pyo.Expression(rule=FeedFactorRule)

    # retentate-side material balance    
    def RetentateMaterialRule(m, j, i):
        '''Add expression for material balances on the retentate-side.'''
        if feed_input_mode == "flowrate":
            if j == m.ELEMENTS.first():
                element_feed = m.y_feed[i] # feed composition (dimensionless)
            else:
                element_feed = m.flow_retentate[j - 1, i] # prior element in cascade
        elif feed_input_mode == "GHSV" or feed_input_mode == "ghsv":
            if j == m.ELEMENTS.first():
                element_feed = m.y_feed[i] 
            else:
                element_feed = m.flow_retentate[j - 1, i] # prior element in cascade
        else:
            raise ValueError("feed_input_mode must be 'flowrate','GHSV' or 'ghsv'")
        return (
            m.flow_retentate[j, i]
            == element_feed
            + m.coeff[i] * m.reaction_rate[j] * 16.6667 * m.rho_catalyst * m.volume * m.input_factor
            - m.N_flux[j, i] * m.area * m.input_factor 
        )  # multiply by 16.6667 to convert reaction rate from [mol/g/min] to [mol/kg/s]. # Divide by "total_feed" to make dimensionless

        
    m.retentate_material_constraint = pyo.Constraint(
        m.ELEMENTS, m.COMPONENTS, rule=RetentateMaterialRule
    )
    
    # permeate side material balance
    def PermeateMaterialBalanceRule(m, j, i):
        '''Add expression for material balances on the permeate-side'''
        if feed_input_mode == "flowrate":
            if j == m.ELEMENTS.last():
                element_feed = m.flow_sweep[i]/m.total_feed
            else:
                element_feed = m.flow_permeate[j + 1, i] # next element in cascade
        elif feed_input_mode == "GHSV" or feed_input_mode == "ghsv":
            if j == m.ELEMENTS.last():
                element_feed = m.y_sweep[i] * m.ghsv_sweep/m.ghsv # sweep boundary condition. 
            else:
                element_feed = m.flow_permeate[j + 1, i] # next element in cascade
        else:
            raise ValueError("feed_input_mode must be 'flowrate','GHSV' or 'ghsv'")
            
        return (
            m.flow_permeate[j, i] == element_feed + m.N_flux[j, i] * m.area * m.input_factor 
        ) # Divide by "total_feed" to make dimensionless

    m.permeate_material_constraint = pyo.Constraint(
        m.ELEMENTS, m.COMPONENTS, rule=PermeateMaterialBalanceRule
    )
    
    # retentate-side molar composition constraint
    def RetentateCompositionRule(m, j, i):
        '''Add expression for retentate-side molar composition'''
        return (
            m.composition_retentate[j, i]
            * sum(m.flow_retentate[j, i] for i in m.COMPONENTS)
            == m.flow_retentate[j, i]
        )

    m.retentate_composition_constraint = pyo.Constraint(
        m.ELEMENTS, m.COMPONENTS, rule=RetentateCompositionRule
    )

    # permeate-side molar composition constraint
    def PermeateCompositionRule(m, j, i):
        '''Add expression for permeate-side molar composition'''
        return (
            m.composition_permeate[j, i]
            * sum(m.flow_permeate[j, i] for i in m.COMPONENTS)
            == m.flow_permeate[j, i]
        )

    m.permeate_composition_constraint = pyo.Constraint(
        m.ELEMENTS, m.COMPONENTS, rule=PermeateCompositionRule
    )

    ###########################################################
                          # OBJECTIVE 
    ###########################################################

    def ObjectiveRule(m):
        '''Add objective for optimization problem'''
        if max_h2:
            return m.flow_permeate[m.ELEMENTS.first(), "H2"]
        else:
            return 0
   

    m.OBJ = pyo.Objective(rule=ObjectiveRule, sense=pyo.maximize)

    return m

###########################################################################
                        # HELPER FUNCTIONS
###########################################################################
# Pa to psig
def _pascal_to_psig(pressure):
    '''
    Function to convert pressure from units of pascal (Pa) to Pound-force per square inch - gauge (psig).

    Argument
    --------
    pressure: float, pressure in Pa

    Returns
    -------
    P_psig: float, pressure in psig

    Note
    ----
    This function assumes that the pressure is given in Pascals AND is an absolute pressure
    
    '''
    # define unit registryureg = UnitRegistry()
    ureg = UnitRegistry()
    ureg.default_format = '.2f'
   
    
    # define atmospheric pressure in psig
    one_atm = 1*ureg.atm
    one_atm.ito(ureg.psi)
    
    # define absolute pressure
    p_abs = pressure*ureg.pascal
    
    # convert absolute pressure to psig
    p_abs.ito(ureg.psi)
    
    # convert to psig: Ppsig = Pabs - 1atm
    p_psig = p_abs - one_atm
    return p_psig

###########################################################################
# psig to pascal
def _psig_to_pascal(P):
    '''
    Function to convert pressure units from psig to pascal.
    
    Argument
    --------
    P:float, pressure in psig
        
    Returns
    -------
    P_pascal: float, pressure in pascals
    '''
    # define unit registryureg = UnitRegistry()
    ureg = UnitRegistry()
    ureg.default_format = '.2f'
    
    # convert pressure units from psig to pascal
    # declare pressure in psig
    P_psig = P*ureg.psi
    
    # define atmospheric pressure in psig
    one_atm = 1*ureg.atm
    one_atm.ito(ureg.psi)
    
    # convert from guage absolute pressure: Pabs [psi] = Ppsig [psi] + 14.7 [psi]
    P_abs = P_psig + one_atm 
    
    # convert from psi to pascal
    P_pascal = P_abs.to(ureg.pascal)
    
    return P_pascal

###########################################################################

def toggle_reaction_off(m):
    """
    Function to deactiavte reaction from the membrane reactor model.

    Argument
    --------
    m: concrete pyomo model of membrane reactor

    """
    # deactivate reaction rate constraint in model
    m.reaction_rate_constraint.deactivate()

    # fix reaction rate, set to zero
    m.reaction_rate.fix(0)

#################################################################################

def toggle_reaction_on(m):
    """
    Function to activate reaction in membrane reactor model

    Argument
    --------
    m: concrete pyomo model

    """
    # unfix reaction rate
    m.reaction_rate.unfix()

    # activate reaction rate constraint
    m.reaction_rate_constraint.activate()

#################################################################################
# initialize reaction rate
def init_reaction_rate(m=None, ref_model=None):
    """
    Function to initialize reaction_rate variable using output from previous solve.

    Arguments
    ---------
    m: concrete Pyomo model
    ref_model: solved pyomo model used as base for initialization of model m. If not provided, m is initialized using temperature
        and composition values from itself
        
    Note
    ----
    This function is only useful when there is a solved results for the model to use as initialization point

    """
    # check reference model
    if not ref_model:
        ref_model = m.clone()
    
    # initialize reaction rate for model with distributed temperature
    if len(m.temp_retentate)>1:
        for j in m.ELEMENTS:
            m.reaction_rate[j] = (ref_model.reaction_scaling.value*
                0.92
                * math.exp(-454.3 / pyo.value(ref_model.temp_retentate[j]))
                * 1.03982e-10
                * pyo.value(ref_model.pressure_retentate[j]) ** 2
                * (
                    ref_model.composition_retentate[j, "CO"]
                    * ref_model.composition_retentate[j, "H2O"]
                    * (0.0126 * math.exp(4639 / pyo.value(ref_model.temp_retentate[j])))
                    - ref_model.composition_retentate[j, "CO2"] * ref_model.composition_retentate[j, "H2"]
                )
            ) / (
                (0.0126 * math.exp(4639 / pyo.value(ref_model.temp_retentate[j])))
                * (
                    1
                    + 1.0197e-5
                    * pyo.value(ref_model.pressure_retentate[j])
                    * (
                        2.2
                        * math.exp(101.5 / pyo.value(ref_model.temp_retentate[j]))
                        * m.composition_retentate[j, "CO"]
                        + 0.4
                        * math.exp(158.3 / pyo.value(ref_model.temp_retentate[j]))
                        * m.composition_retentate[j, "H2O"]
                        + 0.047
                        * math.exp(2737.9 / pyo.value(ref_model.temp_retentate[j]))
                        * ref_model.composition_retentate[j, "CO2"]
                        + 0.05
                        * math.exp(596.1 / pyo.value(ref_model.temp_retentate[j]))
                        * ref_model.composition_retentate[j, "H2"]
                    )
                )
                ** 2
            )
    # initialize reaction rate for model with uniform temperature
    else:
        for j in m.ELEMENTS:
            m.reaction_rate[j] = (ref_model.reaction_scaling.value*
                0.92
                * math.exp(-454.3 / pyo.value(ref_model.temp_retentate))
                * 1.03982e-10
                * pyo.value(ref_model.pressure_retentate[j]) ** 2
                * (
                    ref_model.composition_retentate[j, "CO"]
                    * ref_model.composition_retentate[j, "H2O"]
                    * (0.0126 * math.exp(4639 / pyo.value(ref_model.temp_retentate)))
                    - ref_model.composition_retentate[j, "CO2"] * ref_model.composition_retentate[j, "H2"]
                )
            ) / (
                (0.0126 * math.exp(4639 / pyo.value(ref_model.temp_retentate)))
                * (
                    1
                    + 1.0197e-5
                    * pyo.value(ref_model.pressure_retentate[j])
                    * (
                        2.2
                        * math.exp(101.5 / pyo.value(ref_model.temp_retentate))
                        * ref_model.composition_retentate[j, "CO"]
                        + 0.4
                        * math.exp(158.3 / pyo.value(ref_model.temp_retentate))
                        * ref_model.composition_retentate[j, "H2O"]
                        + 0.047
                        * math.exp(2737.9 / pyo.value(ref_model.temp_retentate))
                        * ref_model.composition_retentate[j, "CO2"]
                        + 0.05
                        * math.exp(596.1 / pyo.value(ref_model.temp_retentate))
                        * ref_model.composition_retentate[j, "H2"]
                    )
                )
                ** 2
            )

#################################################################################
def fix_temperature(m, temp=None, temp_list=[], LOUD=False):
    """
    Function to fix membrane temperature.
    
    Arguments
    ---------
    m: concrete Pyomo model, WGS-MR
    temp: float, temperature in Kelvin
    temp_list: list of temperature values, same size as # of elements in model (m)
    LOUD: bool, if true print
        
    Note
    ----
    Provide either temp or temp_list, NOT both
    """
    # check input
    if temp:
        T_list = [temp]*len(m.ELEMENTS)
    elif len(temp_list) == len(m.ELEMENTS):
        T_list = temp_list
    else:
        raise ("temperature not given; must provide either a single temperature value 'temp' or a list of  temperatures same size as number of elements in model 'temp_list'")
    if LOUD:
        print("Temperature list =\n",T_list)
    
    # fix temperature
    for e in m.ELEMENTS:
        m.temp_retentate[e].fix(T_list[e])
        
    return m

    
#################################################################################
def unfix_temperature(m):
    """
    Function to unfix temperature in the WGS-MR model.
    
    Argument
    --------
    m: concrete Pyomo model of the WGS-MR
    """
    for e in m.ELEMENTS:
        m.temp_retentate[e].unfix()

#################################################################################

def check_finite_element_mass_balance(m,side,print_level=1):
    """
    Check mass balances.
    
    Arguments
    ---------
    m: concrete Pyomo model, (solved) WGS-MR model
    side: str, either 'perm' or 'ret'
    print_level: int, <2 to print basic, >=2 to print detailed
        
    """
    # specify factor to convert model output flow rates from nondimensional to dimensional
    # empty dictionaries to hold component flow rates
    flow_feed_dict = {}
    flow_sweep_dict = {}
    if 'ghsv' in m.component_map(pyo.Var):
        P = 101325 # stanadard pressure (Pa)
        V = m.volume()*m.N # entire reactor volume
        R = 8.3145 # gas constant (J/mol/K)
        T = 273 # standard temperature (K)
        total_feed_flow =  P*V*pyo.value(m.ghsv)/(3600*R*T)
        total_sweep_flow = P*V*pyo.value(m.ghsv_sweep)/(3600*R*T)
        if print_level>2:
            print('total_feed_flow =',total_feed_flow)
            print('total_sweep_flow =',total_sweep_flow)
        # feed and sweep flows for each gas component 
        flow_feed_dict = {}
        flow_sweep_dict = {}
        for comp in m.y_feed.index_set():
            flow_feed_dict[comp] = pyo.value(m.y_feed[comp])*total_feed_flow
            flow_sweep_dict[comp] = pyo.value(m.y_sweep[comp])*total_sweep_flow
    else:
        total_feed_flow = pyo.value(m.total_feed)
        for comp in m.flow_feed.index_set():
            flow_feed_dict[comp] = pyo.value(m.flow_feed[comp])
            flow_sweep_dict[comp] = pyo.value(m.flow_sweep[comp])
                
    if print_level>2:
        print('feed flow dict: ',flow_feed_dict)
        print('\nSweep flow dict: ',flow_sweep_dict)
        
    # Define atoms
    atoms = ['H','C','O','N']
    # Populate matrix with zeros
    atom_matrix = {}
    for c in m.COMPONENTS:
        for a in atoms:
            atom_matrix[(c,a)] = 0

    # Fill in non-zero elements
    atom_matrix[('CO','C')] = 1
    atom_matrix[('CO','O')] = 1
    atom_matrix[('N2','N')] = 2
    atom_matrix[('CO2','C')] = 1
    atom_matrix[('CO2','O')] = 2
    atom_matrix[('H2O','O')] = 1
    atom_matrix[('H2O','H')] = 2
    atom_matrix[('CH4','H')] = 4
    atom_matrix[('CH4','C')] = 1
    atom_matrix[('H2','H')] = 2

    # Loop over finite elements
    for e in m.ELEMENTS:
        # Set empty dictionaries by component
        flow_in = {}
        flow_out = {}
        net_flux = {}
        print("Element",e)
        # Loop over atoms
        for c in m.COMPONENTS:

            # Retentate specific
            if side == 'ret':
                if e == m.ELEMENTS.first():
                    flow_in[c] = flow_feed_dict[c]
                else:
                    flow_in[c] = m.flow_retentate[e-1,c].value*total_feed_flow
                flow_out[c] = m.flow_retentate[e,c].value*total_feed_flow
                flux_sign = 1
            elif side == 'perm':
                if e == m.ELEMENTS.last():
                    flow_in[c] = flow_sweep_dict[c]
                else:
                    flow_in[c] = m.flow_permeate[e+1,c].value*total_feed_flow
                flow_out[c] = m.flow_permeate[e,c].value*total_feed_flow
                flux_sign = -1
            net_flux[c] = pyo.value(m.N_flux[e, c]) * m.area.value

            if print_level >=2:

                print("\tFlow out[",c,"]=",flow_in[c],"mol/s")
                print("\tFlow in[",c,"]=",flow_out[c],"mol/s")
                print("\tNet Flux[",c,"]=",net_flux[c],"mol/s")

        for a in atoms:
            # Compute residual by looping over components
            val = 0
            for c in m.COMPONENTS:
                # flow out - flow in + flux out
                val += atom_matrix[(c,a)]*(flow_out[c] - flow_in[c] + flux_sign*net_flux[c])
            # Print to screen
            print("Atom ",a," residual=",val)
        print(" ")

#################################################################################

def CO_conversion(m):
    '''Calculate CO conversion in a WGS-MR model, m.'''
    if 'ghsv' in m.component_map(pyo.Var):
        P = 101325 # stanadard pressure (Pa)
        V = m.volume()*m.N # entire reactor volume
        R = 8.3145 # gas constant (J/mol/K)
        T = 273 # standard temperature (K)
        total_feed =  P*V*pyo.value(m.ghsv)/(3600*R*T)
        CO_feed = pyo.value(m.y_feed["CO"])*total_feed
    else:
        total_feed = pyo.value(m.total_feed)
        CO_feed = pyo.value(m.flow_feed['CO'])
    return (1 - pyo.value(m.flow_retentate[m.ELEMENTS.last(),'CO'])*total_feed/CO_feed) * 100

################################################################################

def H2_recovery(m):
    '''calculate the H2 recovery in a WGS-MR model, m.
       NB: H2 recovery = H2 in permeate / (H2 in permeate + H2 in retentate).
    '''
    return pyo.value(m.flow_permeate[m.ELEMENTS.first(),'H2'])/(pyo.value(m.flow_retentate[m.ELEMENTS.last(),'H2']) 
                                                                + pyo.value(m.flow_permeate[m.ELEMENTS.first(),'H2'])) * 100

################################################################################

def feed_utilization_efficiency(m):
    '''Calculate the feed utilization efficiency for a WGS-MR model, m.
       NB: feed utilization efficiency (%) = 100* H2 in permeate / (H2 in feed + CO in feed + 4*CH4 in feed).
    '''
    # get the total feed flow rate (mol/s)
    if 'ghsv' in m.component_map(pyo.Var):
        P = 101325 # stanadard pressure (Pa)
        V = m.volume()*m.N # entire reactor volume
        R = 8.3145 # gas constant (J/mol/K)
        T = 273 # standard temperature (K)
        total_feed =  P*V*pyo.value(m.ghsv)/(3600*R*T)
    else:
        total_feed = pyo.value(m.total_feed)
    
    # calculate feed utiliation efficicency from theory
    H2_in_permeate = pyo.value(m.flow_permeate[m.ELEMENTS.first(), "H2"])*total_feed 

    return H2_in_permeate*100/((pyo.value(m.y_feed['H2']) + pyo.value(m.y_feed['CO']) 
                                + 4*pyo.value(m.y_feed['CH4']))*total_feed)
    


#################################################################################

def conc_profile_multiple(m, sweep=False, methane=True, save_plot=False):
    """
    Function to plot concentration profile of gas species along the length of the membrane module.

    Arguments
    ---------
    m: concrete model
    sweep: bool; If true, there is sweep, so plot concentration of hydrogen in the permeate
    methane: bool, if True add concentration profile for methane
    save_plot: bool; if True, save plot, if False, pass

    """
        
    # create container for compositions on both sides
    ret_comp_H2 = []
    if sweep:
        perm_comp_H2 = []
    ret_comp_CO2 = []
    ret_comp_CO = []
    ret_comp_H2O = []
    ret_comp_CH4 = []

    # loop through membrane module to get composition of each finite element
    for j in m.ELEMENTS:
        ret_comp_H2.append(100 * pyo.value(m.composition_retentate[j, "H2"]))
        if sweep:
            perm_comp_H2.append(100 * pyo.value(m.composition_permeate[j, "H2"]))
        # get CO2 compositions
        ret_comp_CO2.append(100 * pyo.value(m.composition_retentate[j, "CO2"]))
        # get CO compositions
        ret_comp_CO.append(100 * pyo.value(m.composition_retentate[j, "CO"]))
        # get H2O compositions
        ret_comp_H2O.append(100 * pyo.value(m.composition_retentate[j, "H2O"]))
        if methane:
            # get CH4 compositions
            ret_comp_CH4.append(100 * pyo.value(m.composition_retentate[j, "CH4"]))

    # plot concentration profile
    # configure plot characteristics
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.rcParams.update({"font.size": 10})
    plt.rcParams["axes.labelweight"] = "bold"

    # xvalues
    xvalues = np.linspace(0, 1, pyo.value(m.N))
    # plot H2
    plt.plot(xvalues, ret_comp_H2, "ro-", label="H$_{2}$")
    if sweep:
        plt.plot(xvalues, perm_comp_H2, "yo:", label="H$_{2}$ perm")
    # plot CO2
    plt.plot(xvalues, ret_comp_CO2, "kx-", label="CO$_{2}$")
    # plot CO
    plt.plot(xvalues, ret_comp_CO, "md-", label="CO")
    # plot H2O
    plt.plot(xvalues, ret_comp_H2O, "g>-", label="H$_{2}$O")
    if methane:
        # plot CH4
        plt.plot(xvalues, ret_comp_CH4, "b*-", label="CH$_{4}$")

    plt.xlabel("Dimensionless Membrane Length", fontsize=10)
    plt.ylabel("Composition (mol%)", fontsize=10)
    plt.legend(
        loc="lower center", fontsize=10, bbox_to_anchor=(0.5, 1), ncol=4
    )  # place legend at top of y-axis, middel of x-axis, 3 columns
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tick_params(direction="in",top=True, right=True)
    fig.tight_layout()
    
    # save plot
    if save_plot:
        plt.savefig("output/concentration_profile.png")
        plt.savefig("output/concentration_profile.pdf")

#################################################################################

def pressure_sensitivity(data,
    pressure_list=np.arange(3500,400,-100),
    temp_list=[625, 675, 725, 775, 825],
    equilibrium_conversion=None,
    PFR_conversion=None,
    with_reaction=True,
    industry_units=False,
    save_data=False,
    verbose=False,
):
    """
    Function to run sensitivity simulations with lists of pressure and temperature
    and save corresponding CO conversion data to a .csv file.

    Arguments
    ---------
    data: dict, 
        contains input data for model based on study case
    pressure_list: list, default np.arange(3500,400,-100)
        contain pressures for simulation
    temp_list: list, default [625, 675, 725, 775, 825]
        contain emperatures for simulation
    equilibrium_conversion: float, 
        equilibrium conversion of CO in the WGS reaction in %
    PFR_conversion: float, 
        CO conversion for plug flow reactor (PFR) in %
    with_reaction: bool, default False
        toggle between WGS-MR and separation-only membrane (no reaction)
    industry_units: bool, default False
        if True, use industry units (degrees celcius, psig) instead of SI units
    save_data: bool, default False
        if True, save data to .csv
    verbose: bool, default False
        if True, print sweep details

    Returns
    -------
    conversion_pd: Pandas dataframe, contains CO conversions for various combinations of temperature and pressure

    """
    # Unit conversions
    # setup unit conversion registry from Pint
    ureg = UnitRegistry()
    ureg.default_format = '.2f' # report units to 2 decimal places
    Q_ = ureg.Quantity # physical quantity generator
    
    
    # create an empty dictionaries to hold CO conversion and recovery data
    conversion_dict = {}
    recovery_dict = {}
    
    # loop through list of temperatures and run simulations
    for i, T in enumerate(temp_list):
        
        # create containers for pressure lists
        if i ==0:
            p_list = []
        # convert temperature units
        Tc = Q_(T,ureg.kelvin)
        Tc.ito('degC')

        # create empty list to hold CO conversion and H2 recovery
        conversion_list = []
        recovery_list = []
        
        # create and initialize model
        m = create_model(
            temp_retentate=T,
            feed_pressure=pressure_list[0] * 1000,
            CO_comp_feed=data['CO_comp_feed'],
            H2O_comp_feed=data['H2O_comp_feed'],
            CO2_comp_feed=data['CO2_comp_feed'],
            H2_comp_feed=data['H2_comp_feed'],
            CH4_comp_feed=data['CH4_comp_feed'],
            N2_comp_feed=data['N2_comp_feed'],
            feed_flow=data['feed_flow'],
            pressure_drop_retentate=data['pressure_drop_retentate'],
            sweep_pressure=data['sweep_pressure'],
            pre_exponent=data['pre_exponent'],
            E_R=data['E_R'],
            pressure_exponent=data['pressure_exponent'],
            vol_reactor=data['vol_reactor'],
            area_membrane=data['area_membrane'],
            rho_catalyst=data['rho_catalyst'],
            num_elements=20,
            with_reaction=with_reaction,
            discretize_temperature=False,
            max_h2=False
        )
        m.T.fix(T)
        # solve model
        if verbose:
            print("Beging initialization")
        # deactivate reaction
        toggle_reaction_off(m)
        # define solver
        model_solver = pyo.SolverFactory("ipopt")
        # solve model without reaction
        results = model_solver.solve(m, tee=True)
        # toggle reaction on
        toggle_reaction_on(m)
        # initialize reaction rate
        init_reaction_rate(m)
        if verbose:
            print("End initialization\n")
        # solve full model
        results = model_solver.solve(m, tee=True)

        # loop through list of pressures to run simulations
        for P in pressure_list:
            
            # add pressure to list
            if i == 0:
                p_list.append(P/1000)

            print("\nT = ",T,"P = ",P)
            
            # update model pressure
            m.pressure_feed = P*1000
            # solve model
            results = model_solver.solve(m, tee=True)
            if verbose:
                print("solver status: ",results.solver.termination_condition)
                
            # specify factor to convert model output flow rates from nondimensional to dimensional
            flow_factor = pyo.value(m.total_feed)

            # evaluate CO conversion and append to list
            conversion_list.append(
                100
                * (
                    1
                    - pyo.value(m.flow_retentate[pyo.value(m.ELEMENTS.last()), "CO"])*flow_factor
                    / pyo.value(m.flow_feed["CO"])
                )
            )
            
            # evaluate H2 recovery and append to list
            recovery_list.append(
                100
                * pyo.value(m.flow_permeate[m.ELEMENTS.first(), "H2"])
                / (
                    pyo.value(m.flow_permeate[m.ELEMENTS.first(), "H2"])
                    + pyo.value(m.flow_retentate[m.ELEMENTS.last(), "H2"])
                )
            )
                                   
        # add pressure lists to dictionary
        if i == 0:
            conversion_dict["Feed Pressure (MPa)"] = p_list
            recovery_dict["Feed Pressure (MPa)"] = p_list

        # add CO conversion and H2 recovery lists to dictionaries (key [=] T)
        # using industry units
        if industry_units:
            conversion_dict["{}°C".format(int(Tc.magnitude))] = conversion_list
            recovery_dict["{}°C".format(int(Tc.magnitude))] = recovery_list
        
        # using SI units
        else:
            conversion_dict["{}K".format(int(T))] = conversion_list
            recovery_dict["{}K".format(int(T))] = recovery_list
    
    # equilibrium conversion
    if equilibrium_conversion:
        eq_conversion_list = [equilibrium_conversion]*len(pressure_list)
        if industry_units:
            conversion_dict["Equilibrium"] = eq_conversion_list # "Equilibrium @ {}°C" .format(int(Tc.magnitude))
        else:
            conversion_dict["Equilibrium"] = eq_conversion_list
    # PFR conversion
    if PFR_conversion:
        PFR_conversion_list = [PFR_conversion]*len(pressure_list)
        if industry_units:
            conversion_dict["PFR"] = PFR_conversion_list # "Equilibrium @ {}°C" .format(int(Tc.magnitude))
        else:
            conversion_dict["PFR"] = PFR_conversion_list
            
    # convert conversion dictionary to a Pd
    conversion_pd = pd.DataFrame.from_dict(conversion_dict, orient="index")
    conversion_pd = conversion_pd.transpose()
    
    # convert recovery dictionary to a pd
    recovery_pd = pd.DataFrame.from_dict(recovery_dict, orient="index")
    recovery_pd = recovery_pd.transpose()

    # Save Pd to .csv file
    if save_data:
        # check date and time of simulation
        sim_date = date.today()
        # save data with date and time of simulation on it
        conversion_pd.to_csv("output/CO_conversion_simulation_output_{}.csv".format(sim_date))
        recovery_pd.to_csv("output/H2_recovery_simulation_output_{}.csv".format(sim_date))

    return conversion_pd, recovery_pd


################################################################################
def _hydrogen_per_feed(create_model=create_model,
                      T=723.15,
                      GHSV=3000,
                      vol_reactor=3.93e-5,
                       feed_pressure=3103000,
                      feed_composition={'CO':0.1,
                    'H2O':0.28,
                    'CO2':0.21,
                    'H2':0.35,
                    'CH4':0.06,
                    'N2':0},
                      molar_masses={'CO':28.01,
                    'H2O':18.015,
                    'CO2':44.01,
                    'H2':2.016,
                    'CH4':16.04,
                    'N2':28.014},
                     data_source='public',
                      dimensionless=True,
                      LOUD=False):
    '''
    Function to create run WGS-MR simulation and estimate H2 production per unit feed (kgH2/kgfeed).
    
    Arguments
    ---------
    create_model: callable function, 
        function to create an instance of a concrete Pyomo model of the WGS-MR
    T: float, default 723.15
        tempersture, in K
    GHSV: float, default 3000
        gas hourly space veolcity, in hr-1
    vol_reactor: float, default 3.93e-5
        reactor volume, in m3
    feed_composition: dict, default {'CO':0.1,'H2O':0.28,'CO2':0.21,'H2':0.35,'CH4':0.06,'N2':0}
        composition of feed gas, mole composition or volume composition (assume ideal gas)
        in mole fraction
    molar_masses: dict, default {'CO':28.01,'H2O':18.015,'CO2':44.01,'H2':2.016,'CH4':16.04,'N2':28.014}
        molar masses of gas components, in g/mol
    data_source: str, default 'public'
        specify dataset to use, options are 'pci' and 'public'
    dimensionless: bool, default True
        if True scale flow rate and membrane length
    LOUD: bool, default False
        if True, print to progress
        
        
    Returns
    -------
    H2_per_feed: H2 produced per unit feed, in kgH2/kgfeed
        
    Note
    ----
    This function is intended for use by other functions, but not for an end-user
    '''
    # print input
    if LOUD:
        print("\nT=",T," GHSV=",GHSV)
    # define constants
    R = 8.3145 # J/mol/K, universal gas constant
    Tstd = 294.25 # K, standard temperature, source:PCI
    Pstd = 101325 # Pa, standard pressure, source:PCI
    
    # calculate volumetric flow from GHSV
    volumetric_flow = GHSV*vol_reactor/3600 # divide by 3600 to convert from hr-1 to s-1
    # calculate molar flow from ideal gas law (Pv = nRT) at STP (25 C, 1 atm)
    feed = Pstd*volumetric_flow/(R*Tstd)
    
    # convert feed flow rate to kg basis
    mixture_MM = sum(feed_composition[comp]*molar_masses[comp] for comp in feed_composition) # feed gas MM
    kg_feed = feed*mixture_MM/1000 # feed gas mass fow rate, divide by 1000 to convert from g to kg
    
    # create reactor model, m = create_model()
    if data_source == "public":
        m = create_model(temp_reactor=T,
                         feed_flow=feed,
                        nondimensional=dimensionless)
    elif data_source == "pci":
        m = create_model(
            temp_reactor=T,
            CO_comp_feed=feed_composition['CO'],
            H2O_comp_feed=feed_composition['H2O'],
            CO2_comp_feed=feed_composition['CO2'],
            H2_comp_feed=feed_composition['H2'],
            CH4_comp_feed=feed_composition['CH4'],
            N2_comp_feed=feed_composition['N2'],
            feed_flow=feed,
            feed_pressure=feed_pressure,
            pressure_drop_retentate=35000,
            CO_comp_sweep=0,
            H2O_comp_sweep=0,
            CO2_comp_sweep=0,
            H2_comp_sweep=0,
            CH4_comp_sweep=0,
            N2_comp_sweep=1,
            sweep_flow=0,
            sweep_pressure=101325,
            pressure_drop_permeate=0,
            pre_exponent=5.2e-3,
            E_R=748.98,
            pressure_exponent=0.5,
            vol_reactor=vol_reactor,
            area_membrane=2.71e-3,
            rho_catalyst=800.6,
            num_elements=20,
            with_reaction=True,
            nondimensional=dimensionless
        )
    # solve model
    model_solver = pyo.SolverFactory("ipopt")
    # deactivate reaction
    toggle_reaction_off(m)
    results = model_solver.solve(m, tee=False)
    results = model_solver.solve(m, tee=False)
    # toggle reaction on
    toggle_reaction_on(m)
    # initialize reaction rate
    init_reaction_rate(m)
    results = model_solver.solve(m, tee=False)
    
    # save solve status
    if results.solver.termination_condition == 'optimal':
        solve_status = 1
    else:
        solve_status = 0
    if LOUD:
        print("Solve status: ",solve_status)
    
    # conversion for dimensionless flow
    if dimensionless:
        flow_factor = pyo.value(m.total_feed)
    else:
        flow_factor = 1
    
    # extract H2 product and convert to kg basis
    mole_H2 = pyo.value(m.flow_permeate[m.ELEMENTS.first(), "H2"])*flow_factor
    kg_H2 = mole_H2*molar_masses['H2']/1000 # H2 product mass fow rate [kg/s], divide by 1000 to convert from g to kg
    # calculate H2 product per unit feed
    H2_per_feed = kg_H2/kg_feed
    
    return H2_per_feed,solve_status


################################################################################

def sweepflow_sensitivity(data,
                            sweep_list=np.linspace(0,1,4),
                            temp_list=[624,674,724],
                            feed_pressure=200,
                            industry_units=False,
                            LOUD=False,
                            save_data=False):
    '''
    Function to run WGS-MR simulations for various sweep rates at 
    given temperatures and pressures, and return the corresponding CO conversion data.
    
    Arguments
    ---------
    data: dict, 
        model input data
    sweep_list: list, default np.linspace(0,1,4)
        contains sweep flow rates 
    temp_list: list, default [624,674,724]
        contains temperature values
    feed_pressure: float, default 200
        feed pressure in psig
    LOUD: bool, default False
        if True, print progress during iterations
    save_data: bool, default False
        if True, save data to .csv 
        
    Returns
    -------
    conversion_pd: pandas dataFrame
        
    '''
    # Unit conversions
    # setup unit conversion registry from Pint
    ureg = UnitRegistry()
    ureg.default_format = '.2f' # report units to 2 decimal places
    Q_ = ureg.Quantity # physical quantity generator
    
    # convert pressure to pascal
    P_pascal = _psig_to_pascal(feed_pressure)
    if LOUD:
        print('P =',P_pascal,'Pa')
    
    # specify feed flow
    feed = data['feed_flow']
    
    # create conatiner dictionary
    conversion = {}
    recovery = {}
    
    # add sweeplist to dictionary
    conversion["Sweep to Feed Ratio"] = sweep_list
    recovery["Sweep to Feed Ratio"] = sweep_list
                          
    # loop through temperature
    for T in temp_list:
        # create container list
        conversion_list = []
        recovery_list = []
        # convert temperature units from K to C
        Tc = Q_(T,ureg.kelvin)
        Tc.ito('degC')
        
        # create model
        s = 0  # first solve at zero sweep
        m = create_model(
            temp_retentate=T,
            feed_pressure=P_pascal.magnitude,
            feed_flow=feed,
            sweep_flow=s*feed,
            CO_comp_feed=data['CO_comp_feed'],
            H2O_comp_feed=data['H2O_comp_feed'],
            CO2_comp_feed=data['CO2_comp_feed'],
            H2_comp_feed=data['H2_comp_feed'],
            CH4_comp_feed=data['CH4_comp_feed'],
            N2_comp_feed=data['N2_comp_feed'],
            pressure_drop_retentate=data['pressure_drop_retentate'],
            sweep_pressure=data['sweep_pressure'],
            pre_exponent=data['pre_exponent'],
            E_R=data['E_R'],
            pressure_exponent=data['pressure_exponent'],
            vol_reactor=data['vol_reactor'],
            area_membrane=data['area_membrane'],
            rho_catalyst=data['rho_catalyst'],
            discretize_temperature=False,
            num_elements=20,
            with_reaction=True,
        )
        # solve model
        # fix temperature for simulation
        m.T.fix(T)
        # deactivate reaction
        toggle_reaction_off(m)
        # define solver
        model_solver = pyo.SolverFactory("ipopt")
        # solve model without reaction
        results = model_solver.solve(m, tee=True)
        # resolve model
        results = model_solver.solve(m, tee=True)
        # toggle reaction on
        toggle_reaction_on(m)
        # initialize reaction rate
        init_reaction_rate(m)
        # solve full model
        results = model_solver.solve(m, tee=True)


        
        # loop through sweep list
        for s in sweep_list:
            # print
            print('Sweep/feed ratio =',s,'T =',T)
            # update sweep
            sweep_total = s*sum(m.flow_feed[i].value for i in m.COMPONENTS) # calculate total feed from components
            for i in m.COMPONENTS:
                if i == 'N2':
                    m.flow_sweep[i] = sweep_total # Specify sweep as 100% Nitrogen
                else:
                    m.flow_sweep[i] = 0
            if LOUD:
                print('total sweep =',sweep_total)
                for i in m.COMPONENTS:
                    print('\n',i,':',m.flow_sweep[i].value)
            # re-solve model
            results = model_solver.solve(m,tee=True)
            
            # specify factor to convert model output flow rates from nondimensional to dimensional
            flow_factor = pyo.value(m.total_feed)
            
            # evaluate CO conversion and append to list
            conversion_list.append(
                100
                * (
                    1
                    - pyo.value(m.flow_retentate[pyo.value(m.ELEMENTS.last()), "CO"])*flow_factor
                    / pyo.value(m.flow_feed["CO"])
                )
            )
            # evaluate H2 recovery and append to list
            recovery_list.append(
                100
                * pyo.value(m.flow_permeate[m.ELEMENTS.first(), "H2"])
                / (
                    pyo.value(m.flow_permeate[m.ELEMENTS.first(), "H2"])
                    + pyo.value(m.flow_retentate[m.ELEMENTS.last(), "H2"])
                )
            )
        # add list to dictionary
        if industry_units:
            conversion["{}°C".format(int(Tc.magnitude))] = conversion_list
            recovery["{}°C".format(int(Tc.magnitude))] = recovery_list
        else:
            conversion["{}K".format(T)] = conversion_list
            recovery["{}K".format(T)] = recovery_list
    
    # convert dictionary to pandas dataFrame
    conversion_pd = pd.DataFrame.from_dict(conversion, orient = "index")
    conversion_pd = conversion_pd.transpose()
    recovery_pd = pd.DataFrame.from_dict(recovery, orient="index")
    recovery_pd = recovery_pd.transpose()
    
    # if save option True, save data to csv
    if save_data:
        conversion_pd.to_csv("output/conversion_vs_sweep_data.csv")
        recovery_pd.to_csv("output/recovery_vs_sweep_data.csv")
    
    # return dataFrame
    return conversion_pd, recovery_pd
                

################################################################################

def plot_pd(file_path=None, data=None, column=None, y_label=None, x_label=None, y_lim=[], txt=None, txt_loc=None, line_styles = ["rx-","k*-","b+-","mv-","go-","yp-"], legend_loc=(0.5, 1), show_legend=True, legend_col=3, y_ticks=None, x_ticks=None, save_fig=False):
    '''
    Function to plot pandas dataframe.
    
    Arguments
    ---------
    file_path: str, 
        path to csv file containing data
    data: Pandas dataftrame, 
        conversion vs sweep data to be plotted
    column: str, 
        specify column of data to plot (for y-axis), x-axis data is default (first column in dataframe)
    y_label: str, optional
        label for y-axis
    x_label: str, optional 
        specify label for x-axis
    y_lim: list, optional 
        limits for y-axis
    txt: str, 
        text to print inside plot, e.g key simulatin paraters
    txt_loc: list, 
        specify [x,y] location to begin text,
    show_legend: bool, default True
        if True show legend
    line_styles: list, default ["rx-","k*-","b+-","mv-","go-","yp-"]
        list of line styles
    legend_loc: tuple, default (0.5, 1),
        coordinates for legend location, specify as fraction
    legend_col: int, default 3
        number of columns for legend
    y_ticks: list, 
        specify yticks
    x_ticks: list, 
        specify xticks
    save_fig: bool, default False
        if True, save figure
    
    Returns
    -------
    fig: plot object
        
    Notes
    -----
    This function assumes that the first column in the pandas dataframe provided contains x values
        
    '''
    # load data
    if file_path:
        data_pd = pd.read_csv(file_path)
    elif not data.empty:
        data_pd = data
    else:
        raise ValueError("data input missing, provide file_path (path to .csv file) or data_pd (Pandas dataframe)")
        
    # clean data: if pd has an arbitrary unnamed column, drop!
    if "Unnamed: 0" in data_pd.columns:
        data_pd.drop(columns="Unnamed: 0", inplace=True)
    
    # set matplotlib parameters
    fig = plt.figure(figsize=(4,3),dpi=200)
    plt.rcParams.update({"font.size": 10, "axes.labelweight":"bold"})
    
    # plot specific column or all columns
    if column:
        plt.plot(data_pd[data_pd.columns[0]],data_pd[column],line_styles[0],markersize=4)
        plt.xlabel(data_pd.columns[0])
    else:  
        # loop through dataframe columns to plot conversions
        for i,col in enumerate(data_pd.columns):
            if i !=0: # drop first column (column of x values)
                plt.plot(data_pd[data_pd.columns[0]],data_pd[col], line_styles[i-1], label=col, markersize=4)
                plt.xlabel(data_pd.columns[0])
    if y_label:
        plt.ylabel(y_label)
        
    if x_label:
        plt.xlabel(x_label)
    # specify yticks
    if not y_ticks==[]:
        plt.yticks(y_ticks)
    
    # specify xticks
    if not x_ticks==[]:
        plt.xticks(x_ticks)
    
    # y limits
    if not y_lim == []:
        plt.ylim(y_lim[0],y_lim[1])
        
    if show_legend:
        plt.legend(loc="lower center", fontsize=10, bbox_to_anchor=legend_loc, ncol=legend_col)
    # add text to plot
    if not txt is None:
        plt.text(txt_loc[0], txt_loc[1], txt, fontsize=8)
    fig.tight_layout()
    
    
    # save plot to png
    if save_fig:
        # name plot
        if not y_label:
            plot_name = "plot" + data_pd.columns[0]
        plot_name = y_label + " vs " + data_pd.columns[0] + " plot"
        
        # save plot
        plt.savefig('output/'+plot_name + ".png",dpi=300,bbox_inches='tight')
        plt.savefig('output/'+plot_name + ".pdf",dpi=300,bbox_inches='tight')
    return fig
    
################################################################################
    
def print_streams(m,stream='retentate',save_data=False):
    '''
    Function to print and save retentate stream flow rate and compoitions.
    
    Arguments
    ---------
    m: concrete model
        pyomo model of the WGS-MR
    stream: str, default "retentate"
        choose between retentate ("retentate") and permeate ("permeate") streams
    save_data: bool, default False
        if True, save data
    
    Returns
    -------
    stream_flow_pd:pd dataframe
    
    '''
    # get stream flows for retentate side
    if stream == 'retentate':
        # empty dictionary to hold streams
        retentate_streams_dict = {}
        # loop through components and obtian stream flow rates and composition
        for i in m.COMPONENTS:
            # add component flow rates for each module element 
            retentate_streams_dict["{}_flow [mol/s]".format(i)] = [pyo.value(m.flow_retentate[j,i]) for j in m.ELEMENTS]
            # add component's composition for each module element
            retentate_streams_dict["{}_comp [mol/mol]".format(i)] = [pyo.value(m.composition_retentate[j,i]) for j in m.ELEMENTS]
        # convert dictionary into a pandas data frame
        stream_flow_pd = pd.DataFrame.from_dict(retentate_streams_dict, orient='index')

    # get stream flows for permete side
    elif stream == 'permeate':
        # empty dictionary to hold streams
        permeate_streams_dict = {}
        # loop through components and obtian stream flow rates and composition
        for i in m.COMPONENTS:
            # add component flow rates for each module element 
            permeate_streams_dict["{}_flow [mol/s]".format(i)] = [pyo.value(m.flow_permeate[j,i]) for j in m.ELEMENTS]
            # add component's composition for each module element
            permeate_streams_dict["{}_comp [mol/mol]".format(i)] = [pyo.value(m.composition_permeate[j,i]) for j in m.ELEMENTS]
        # convert dictionary into a pandas data frame
        stream_flow_pd = pd.DataFrame.from_dict(permeate_streams_dict, orient='index')
    
    # raise error if stream side is not properly specified
    else:
        raise ValueError("stream must be either 'retentate' or 'permeate'")
        
    if save_data:
        stream_flow_pd.to_csv("{}streams_data.csv".format(stream))
    
    return stream_flow_pd