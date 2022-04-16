module SimulatedAnnealing
    export State, SimState, Settings, annealing_step, run_simulated_annealing

    # Models the current state
    abstract type State end

    # Contains relevant information for the current simulator state
    mutable struct SimState
        # Current state
        s::State
        # Current step
        step::Int64
        # Energy of current state (to avoid re-evaluation)
        e_current::Float64
    end

    # Contains the result of a run
    struct Result
        # Final state
        s::State
        # Final energy
        e::Float64
        # Either nothing, or an array of energy values per step
        energies::Union{Array{Float64}, Nothing}
    end

    struct Settings
        # Function that gets the faction of the time used so far and returns a temperature. Should be 0 at the end
        temperature_fn
        # Function that computes the energy of the given State. Lower is better
        energy_fn
        # Function that generates a random neighbour state given the current state
        neighbour_fn
        # Max steps taken by the algorithm
        max_steps::Int64
    end

    # Computes the acceptance probability of the new state given the current and new energy values and the temperature
    function p(e_current, e_new, temperature) :: Float64
        if e_new < e_current
            return 1
        else
            return exp(-(e_new - e_current)/temperature)
        end
    end

    # Performs one step of the simulated annealing algorithm (in-place on the SimState)
    function annealing_step(simstate::SimState, settings::Settings)
        s_new = settings.neighbour_fn(simstate.s)
        simstate.step += 1
        temperature = settings.temperature_fn(simstate.step / settings.max_steps)

        e_current = simstate.e_current
        e_new = settings.energy_fn(s_new)
        
        acceptance_probability = p(e_current, e_new, temperature)
        if rand() <= acceptance_probability
            simstate.s = s_new
            simstate.e_current = e_new
        end
    end

    # Runs simulated annealing from s_initial with the given settings. 
    # If collect_energy_vals is set to true, the Results.energies array will be filled with the energy for each step.
    function run_simulated_annealing(s_initial::State, settings::Settings; collect_energy_vals::Bool=false) :: Result
        simstate = SimState(s_initial, 0, settings.energy_fn(s_initial))

        energy_vals = nothing
        if collect_energy_vals
            # Collect results
            energy_vals = zeros(Float64, settings.max_steps + 1)
            energy_vals[1] = simstate.e_current

            while simstate.step < settings.max_steps
                annealing_step(simstate, settings)
                energy_vals[simstate.step + 1] = simstate.e_current
            end
        else 
            # Just execute without collecting
            while simstate.step < settings.max_steps
                annealing_step(simstate, settings)
            end
        end

        return Result(simstate.s, simstate.e_current, energy_vals)
    end
end # module
