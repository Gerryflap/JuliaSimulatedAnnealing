module SortingExample
    include("SimulatedAnnealing.jl")
    using Statistics
    using Plots
    gr()

    struct SortState <: SimulatedAnnealing.State
        arr :: Array{Float64}
    end

    # Function that gets the faction of the time used so far and returns a temperature. Should be 0 at the end
    function temperature_fn(budget_used::Float64) :: Float64
        return 1-budget_used
    end

    # Computes energy based on number of element pairs that is in incorrect order
    function energy_fn(sortState::SortState) :: Float64
        wrong :: Float64 = 0
        for i in 2:length(sortState.arr)
            if sortState.arr[i-1] > sortState.arr[i]
                wrong += 1
            end
        end
        return wrong/length(sortState.arr)
    end

    # Function that computes the energy of the given State based on 
    # the sum of how much larger the max of all elements on the left and how much smaller the min of all elements on the right is
    # This is done over a normalized array
    function minmax_based_energy_fn(sortState::SortState) :: Float64
        normalized = normalize(sortState.arr)

        energy = 0

        max_left = normalized[1]
        for i in 2:length(normalized)
            diff = (max_left - normalized[i])
            if diff > 0
                energy += diff
            else
                max_left = normalized[i]
            end            
        end

        max_right = normalized[length(normalized)]
        for i in (length(normalized) - 1):-1:1
            diff = (normalized[i] - max_right)
            if diff > 0
                energy += diff
            else
                max_right = normalized[i]
            end      
        end

        return energy
    end

    # Normalizes the given array to 0 mean and 1 standard deviation
    function normalize(arr::Array{Float64}) :: Array{Float64}
        return (arr.-mean(arr))./std(arr)
    end


    # Function that generates a random neighbour state given the current state by randomly swapping 2 neighbouring elements
    function neighbour_fn(sortState::SortState)
        copied = copy(sortState.arr)
        # Random number between 1 and length - 1
        index = 1 + abs(rand(Int32) % (length(copied) - 1))

        # Swap elements
        temp = copied[index]
        copied[index] = copied[index+1]
        copied[index+1] = temp

        return SortState(copied)
    end

    # Function that generates a random neighbour state given the current state by randomly swapping 2 random elements
    function global_neighbour_fn(sortState::SortState)
        copied = copy(sortState.arr)
        # Random number between 1 and length - 1
        index_1 = 1 + abs(rand(Int32) % (length(copied)))
        index_2 = 1 + abs(rand(Int32) % (length(copied)))

        # Swap elements
        temp = copied[index_1]
        copied[index_1] = copied[index_2]
        copied[index_2] = temp

        return SortState(copied)
    end

    settings = SimulatedAnnealing.Settings(
        temperature_fn,
        minmax_based_energy_fn,
        global_neighbour_fn,
        10000
    )

    initial = SortState([5 7 1 2 0 -1 8 -10 5 2 7 9 2 5 1 2 7 10 4 1])
    result = SimulatedAnnealing.run_simulated_annealing(initial, settings; collect_energy_vals=true)

    println(result.s)
    println("Energy: ", result.e)

    plt = plot(
        result.energies,
        title="Energy of current state per step",
        xlabel="step",
        ylabel="energy",
        legend=false
    )
    display(plt)
end