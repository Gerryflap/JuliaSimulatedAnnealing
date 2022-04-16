#=
    Small example of how to use the SimulatedAnnealing package.
    These comments will explain step-by-step what I'm doing and why.

    The example problem is sorting an array of numbers
=#
module SortingExample
    include("SimulatedAnnealing.jl")
    using Statistics
    using Plots
    gr()

    # First we have to define a struct that will describe our state. In our case it'll only contain the current state of the array
    # Notice that the new SortState struct is a subtype of SimulatedAnnealing.State.
    struct SortState <: SimulatedAnnealing.State
        arr :: Array{Float64}
    end

    #= 
        Next up we define a function for the temperature, which controls how much the algorithm will switch to less optimal states.
        A temperature of 0 means that the algorithm will only switch to better states, a higher temperature will increase this probability.

        In this case I'll simply let it go from 1 -> 0 over the course of training, but this is very dependent on your problem and energy function.
        It does not strictly have to go from 1 to 0, it can also go from 100 to 0 for example. My chosen function is linear, but this also is not required.
        Do make sure that it always ends at 0 and doesn't ever go below 0.
    =#
    function temperature_fn(budget_used::Float64) :: Float64
        return 1-budget_used
    end

    #=
        The energy function basically computes how "wrong" the current state is.
        Simulated annealing will slowly "freeze" into a low energy state.
        Your choice of energy function is very important. It should provide a clear indication of how bad a current state is (or isn't).

        This is an example of a very simple (and naive) energy function.
        It will iterate over the array and count the out of order pairs, pairs where the left element is larger than the right one.
    =#
    function energy_fn(sortState::SortState) :: Float64
        wrong :: Float64 = 0
        for i in 2:length(sortState.arr)
            if sortState.arr[i-1] > sortState.arr[i]
                wrong += 1
            end
        end
        return wrong
    end

    #=
        The previous energy function does not often lead to convergence
        While the best state (fully sorted) does have the lowest energy, 
            the function considers a lot of states as equivalent that really shouldn't be.
        Some examples:
                - [0 1 2 3 4 5 -1] and [0 -1 1 2 3 4 5] both have 1 wrong pair,
                    but the latter is almost in order while in the former the -1 is as far from it's preferred location as it can be.
                - [0 1 -10] and [-10 1 0] both have 1 wrong pair, but arguably the latter is way more acceptable than the former because 1 and 0 only differ by 1.
        
        The following function tries to resolve this a bit. Here the energy is increased when an element is smaller than the largest element to the left 
            and/or larger than the smallest element on the right. 
            The increase in energy is proportional to these differences, if the max value on the left is 100 and the current element is 1 then this will add 99 energy.
        
        Using this function will often lead to full convergence.
        One of the hard parts about tuning simulated annealing is that this may not always be the result of the energy function being better,
            it may also be a result of the scale of the energy function. 
        Always keep in mind how temperature and energy interact before drawing conclusions. 
        The fact that this function works better might just be a scale difference :p
    =#
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


    #=
        Similar to how the energy function influences convergence, the neighbour function is also very important.
        It should generate a random neighbour of the current state, meaning that it should be reasonably close.
        At the same time it should also be relatively easy to reach a desirable state. 

        The following neighbour function swaps 2 random elements in the array and returns the modified array as the new neighbour state.
        This is not an in-place change though, since we want the current state to remain untouched.
    =#
    function neighbour_fn(sortState::SortState)
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

    # Here we initialize the settings by giving the functions we just made and a the number of steps we want to run for
    settings = SimulatedAnnealing.Settings(
        temperature_fn,
        minmax_based_energy_fn,
        neighbour_fn,
        10000
    )

    # Then we implement an initial state
    initial = SortState([5 7 1 2 0 -1 8 -10 5 2 7 9 2 5 1 2 7 10 4 1])

    # And then we run the actual algorithm
    result = SimulatedAnnealing.run_simulated_annealing(initial, settings; collect_energy_vals=true)

    # Print final state and energy
    println(result.s)
    println("Energy: ", result.e)

    # And finally show a plot of the energy during training
    plt = plot(
        result.energies,
        title="Energy of current state per step",
        xlabel="step",
        ylabel="energy",
        legend=false
    )
    display(plt)
end