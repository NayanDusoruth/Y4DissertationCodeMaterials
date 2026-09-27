# =========================================================
# code block 1
# =========================================================
using LinearAlgebra
using Plots
using ColorSchemes

import Pkg; 
Pkg.add("ColorSchemes")
startTime = time()

println("starting!")
# =========================================================
# code block 2
# =========================================================
function f()
    

    "Spectral function of choice"
    J(omega; prefactor=0.005, D=1.0) = prefactor .* sqrt.(1.0 .- (omega ./ D) .^ 2)

    n = 300 # an arbitrary choice for now, not too large
    N = n + 1

    H = zeros(ComplexF64, N, N)
    epsilon = 0.0 # our scale of energies

    D = 1.0 # choice of the bandwidth
    omegas = collect(range(-D, D, length=n)) # discretisation of the bath
    Js = J(omegas; D=D)

    # first column (index 1 is the system, 2:end is the bath)
    # scaled for correct thermodynamic limit, since the rates depend on j_i^2
    H[2:end, 1] .= sqrt.(Js) ./ sqrt(n)
    # first row
    H[1, 2:end] .= sqrt.(Js) ./ sqrt(n)
    # diagonal
    H[diagind(H)] .= [epsilon; omegas]
    # println(H)

    # plot the spectral function and its discretisation
    """_x = range(minimum(omegas), maximum(omegas), length=1000)
    plot(_x, J(_x), label="J(ω)")
    scatter!(omegas, Js, label="discretised")
    xlabel!("ω")
    ylabel!("J(ω)")""" # TODO: Uncomment plotting as necessary


    # =========================================================
    # code block 3
    # =========================================================

    C0 = zeros(ComplexF64, N, N)
    # choosing the temperature
    beta = 1.0
    mu = 0.0
    fermi(x) = 1 ./ (1 .+ exp.(beta .* x .- beta .* mu))

    density_0 = 0
    C0[diagind(C0)] .= [0.0; fermi.(omegas)]

    C0

    # =========================================================
    # code block 4
    # =========================================================

    # Step 1: Diagonalize H
    e_vals, U = eigen(Hermitian(H))  # H = U * Diagonal(e_vals) * U'

    # Step 2: Transform C0 into the energy eigenbasis
    C_energy_basis = U' * C0 * U

    # Step 3: Check diagonality
    off_diagonal_norm = norm(C_energy_basis - Diagonal(diag(C_energy_basis)))
    #println("Off-diagonal norm of C in energy basis: ", off_diagonal_norm)

    # =========================================================
    # code block 5
    # =========================================================

    """
    p1 = heatmap(log10.(abs.(C0) .+ 1e-6), yflip=true, color=:inferno, title="C in the original basis")
    p2 = heatmap(log10.(abs.(C_energy_basis) .+ 1e-6), yflip=true, color=:inferno, title="C in the energy basis")
    plot(p1, p2, layout=(1, 2), size=(1000, 450))
    """ # TODO: Uncomment plotting as necessary

    # =========================================================
    # code block 6
    # =========================================================

    unitary(t) = exp(-1im * H * t)

    # =========================================================
    # code block 7
    # =========================================================

    function evolve(C, t)
        U = unitary(t)
        return U * C * U'
    end
    # =========================================================
    # code block 8
    # =========================================================

    dt = 0.1
    tmax = 1000.0
    times = collect(0:dt:tmax-dt) # mirrors np.arange(0, tmax, dt)

    # =========================================================
    # code block 9
    # =========================================================

    # supposedly faster iterative calculation
    time1 = time()
    Udt = unitary(dt)
    UdtT = Udt'
    Css = Dict(0.0 => C0)
    #println(typeof(C0))
    #CssArr = Array{Matrix, 1}(undef, size(times)[1])
    #println(size(CssArr))
    #CssArr[1] = C0

    
    #print(size(times)[1])
    for i=2:size(times)[1]
        Css[times[i]] = Udt * Css[times[i-1]] * UdtT
        #CssArr[i]=Udt * CssArr[i-1] * UdtT
    end
    time2 = time()
    timeDiffGlob = time2-time1
    print("timeglob ",timeDiffGlob)
    #print(size(CssArr))
    # =========================================================
    # code block 10
    # =========================================================

    density = [Css[t][1, 1] for t in times]

    # =========================================================
    # code block 11
    # =========================================================
    """
    plot(times, real.(density), xlabel="Time", ylabel="Occupation on the system", legend=false)
    """# TODO: Uncomment plotting as necessary
    # =========================================================
    # code block 12
    # =========================================================
    """
    plt = plot(legend=false) # TODO: Uncomment plotting as necessary
    for t in times
        val = Css[t]
        c = get(ColorSchemes.coolwarm, t / tmax)
        plot!(plt, omegas, real.(diag(val[2:end, 2:end])), color=c) # TODO: Uncomment plotting as necessary
    end

    u = range(-1.2, 1.2, length=50)
    scatter!(plt, u, fermi.(u), color=:black, markersize=2, markerstrokewidth=0)
    xlabel!("ω")
    ylabel!("Occupation")
    plt
    """ # TODO: Uncomment plotting as necessary

    
end

f()

endTime = time()
timeDiff = endTime - startTime
println("all done! time elapsed: " * string(timeDiff))