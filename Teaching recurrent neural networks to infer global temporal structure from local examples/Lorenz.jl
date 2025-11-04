using Printf

mutable struct Lorenz
    x0::Vector{Float64}
    x::Vector{Float64}
    delT::Float64
    parms::Vector{Float64}

    # 생성자
    function Lorenz(x0, delT, parms)
        new(x0, copy(x0), delT, parms)
    end
end

# Lorenz 시스템의 ODE
function lorenz_dxdt(x, p)
    return [
        p[1] * (x[2] - x[1]),
        x[1] * (p[2] - x[3]) - x[2],
        x[1] * x[2] - p[3] * x[3]
    ]
end

function propagate!(o::Lorenz, n::Int)
    X = zeros(3, n, 4)
    X[:, 1, 1] = o.x
    
    dxdt = (x) -> lorenz_dxdt(x, o.parms)
    
    @printf("%s\n", "."^100)
    nInd = 0.0
    
    for i = 2:n
        if i > nInd * n
            @printf("=")
            nInd += 0.01
        end
        
        k1 = o.delT * dxdt(o.x)
        k2 = o.delT * dxdt(o.x .+ k1/2)
        k3 = o.delT * dxdt(o.x .+ k2/2)
        k4 = o.delT * dxdt(o.x .+ k3)
        
        X[:, i, 1] = o.x .+ (k1 .+ 2*k2 .+ 2*k3 .+ k4) / 6
        X[:, i-1, 2] = o.x .+ k1/2
        X[:, i-1, 3] = o.x .+ k2/2
        X[:, i-1, 4] = o.x .+ k3
        
        o.x = X[:, i, 1]
    end
    
    @printf("\n")
    return X
end