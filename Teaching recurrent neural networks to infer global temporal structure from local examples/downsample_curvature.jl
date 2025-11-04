using LinearAlgebra

function downsample_curvature(X_in, a, v=nothing)
    X = copy(X_in)
    XS = copy(X_in) # v가 있을 경우 원본 저장
    
    if !isnothing(v)
        # Get view vector
        vx = [ sind(v[1])*cosd(v[2]);
              -cosd(v[1])*cosd(v[2]);
               sind(v[2]) ]
        vyz = nullspace(vx') # MATLAB null(vx') -> Julia nullspace(vx')
        
        # Project time series onto 2D plane
        X = vyz' * X
    end

    VInd = [1, 1] # 루프 진입을 위해 2개 요소
    n_start = size(X, 2)
    aInd = 1:n_start
    dInd = Int[] # 제거된 인덱스

    while length(VInd) > 1
        # Initialize
        V = diff(X, dims=2) # MATLAB diff(X,1,2) -> Julia diff(X, dims=2)
        
        # MATLAB: VMag = sum(V(:,2:end) .* V(:,1:end-1))
        # Julia: sum(..., dims=1)로 열별 합계를 벡터로 만듦
        VMag_mat = sum(V[:, 2:end] .* V[:, 1:end-1], dims=1)
        VNorm_mat = sqrt.(sum(V[:, 1:end-1].^2, dims=1))
        VMag = VMag_mat ./ VNorm_mat
        
        VNorm = [a+1; sqrt.(sum(V[:, 2:end].^2, dims=1)' .- VMag'.^2)]'
        
        # Find indices
        VIndL = findall(x -> x < a, VNorm)
        VIndU = findall(x -> x > a, VNorm)
        
        # Remove indices around points
        VIndL = setdiff(VIndL, VIndU .+ 1, VIndU .- 1)
        if isempty(VIndL); break; end
        
        # Search for consecutive points
        VIndD = [2; diff(VIndL)]
        VIndNC = VIndL[VIndD .!= 1] # Nonconsecutive
        VIndC = VIndL[VIndD .== 1]  # Consecutive
        
        # Putative points to remove
        VIndR = [VIndNC; VIndC[1:2:end]] # Remove every other consecutive
        VIndK = setdiff(1:size(X, 2), VIndR)
        
        # Curvature of points to keep
        XP = X[:, VIndK]
        Vp = diff(XP, dims=2)
        VMag_mat_p = sum(Vp[:, 2:end] .* Vp[:, 1:end-1], dims=1)
        VNorm_mat_p = sqrt.(sum(Vp[:, 1:end-1].^2, dims=1))
        VMag_p = VMag_mat_p ./ VNorm_mat_p
        
        VNorm_p = [a; sqrt.(sum(Vp[:, 2:end].^2, dims=1)' .- VMag_p'.^2)]'
        VIndKO = VIndK[VNorm_p .> a]
        
        # Remove bad points
        VIndR = setdiff(VIndR, VIndKO .+ 1, VIndKO .- 1)
        if isempty(VIndR); break; end
        
        # Keep track of removed points
        append!(dInd, aInd[VIndR])
        dr = setdiff(1:size(X, 2), VIndR)
        aInd = aInd[dr]
        X = X[:, dr]
    end
    
    dInd = setdiff(1:n_start, dInd)
    
    if !isnothing(v)
        X = XS[:, dInd]
    end
    XC = X[:, 1:end]
    
    compression_ratio = size(XC, 2) / n_start
    println("compression ratio: $compression_ratio")
    
    return XC, dInd
end