function plot_spline!(p, X; color=:black, linewidth=0.7, head=false, headpos=0.5, headwidth=5, headlength=5, kwargs...)
    # 1. 스플라인 계산
    # MATLAB의 cscvn(X)는 x-y 좌표를 받아 파라메트릭 스플라인을 만듭니다.
    nodes = (X[1, :],)
    itp = scale(interpolate(X[2, :], BSpline(Cubic(Line(OnGrid())))), nodes...)
    
    t_fine = range(X[1, 1], X[1, end], length=100)
    y_fine = itp(t_fine)

    # 2. 선 그리기
    plot!(p, t_fine, y_fine, color=color, linewidth=linewidth, label=nothing; kwargs...)

    # 3. 화살표 (Annotation)
    # Plots.jl의 화살표는 구현이 까다롭습니다. 
    # 여기서는 head=true일 때 마지막 지점에 간단한 화살표를 추가합니다.
    if head
        p_ind = floor(Int, headpos * (length(t_fine)-1)) + 1
        if p_ind < 2
            p_ind = 2
        end
        
        # 화살표 방향 계산
        dx = t_fine[p_ind] - t_fine[p_ind-1]
        dy = y_fine[p_ind] - y_fine[p_ind-1]
        
        # quiver!를 사용하여 화살표 추가
        plot!(p, [t_fine[p_ind-1]], [y_fine[p_ind-1]], 
              quiver=([dx*0.1], [dy*0.1]), 
              color=color, linewidth=linewidth, 
              arrow=arrow(:closed, :head, headwidth/10, headlength/10),
              label=nothing; kwargs...)
    end
    
    return p
end

# 2D/3D 스플라인을 위한 추가 헬퍼 (b, d, g 플롯용)
# 2xN 또는 3xN 매트릭스를 입력받음
function parametric_spline(X; n_points=100)
    dims = size(X, 1)
    t = 0:1:(size(X, 2)-1) # 파라미터
    
    itps = []
    for d = 1:dims
        push!(itps, scale(interpolate(X[d, :], BSpline(Cubic(Line(OnGrid())))), t))
    end
    
    t_fine = range(t[1], t[end], length=n_points)
    
    pts = zeros(dims, n_points)
    for d = 1:dims
        pts[d, :] = itps[d](t_fine)
    end
    
    return pts
end