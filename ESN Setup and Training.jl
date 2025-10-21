using ReservoirComputing

# 하이퍼파라미터 설정 (Rössler 시스템에 최적화된 예시 값) [9]
N_res = 1000       # Reservoir 노드 수
α = 0.178          # Leaking rate (누출률)
ρ = 0.2            # Spectral radius (스펙트럼 반경)
d = 0.301          # Density of W_res (연결 밀도)
σ = 0.087          # Scaling factor of W_in (입력 가중치 스케일링)
β = 4.11e-5        # Regularization parameter (규제 매개변수)

# ESN 객체 초기화
esn = ESN(U_train,
        Wres = generate_internal_weights(N_res, ρ, density=d),
        Win = generate_input_weights(N_res, size(U_train, 1), σ),
        leaky_rate = α)

# W_out 훈련 및 계산 (Ridge Regression 사용)
Wout = train(esn, Y_target, β) # ReservoirComputing.jl 내의 train 함수 사용

# ESN 모델 완성
rc_model = (esn, Wout)