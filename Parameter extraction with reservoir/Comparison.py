def run_comparison_experiment(system='lorenz'):
    n_samples = 300
    params = np.linspace(0, 40, n_samples) if system == 'lorenz' else np.linspace(0, 0.38, n_samples)
    extractors = FeatureExtractors(n_nodes=300)
    
    # 모델 정의
    model_names = ['ESN', 'TRFM', 'MFSC', 'SRFM', 'DEM']
    results = {name: [] for name in model_names}
    
    print(f"[{system.upper()}] 실험 시작...")
    for p in params:
        # 데이터 생성
        u = generate_lorenz(p) if system == 'lorenz' else generate_rossler(p)
        
        # 각 모델별 특징 추출
        results['ESN'].append(extractors.get_esn(u))
        results['TRFM'].append(extractors.get_trfm(u))
        results['MFSC'].append(extractors.get_mfsc(u))
        results['SRFM'].append(extractors.get_srfm(u))
        results['DEM'].append(extractors.get_dem(u))

    # 비교 결과 저장용
    comparison_table = []

    for name in model_names:
        X = np.array(results[name]).T
        Z = params.reshape(1, -1)
        
        # 데이터 분할 (훈련 200, 테스트 100)
        X_train, X_test = X[:, :200], X[:, 200:]
        Z_train, Z_test = Z[:, :200], Z[:, 200:]
        
        # 학습 및 예측
        R = train_readout(X_train, Z_train, lambd=1e-6)
        Z_pred = R @ X_test
        
        # 성능 측정 (NRMSE)
        error = calculate_nrmse(Z_test, Z_pred)
        comparison_table.append((name, error))
        
    return comparison_table

# 실험 실행
lorenz_comparison = run_comparison_experiment('lorenz')
print("\n--- Lorenz System 모델 비교 결과 (NRMSE) ---")
for name, err in lorenz_comparison:
    print(f"{name:5}: {err:.5f}")