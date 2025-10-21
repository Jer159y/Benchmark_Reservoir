
# 초기 설정
r_actual = 3.2      # 실제 시스템 파라미터 (r)
r_prime_n = 0.0     # 현재 제어 입력 (r')
r_c = r_actual + r_prime_n # 현재 유효 파라미터 (rc)
r_star = 3.5        # 목표 파라미터 값 (r*)
epsilon_logm = 0.005 # 제어 강성 [9]
window_length = 200 # 평균 창 크기 [9]

# 1. 초기 시스템 상태 설정
x_n = 0.5 

# 2. 제어 루프 시작 (단계 n = 1, 2, ...)
for n in 1:N_max
    # A. 시스템 상태 업데이트 (Logistic Map)
    x_n_plus_1 = r_c * x_n * (1 - x_n)
    
    # B. ESN 입력 준비 (단일 변수 x_n+1)
    U_input = [x_n_plus_1] 
    
    # C. Reservoir 상태 업데이트 및 파라미터 예측 (r_p)
    # ... (Step 3.1의 C, D, E와 동일한 과정) ...
    r_p = calculate_parameter_prediction(rc_model, U_input, window_length) # 예측된 현재 파라미터
    
    # D. 제어 입력 r' 업데이트 (식 10)
    r_prime_n_plus_1 = r_prime_n + epsilon_logm * (r_star - r_p)
    
    # E. 유효 파라미터 업데이트
    r_c = r_actual + r_prime_n_plus_1
    
    # F. 다음 단계 준비
    x_n = x_n_plus_1
    r_prime_n = r_prime_n_plus_1
end