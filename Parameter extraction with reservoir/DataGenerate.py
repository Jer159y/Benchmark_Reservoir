import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def generate_lorenz(c_val, t_points=5000, dt=0.1, transient=1000):
    """
    논문 Sec. IV A 설정을 따르는 Lorenz 시스템 데이터 생성
    """
    # 시스템 방정식 [cite: 365]
    def lorenz_deriv(t, state, a, b, c):
        v1, v2, v3 = state
        return [a * (v2 - v1), 
                v1 * (c - v3) - v2, 
                v1 * v2 - b * v3]

    # 논문 고정 파라미터 
    a, b = 10.0, 8/3.0
    
    # 초기 조건: 랜덤 [cite: 368]
    initial_state = np.random.uniform(-1, 1, 3)
    
    # 총 시간 계산 (과도기 포함)
    t_span = (0, (t_points + transient) * dt)
    t_eval = np.linspace(0, t_span[1], t_points + transient)
    
    # solve_ivp를 이용한 RK45 (논문의 RK4와 유사한 고정밀도) 해결
    sol = solve_ivp(lorenz_deriv, t_span, initial_state, 
                    args=(a, b, c_val), t_eval=t_eval, method='RK45')
    
    # 과도기 제거 (Transient removal) [cite: 369]
    data = sol.y[:, transient:]
    
    # 정규화: Zero mean, Unit variance [cite: 349, 350]
    # (실제 학습 시에는 Training set의 통계량을 사용해야 함)
    v1_signal = data[0, :]
    v1_norm = (v1_signal - np.mean(v1_signal)) / np.std(v1_signal)
    
    return v1_norm

def generate_rossler(a_val, t_points=5000, dt=1.0, transient=2000):
    """
    논문 Sec. IV B 설정을 따르는 Rössler 시스템 데이터 생성
    """
    # 시스템 방정식 [cite: 409]
    def rossler_deriv(t, state, a, b, c):
        v1, v2, v3 = state
        return [-v2 - v3,
                v1 + a * v2,
                b + v3 * (v1 - c)]

    # 논문 고정 파라미터 
    b, c = 0.2, 5.7
    
    initial_state = np.random.uniform(-1, 1, 3)
    
    # Rössler는 dt=1.0으로 다운샘플링함 
    t_span = (0, (t_points + transient) * dt)
    t_eval = np.linspace(0, t_span[1], t_points + transient)
    
    sol = solve_ivp(rossler_deriv, t_span, initial_state, 
                    args=(a_val, b, c), t_eval=t_eval, method='RK45')
    
    data = sol.y[:, transient:]
    
    v1_signal = data[0, :]
    v1_norm = (v1_signal - np.mean(v1_signal)) / np.std(v1_signal)
    
    return v1_norm

# --- 실행 예시 ---
# Lorenz 시스템에서 c=28 (카오스 영역)인 경우의 데이터 생성
lorenz_data = generate_lorenz(c_val=28.0)

plt.figure(figsize=(12, 4))
plt.plot(lorenz_data[:1000])
plt.title("Generated Lorenz v1 Signal (Normalized, c=28)")
plt.xlabel("Time steps")
plt.ylabel("Value")
plt.show()