import numpy as np
from scipy.fftpack import fft

class FeatureExtractors:
    def __init__(self, n_nodes=300):
        self.n_nodes = n_nodes
        # ESN/TRFM용 랜덤 가중치
        self.W_in = np.random.randn(n_nodes, 1)
        self.phi = np.random.randn(n_nodes)
        # ESN용 내부 연결 (Spectral Radius=1.0)
        W = np.random.randn(n_nodes, n_nodes)
        self.W = W * (1.0 / np.max(np.abs(np.linalg.eigvals(W))))
        # SRFM용 랜덤 투영 행렬 (MFSC 33차원 -> 300차원)
        self.W_srfm = np.random.randn(n_nodes, 33)

    def get_esn(self, u):
        """ESN: 시간적 기억을 가진 레저버의 평균 활성화 [cite: 263, 282]"""
        x = np.zeros(self.n_nodes)
        states = []
        for val in u:
            x = np.tanh(self.W @ x + self.W_in @ [val] + self.phi)
            states.append(x)
        return np.concatenate(([1.0], np.mean(states, axis=0)))

    def get_trfm(self, u):
        """TRFM: 내부 연결(W=0)이 없는 메모리리스 레저버 [cite: 279, 280]"""
        # x(t) = tanh(W_in * u(t) + phi)
        states = np.tanh(np.outer(u, self.W_in.flatten()) + self.phi)
        return np.concatenate(([1.0], np.mean(states, axis=0)))

    def get_mfsc(self, u):
        """MFSC: 주파수 도메인 특징 (평균 + 32개 계수) [cite: 300, 311]"""
        # 단순화를 위해 FFT 절반 크기를 32개 빈으로 평균화 (Mel-scale 유사 구현)
        spec = np.abs(fft(u)[:len(u)//2])
        bins = np.array_split(spec, 32)
        coeffs = [np.mean(b) for b in bins]
        return np.concatenate(([1.0], [np.mean(u)], coeffs))

    def get_srfm(self, u):
        """SRFM: MFSC 특징을 다시 비선형 레저버로 투영 [cite: 320, 324]"""
        mfsc_feat = self.get_mfsc(u)[1:] # 편향 제외
        srfm_feat = np.tanh(self.W_srfm @ mfsc_feat + self.phi)
        return np.concatenate(([1.0], srfm_feat))

    def get_dem(self, u, M=300):
        """DEM: 평균과 자기공보산(Auto-covariance) 함수 [cite: 341, 345]"""
        u_mean = np.mean(u)
        u_centered = u - u_mean
        # M개의 래그(lag)에 대한 자기공보산 계산
        covs = [np.mean(u_centered[:len(u)-m] * u_centered[m:]) for m in range(M)]
        return np.concatenate(([1.0, u_mean], covs))