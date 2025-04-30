import numpy as np

class IndistinguishabilityMetrics:
    def __init__(self, epsilon: float, delta: float = 0.0):
        self.epsilon = epsilon
        self.delta = delta

    def differential_privacy_check(self, p_d1: float, p_d2: float) -> bool:
        """
        Check ε-DP: p(D1) <= e^ε * p(D2)
        """
        lhs = p_d1
        rhs = np.exp(self.epsilon) * p_d2
        return lhs <= rhs

    def approximate_dp_check(self, p_d1: float, p_d2: float) -> bool:
        """
        Check (ε, δ)-DP: p(D1) <= e^ε * p(D2) + δ
        """
        lhs = p_d1
        rhs = np.exp(self.epsilon) * p_d2 + self.delta
        return lhs <= rhs
    


if __name__ == "__main__":
    dp = IndistinguishabilityMetrics(epsilon=0.5, delta=1e-5)

    p_d1 = 0.12
    p_d2 = 0.10

    print("ε-DP satisfied:", dp.differential_privacy_check(p_d1, p_d2))
    print("(ε, δ)-DP satisfied:", dp.approximate_dp_check(p_d1, p_d2))
