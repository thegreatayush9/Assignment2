import numpy as np

# -----------------------------
# 1. Model Parameters (Source 20)
# -----------------------------
F       = 1.0        # m3/h
V       = 1.0        # m3
CAf     = 10.0       # kgmol/m3
k0      = 36e6       # h^-1
E       = 12000.0    # kcal/kgmol
R       = 1.987      # kcal/(kgmol.K)
dH_neg  = 6500.0     # (-deltaH) kcal/kgmol
UA      = 150.0      # kcal/(C h)
Tf      = 298.0      # K
Tj0     = 298.0      # K
rhoCp   = 500.0      # (rho Cp) kcal/(m3 C)
rhojCj  = 600.0      # (rhoj Cj) kcal/(m3 C)
Fj      = 1.25       # m3/h
Vj      = 0.25       # m3

# -----------------------------
# 2. Steady State Equations (LHS = 0)
# -----------------------------
def F_system(x):
    CA, T, Tj = x
   
    # Rate of reaction [cite: 7]
    r = k0 * np.exp(-E / (R * T)) * CA
   
    # Mass Balance: V(dCA/dt) = F*CAf - F*CA - r*V
    f1 = F * CAf - F * CA - r * V
   
    # Reactor Energy: rho*Cp*V(dT/dt) = rho*Cp*F(Tf - T) + (-dH)*V*r - UA(T - Tj)
    f2 = rhoCp * F * (Tf - T) + dH_neg * V * r - UA * (T - Tj)
   
    # Jacket Energy: rhoj*Cj*Vj(dTj/dt) = rhoj*Cj*Fj(Tj0 - Tj) + UA(T - Tj)
    f3 = rhojCj * Fj * (Tj0 - Tj) + UA * (T - Tj)
   
    return np.array([f1, f2, f3])

# -----------------------------
# 3. Custom Newton-Raphson
# -----------------------------
def get_jacobian(x, eps=1e-5):
    """Numerically calculates the Jacobian matrix"""
    J = np.zeros((3, 3))
    f0 = F_system(x)
    for i in range(3):
        x_plus = np.copy(x)
        x_plus[i] += eps
        f1 = F_system(x_plus)
        J[:, i] = (f1 - f0) / eps
    return J

def solve_newton(guess, tol=1e-7, max_iter=50):
    x = np.array(guess, dtype=float)
    for i in range(max_iter):
        f = F_system(x)
        if np.linalg.norm(f) < tol:
            return x
       
        J = get_jacobian(x)
       
        # --- THE ONE-LINER CHECK ---
        # If the determinant is 0 (or very close to it), the matrix is not invertible.
        if np.isclose(np.linalg.det(J), 0, atol=1e-12):
            print(f"Iteration {i}: Matrix is singular (non-invertible). Stopping.")
            return None
       
        try:
            delta = np.linalg.solve(J, -f)
            x = x + delta
        except np.linalg.LinAlgError:
            return None
    return x

# -----------------------------
# 4. Find the Three Steady States
# -----------------------------
# We use distinct guesses for the three regions: cold, middle, and hot.
initial_guesses = [
    [9.5, 300.0, 300.0],  # Guess for Low conversion
    [5.0, 350.0, 320.0],  # Guess for Middle state
    [1.0, 410.0, 340.0]   # Guess for High conversion
]

print(f"{'Steady State':<15} | {'CA (kgmol/m3)':<15} | {'T (K)':<10} | {'Tj (K)':<10}")
print("-" * 60)

labels = ["Low Conversion", "Intermediate", "High Conversion"]
for i, guess in enumerate(initial_guesses):
    result = solve_newton(guess)
    if result is not None:
        print(f"{labels[i]:<15} | {result[0]:<15.4f} | {result[1]:<10.2f} | {result[2]:<10.2f}")
    else:

        print(f"{labels[i]:<15} | Failed to converge")
