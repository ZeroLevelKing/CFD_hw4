# task3.py
import numpy as np
import matplotlib.pyplot as plt
from common import initialize_grid, sor_step

def find_optimal_omega(Nx_list, Ny_list, max_omega=1.99, tol=1e-5):
    optimal_omegas = []
    theoretical_omegas = []
    
    for Nx, Ny in zip(Nx_list, Ny_list):
        # 理论最优omega
        N = max(Nx, Ny)
        omega_opt = 2 / (1 + np.sin(np.pi/(N-1)))
        theoretical_omegas.append(omega_opt)
        
        # 实验寻找最优omega
        best_omega = 1.0
        min_iter = float('inf')
        
        # 在理论值附近搜索
        candidates = np.linspace(omega_opt*0.9, omega_opt*1.1, 11)
        for omega in candidates:
            T = initialize_grid(Nx, Ny)
            for iter in range(10000):
                residual = sor_step(T, 15/(Nx-1), 12/(Ny-1), omega)
                if residual < tol:
                    if iter < min_iter:
                        min_iter = iter
                        best_omega = omega
                    break
        
        optimal_omegas.append(best_omega)
        print(f"Nx={Nx}, Ny={Ny}: Optimal ω={best_omega:.3f} (Theory={omega_opt:.3f})")
    
    # 绘制比较图
    plt.plot(Nx_list, optimal_omegas, 'o-', label='Experimental')
    plt.plot(Nx_list, theoretical_omegas, 's--', label='Theoretical')
    plt.xlabel('Grid Size (Nx)')
    plt.ylabel('Optimal Relaxation Factor')
    plt.title('Optimal Relaxation Factor vs Grid Size')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    grid_sizes = [(21,17), (31,25), (41,33), (51,41)]
    find_optimal_omega([s[0] for s in grid_sizes], [s[1] for s in grid_sizes])