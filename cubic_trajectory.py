import numpy as np
import matplotlib.pyplot as plt


# Function to compute cubic polynomial coefficients
def cubic_coefficients(q0, qf, v0, vf, t0, tf):
    """
    Calculate the cubic polynomial coefficients for a trajectory.

    Parameters:
    q0 : float : initial position
    qf : float : final position
    v0 : float : initial velocity
    vf : float : final velocity
    t0 : float : initial time
    tf : float : final time
 Returns:
    a0, a1, a2, a3 : float : coefficients of the cubic polynomial
    """
    # Define the matrix for boundary conditions
    M = np.array([[1, t0, t0**2, t0**3],
                  [0, 1, 2*t0, 3*t0**2],
                  [1, tf, tf**2, tf**3],
                  [0, 1, 2*tf, 3*tf**2]])
   
    # Define the boundary conditions vector
    b = np.array([q0, v0, qf, vf])
   
    # Solve for the coefficients a0, a1, a2, a3
    a = np.linalg.solve(M, b)
    return a
def cubic_trajectory(a0, a1, a2, a3, t):
    """
    Calculate the trajectory of a joint based on the cubic polynomial coefficients.

    Parameters:
    a0, a1, a2, a3 : float : coefficients of the cubic polynomial
    t : float : time

    Returns:
    q_t : float : position of the joint at time t
    """
    q_t = a0 + a1*t + a2*t**2 + a3*t**3
    return q_t
def plot_joint_trajectories(q0, qf, v0, vf, t0, tf, num_points=100):
    """
    Plot cubic trajectories for three joints from initial to final positions.

    Parameters:
    q0 : list : initial positions [q1_0, q2_0, q3_0]
    qf : list : final positions [q1_f, q2_f, q3_f]
    v0 : list : initial velocities [v1_0, v2_0, v3_0]
    vf : list : final velocities [v1_f, v2_f, v3_f]
    t0 : float : initial time
    tf : float : final time
    num_points : int : number of points for the trajectory
    """
    # Time vector
    t_range = np.linspace(t0, tf, num_points)

    # Storage for joint trajectories
    q1_traj = []
    q2_traj = []
    q3_traj = []
    a1 = cubic_coefficients(q0[0], qf[0], v0[0], vf[0], t0, tf)
    a2 = cubic_coefficients(q0[1], qf[1], v0[1], vf[1], t0, tf)
    a3 = cubic_coefficients(q0[2], qf[2], v0[2], vf[2], t0, tf)

    # Calculate the trajectory for each joint
    for t in t_range:
        q1_traj.append(cubic_trajectory(a1[0],a1[1],a1[2],a1[3], t))
        q2_traj.append(cubic_trajectory(a2[0],a2[1],a2[2],a2[3], t))
        q3_traj.append(cubic_trajectory(a3[0],a3[1],a3[2],a3[3], t))

    # Plot the joint trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(t_range, q1_traj, label='Theta 3')
    plt.plot(t_range, q2_traj, label='Theta 1')
    plt.plot(t_range, q3_traj, label='Theta 5')
    plt.title('Cubic Trajectory for Each Joint')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Initial and final conditions for each joint
    q0 = [np.pi/2, 0, 0]  # Initial positions
    qf = [np.pi/4, np.pi/4, -np.pi/4]  # Final positions
    v0 = [0, 0, 0]  # Initial velocities
    vf = [0, 0, 0]  # Final velocities
    t0 = 0  # Initial time
    tf = 5  # Final time

    # Plot the joint trajectories
    plot_joint_trajectories(q0, qf, v0, vf, t0, tf)
