import numpy as np
import matplotlib.pyplot as plt

# Robot Link Lengths
L1 = 0.675 * 2  # Length of Link 1
L2 = 0.675 * 2  # Length of Link 2
L3 = 0.467  # Length of Link 3

def inverse_kinematics_with_orientation(x, y, phi):
    """
    Calculate the joint angles q1, q2, q3 given a target position (x, y)
    and orientation phi for the end-effector.
    """
    # Calculate the wrist center position
    x_wrist = x - L3 * np.cos(phi)
    y_wrist = y - L3 * np.sin(phi)

    # Calculate the distance to the wrist center
    D = (x_wrist**2 + y_wrist**2 - L1**2 - L2**2) / (2 * L1 * L2)

    if abs(D) > 1:
        raise ValueError("Target is outside of the reachable workspace.")

    # Compute q2 (elbow angle)
    q2 = np.arctan2(np.sqrt(1 - D**2), D)  # Elbow-up solution

    # Compute q1 (shoulder angle)
    q1 = np.arctan2(y_wrist, x_wrist) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))

    # Compute q3 (end-effector orientation relative to the base frame)
    q3 = phi - q1 - q2

    return q1, q2, q3

def cubic_polynomial(q0, qf, v0, vf, t0, tf, t):
    """
    Generate a cubic polynomial trajectory for a single joint.
    """
    M = np.array([[1, t0, t0**2, t0**3],
                  [0, 1, 2*t0, 3*t0**2],
                  [1, tf, tf**2, tf**3],
                  [0, 1, 2*tf, 3*tf**2]])
    
    b = np.array([q0, v0, qf, vf])
    a = np.linalg.solve(M, b)
    return a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3

def forward_kinematics(q1, q2, q3):
    """
    Calculate the (x, y) position of the end-effector based on the joint angles.
    """
    x = (L1 * np.cos(q1) +
         L2 * np.cos(q1 + q2) +
         L3 * np.cos(q1 + q2 + q3))
    y = (L1 * np.sin(q1) +
         L2 * np.sin(q1 + q2) +
         L3 * np.sin(q1 + q2 + q3))
    return x, y

def plot_via_points_trajectory(via_points_cartesian, time_intervals):
    """
    Plot the Cartesian space trajectory of the manipulator as it moves through multiple via points.
    """
    cartesian_trajectory = []

    # Convert Cartesian via points to joint angles using inverse kinematics
    via_points = []
    for point in via_points_cartesian:
        x, y, phi = point[0], point[1], point[2]  # Use phi for each via point
        try:
            q1, q2, q3 = inverse_kinematics_with_orientation(x, y, phi)
            via_points.append((q1, q2, q3))
        except ValueError as e:
            print("Error: Point ({}, {}) is outside of the reachable workspace: {}".format(x,y,e))

    # Generate the joint-space trajectory using cubic polynomials
    for i in range(len(via_points) - 1):
        q0 = via_points[i]
        qf = via_points[i + 1]
        t0 = sum(time_intervals[:i])
        tf = t0 + time_intervals[i]

        for t in np.linspace(t0, tf, 50):  # Increased the number of samples for smoother trajectory
            q1 = cubic_polynomial(q0[0], qf[0], 0, 0, t0, tf, t)
            q2 = cubic_polynomial(q0[1], qf[1], 0, 0, t0, tf, t)
            q3 = cubic_polynomial(q0[2], qf[2], 0, 0, t0, tf, t)
            x, y = forward_kinematics(q1, q2, q3)
            cartesian_trajectory.append((x, y))

    # Convert the list to a NumPy array for proper indexing
    cartesian_trajectory = np.array(cartesian_trajectory)

    # Plot the Cartesian space trajectory and the via points
    plt.figure(figsize=(8, 8))
    plt.plot(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], 'b-', label='Manipulator Trajectory', linewidth=2)
    plt.plot([p[0] for p in via_points_cartesian], [p[1] for p in via_points_cartesian], 'ro', label='Via Points', markersize=8)
    plt.title('Manipulator Cartesian Trajectory with Multiple Via Points and Dynamic Orientation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()
# Example usage to generate and visualize the trajectory
if __name__ == '__main__':
    # Define via points in Cartesian space with varying orientations (x, y, phi)
    via_points_cartesian = [
        (3, 0, np.pi/4),
        (2, 1, np.pi/6),
        (1.5, 1.5, np.pi/3),
        (1, 2, np.pi/2),
        (0, 3, np.pi/2)
    ]

    # Define time intervals between each via point
    time_intervals = [2, 2, 2, 2]  # seconds for each transition between via points

    # Plot the trajectory through the via points
    plot_via_points_trajectory(via_points_cartesian, time_intervals)
