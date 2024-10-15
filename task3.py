import numpy as np
import matplotlib.pyplot as plt

# Robot Link Lengths
x0 = 0.86 # Length of the base that does not move
y0 = 0
L1 = 0.675 * 2  # Length of Link 1
L2 = 0.675 * 2  # Length of Link 2
L3 = 0.467  # Length of Link 3

def inverse_kinematics_with_orientation(x, y, phi):
    """
    Calculate the joint angles q1, q2, q3 given a target position (x, y)
    and orientation phi for the end-effector.

    Parameters:
    x : float : x-coordinate of the end-effector
    y : float : y-coordinate of the end-effector
    phi : float : desired orientation of the end-effector (in radians)

    Returns:
    q1, q2, q3 : float : joint angles in radians
    """
    # Calculate the wrist center position
    x_wrist = x - L3 * np.cos(phi)
    y_wrist = y - L3 * np.sin(phi)

    # Inverse kinematics for the first two joints
    # Calculate distance from base to wrist center
    D = (x_wrist**2 + y_wrist**2 - L1**2 - L2**2) / (2 * L1 * L2)

    # Check if the point is reachable
    if abs(D) > 1:
        raise ValueError("Target is outside of the reachable workspace.")

    # Elbow up solution for q2
    q2 = np.arctan2(np.sqrt(1 - D**2), D)  

    # Solve for q1
    q1 = np.arctan2(y_wrist, x_wrist) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))

    # Solve for q3
    q3 = phi - q1 - q2

    return q1, q2, q3

def plot_manipulator_with_orientation(q1, q2, q3):
    """
    Plot the configuration of the 3-DOF planar robot given joint angles,
    including the orientation of the end-effector.

    Parameters:
    q1, q2, q3 : float : joint angles in radians
    """
    # Joint positions
    x0, y0 = 0, 0
    x1, y1 = L1 * np.cos(q1), L1 * np.sin(q1)
    x2, y2 = x1 + L2 * np.cos(q1 + q2), y1 + L2 * np.sin(q1 + q2)
    x3, y3 = x2 + L3 * np.cos(q1 + q2 + q3), y2 + L3 * np.sin(q1 + q2 + q3)

    # Plot the manipulator
    plt.figure(figsize=(8, 8))
    plt.plot([x0, x1], [y0, y1], 'ro-', label='Link 1')
    plt.plot([x1, x2], [y1, y2], 'go-', label='Link 2')
    plt.plot([x2, x3], [y2, y3], 'bo-', label='Link 3')
    plt.plot(x3, y3, 'mo', label='End Effector')

    # Add end-effector orientation arrow
    arrow_length = 0.2  # Length of the arrow to indicate orientation
    arrow_dx = arrow_length * np.cos(q1 + q2 + q3)
    arrow_dy = arrow_length * np.sin(q1 + q2 + q3)
    plt.arrow(x3, y3, arrow_dx, arrow_dy, head_width=0.05, head_length=0.05, fc='r', ec='r')

    plt.title('Manipulator Configuration with End-Effector Orientation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

# Example usage for Task 3
if __name__ == '__main__':
    # Desired end-effector position and orientation
    x_target = 0.0
    y_target = 3.0
    phi_target = np.pi/2  # 0 degrees (horizontal orientation)

    try:
        # Calculate inverse kinematics
        q1, q2, q3 = inverse_kinematics_with_orientation(x_target, y_target, phi_target)
        print("Calculated Joint Angles:\n q1 = {:.2f} rad, q2 = {:.2f} rad, q3 = {:.2f} rad".format(q1,q2,q3))

        # Plot the manipulator configuration with orientation
        plot_manipulator_with_orientation(q1, q2, q3)
    except ValueError as e:
        print(e)
