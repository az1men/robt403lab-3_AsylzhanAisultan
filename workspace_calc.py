import numpy as np
import matplotlib.pyplot as plt

# Robot Link Lengths
L1 = 0.675 * 2  # Length of Link 1
L2 = 0.675 * 2  # Length of Link 2
L3 = 0.467  # Length of Link 3

# Function for forward kinematics
def forward_kinematics(theta1, theta2, theta3):
    """
    Calculate the (x, y) position of the end-effector based on the joint angles.
    """
    x = (L1 * np.cos(theta1) +
         L2 * np.cos(theta1 + theta2) +
         L3 * np.cos(theta1 + theta2 + theta3))
    y = (L1 * np.sin(theta1) +
         L2 * np.sin(theta1 + theta2) +
         L3 * np.sin(theta1 + theta2 + theta3))
    return x, y
def plot_workspace():
    # Sampling range for joint angles (in radians)
    theta1_range = np.linspace(-np.pi / 2, np.pi / 2, 100)
    theta2_range = np.linspace(-np.pi / 2, np.pi / 2, 100)
    theta3_range = np.linspace(-np.pi / 2, np.pi / 2, 100)

    # Lists to store the workspace coordinates
    x_workspace = []
    y_workspace = []

    # Compute the workspace by sampling joint angles
    for theta1 in theta1_range:
        for theta2 in theta2_range:
            for theta3 in theta3_range:
                x, y = forward_kinematics(theta1, theta2, theta3)
                x_workspace.append(x)
                y_workspace.append(y)
    # Plot the workspace
    plt.figure(figsize=(8, 8))
    plt.plot(x_workspace, y_workspace, 'b.', markersize=1)
    plt.title('Workspace of 3-DOF Planar Robot')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Execute the function to plot the workspace
if __name__ == '__main__':
    plot_workspace()


