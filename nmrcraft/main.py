import rdkit
import matplotlib.pyplot as plt


print("Hello World!")


# Sample data
x = [1, 2, 3, 4, 5]
y = [3, 7, 2, 8, 1]

# Create the plot
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Sample Data')  # Plot the data with markers, line style, and color
plt.title('Example Plot with Matplotlib')  # Add a title
plt.xlabel('X-axis')  # Label the X-axis
plt.ylabel('Y-axis')  # Label the Y-axis
plt.grid(True)  # Add a grid for better readability
plt.legend()  # Add a legend to identify the data series

# Customize the plot (optional)
plt.xticks([1, 2, 3, 4, 5])  # Set specific X-axis ticks
plt.yticks([0, 2, 4, 6, 8, 10])  # Set specific Y-axis ticks
plt.xlim([0, 6])  # Set X-axis limits
plt.ylim([0, 10])  # Set Y-axis limits

# Show the plot
plt.savefig("/code/gen-data/figure.svg")

print("This is ebic??")