import numpy as np

diameter = 10
length = 8.9
outer_circle_diameter = 11.


circle_radius = diameter * 0.5
square_length = length
hexagon_length = outer_circle_diameter * 0.5

circle_area = np.pi  * np.square(circle_radius)
square_area = np.square(square_length)
hexagon_area = 3 * np.sqrt(3) / 2 * np.square(hexagon_length)
base_area = 78.
ratio = 0.05

print('circle area: {:.2f}'.format(circle_area))
print('square area: {:.2f}'.format(square_area))
print('hexagon area: {:.2f}'.format(hexagon_area))
print(base_area * ratio)