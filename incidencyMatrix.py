import tkinter as tk
import numpy as np
from algorithm import ML_prediction
import keras
from keras.models import Sequential
from keras.layers import Dense


# Constants for the grid size and wall properties
GRID_WIDTH = 800 // 7  # 800 pixels wide and each wall is 7px, so we can fit this many columns
GRID_HEIGHT = 600 // 7  # 600 pixels high and each wall is 7px, so we can fit this many rows
CELL_SIZE = 7  # Wall size is 7x7
WALL_COLOR = 'black'
EMPTY_COLOR = 'white'
CURSOR_COLOR = 'red'

# List to store the traversed wall positions
a = []

class WallBuilderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wall Builder with Arrow Keys")

        # Initialize the grid, where 0 is empty, 1 is wall
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # Create a canvas to draw the grid
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg=EMPTY_COLOR)
        self.canvas.pack()

        # Draw the grid and cursor initially
        self.cursor_pos = [0, 0]  # Cursor position at the top-left corner
        self.draw_grid()
        
        # Bind arrow keys to move the cursor
        self.root.bind("<Up>", self.move_cursor)
        self.root.bind("<Down>", self.move_cursor)
        self.root.bind("<Left>", self.move_cursor)
        self.root.bind("<Right>", self.move_cursor)
        
        # Bind Enter to toggle wall creation
        self.root.bind("<Return>", self.toggle_wall_creation)

        # Flag for continuous wall placement
        self.is_creating_wall = False

    def draw_grid(self):
        # Iterate through the grid and draw each cell
        self.canvas.delete("all")  # Clear the canvas
        
        # Draw the walls (black squares) and empty spaces (white background)
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                # Set cell color based on its state (0 = empty, 1 = wall)
                if self.grid[row][col] == 1:
                    color = WALL_COLOR
                else:
                    color = EMPTY_COLOR

                # Draw the rectangle for each cell (no grid lines now)
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='', fill=color, width=0)

        # Draw the cursor position
        cursor_x1 = self.cursor_pos[1] * CELL_SIZE
        cursor_y1 = self.cursor_pos[0] * CELL_SIZE
        cursor_x2 = cursor_x1 + CELL_SIZE
        cursor_y2 = cursor_y1 + CELL_SIZE
        self.canvas.create_rectangle(cursor_x1, cursor_y1, cursor_x2, cursor_y2, outline=CURSOR_COLOR, width=3)

    def move_cursor(self, event):
        # Move the cursor based on key presses
        if event.keysym == "Up" and self.cursor_pos[0] > 0:
            self.cursor_pos[0] -= 1
            if self.is_creating_wall:
                a.append((self.cursor_pos[1], self.cursor_pos[0]))
        elif event.keysym == "Down" and self.cursor_pos[0] < GRID_HEIGHT - 1:
            self.cursor_pos[0] += 1
            if self.is_creating_wall:
                a.append((self.cursor_pos[1], self.cursor_pos[0]))
        elif event.keysym == "Left" and self.cursor_pos[1] > 0:
            self.cursor_pos[1] -= 1
            if self.is_creating_wall:
                a.append((self.cursor_pos[1], self.cursor_pos[0]))
        elif event.keysym == "Right" and self.cursor_pos[1] < GRID_WIDTH - 1:
            self.cursor_pos[1] += 1
            if self.is_creating_wall:
                a.append((self.cursor_pos[1], self.cursor_pos[0]))
        
        # Place wall if wall creation is active
        if self.is_creating_wall:
            self.place_wall()

        self.draw_grid()

    def toggle_wall_creation(self, event):
        # Toggle wall creation on/off with Enter key
        self.is_creating_wall = not self.is_creating_wall
        print(f"Wall creation {'enabled' if self.is_creating_wall else 'disabled'}")

    def place_wall(self):
        # Place walls at the current cursor position
        row, col = self.cursor_pos
        # Set the cell to a wall (1) if it is empty (0)
        if self.grid[row][col] == 0:  # Only place wall if cell is empty
            self.grid[row][col] = 1

        # Redraw the grid after placing a wall
        self.draw_grid()


# Set up the Tkinter window
root = tk.Tk()
app = WallBuilderApp(root)

# Start the Tkinter event loop
root.mainloop()

# After the window is closed, print the recorded positions
print("Positions of placed walls:", a)

import numpy as np

# Function to find vertices (points where the path changes direction)
def find_vertices(coords):
    vertices = []
    if len(coords) < 2:
        return coords
    
    # First coordinate is always a vertex
    vertices.append(coords[0])

    # Traverse the coordinates to find turning points
    for i in range(1, len(coords) - 1):  # Exclude last point
        prev_point = coords[i - 1]
        curr_point = coords[i]
        next_point = coords[i + 1]
        
        # Check if the direction changes (either x or y changes direction)
        if (prev_point[0] == curr_point[0] and curr_point[0] != next_point[0]) or \
           (prev_point[1] == curr_point[1] and curr_point[1] != next_point[1]):
            vertices.append(curr_point)

    return vertices

# Function to check if two vertices are connected in a continuous manner
def are_connected(v1, v2, grid):
    # Check if only one coordinate (either x or y) changes continuously
    if v1[0] == v2[0]:  # Same x-coordinate, check continuous y
        start_y, end_y = min(v1[1], v2[1]), max(v1[1], v2[1])
        for y in range(start_y, end_y + 1):
            if grid[v1[0]][y] != 1:
                return False
        return True
    elif v1[1] == v2[1]:  # Same y-coordinate, check continuous x
        start_x, end_x = min(v1[0], v2[0]), max(v1[0], v2[0])
        for x in range(start_x, end_x + 1):
            if grid[x][v1[1]] != 1:
                return False
        return True
    return False

# Function to build the weight matrix based on connectivity
def build_weight_matrix(vertices, grid):
    n = len(vertices)
    weight_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            if are_connected(vertices[i], vertices[j], grid):
                weight_matrix[i][j] = 1
                weight_matrix[j][i] = 1  # Symmetric matrix
    
    return weight_matrix

vertices = find_vertices(a)

def ML_prediction(X):
    # Labels for camera placement at each node: 1 = camera, 0 = no camera
    y = np.array([1])  # Place cameras at all nodes for simplicity
    for _ in range(len(vertices)-1):
        new_y = np.array([1])

        y = np.append(y,[new_y])

    # Define the model
    model = Sequential()

    # Input layer (number of nodes as features)
    model.add(Dense(32, input_dim=X.shape[1], activation='relu'))

    # Hidden layer
    model.add(Dense(16, activation='relu'))

    # Output layer (binary: camera at node or not)
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=50, batch_size=2)

    # Predict camera placement for each node
    predictions = model.predict(X)
    print(predictions)  # Outputs values between 0 and 1, with values close to 1 indicating camera placement

# Sample grid with 1 for wall and 0 for empty space
grid = np.zeros((len(a), len(a)), dtype=int)

for x, y in a:
    grid[x][y] = 1


# Build the weight matrix based on the vertices and grid
weight_matrix = build_weight_matrix(vertices, grid)

# Print the vertices and the weight matrix
print("Vertices:", vertices)
print("Weight Matrix:")
print(weight_matrix)

X = np.array(weight_matrix)

ML_prediction(X)

