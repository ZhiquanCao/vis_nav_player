import json
import matplotlib.pyplot as plt
import math
import os
import pickle
import time

class TrajectoryMap:
    def __init__(self):
        """Initialize the TrajectoryMap with default values."""
        self.actions_json_filepath = os.path.join(os.getcwd(), "image_actions.json")
        self.actions = []
        self.trajectory_x = []
        self.trajectory_y = []
        self.angle_radians = 0
        self.step_size = 0.1
        self.step_angle = 0.04
        # plt.ion()
        self.figure, self.ax = plt.subplots()
        self.current_dot, self.current_annotation = None, None
        self.load_map()
        
    
    def read_json_file(self, filepath):
        """
        Reads a JSON file and returns the data as a Python object.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict or list: Parsed JSON data, or None if an error occurs.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON data in {filepath}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    
    def load_actions(self, filepath):
        """
        Loads actions from a JSON file and maps them to trajectory commands.

        Args:
            filepath (str): The path to the JSON file containing actions.
        """
        data = self.read_json_file(filepath)
        if data:
            for i in data:
                ac = data[i]['action']
                if ac == "Action.FORWARD":
                    self.actions.append('f')
                elif ac == "Action.LEFT":
                    self.actions.append('l')
                elif ac == "Action.RIGHT":
                    self.actions.append('r')
                elif ac == "Action.BACKWARD":
                    self.actions.append('b')
                else:
                    print(f"Unrecognized action '{ac}' in entry '{i}'. Ignoring.")
    
    def compute_trajectory(self):
        """Computes the trajectory based on the loaded actions."""
        x, y = 0, 0  # Start at the origin
        self.trajectory_x = [x]
        self.trajectory_y = [y]
        self.angle_radians = 0
        
        for action in self.actions:
            if action == 'f':
                x += self.step_size * math.cos(self.angle_radians)
                y += self.step_size * math.sin(self.angle_radians)
            elif action == 'b':
                x -= self.step_size * math.cos(self.angle_radians)
                y -= self.step_size * math.sin(self.angle_radians)
            elif action == 'l':
                self.angle_radians += self.step_angle
            elif action == 'r':
                self.angle_radians -= self.step_angle
            else:
                print(f"Warning: Invalid action '{action}' encountered. Ignoring.")
            self.trajectory_x.append(x)
            self.trajectory_y.append(y)
    
    def draw_map(self):
        """Draws the trajectory on a matplotlib plot."""
        # self.ax.clear()
        self.ax.plot(self.trajectory_x, self.trajectory_y, marker='o', linestyle='-', color='#ADD8E6')
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.ax.set_title("Trajectory Plot")
        self.ax.grid(True)

        self.ax.plot(self.trajectory_x[0], self.trajectory_y[0], 'b+')
        self.ax.annotate(f'Start', (self.trajectory_x[0], self.trajectory_y[0]),
                         textcoords="offset points", xytext=(0,10), ha='center', color='blue')
        self.ax.plot(self.trajectory_x[-1], self.trajectory_y[-1], 'bx')
        self.ax.annotate(f'End', (self.trajectory_x[-1], self.trajectory_y[-1]),
                         textcoords="offset points", xytext=(0,10), ha='center', color='green')
        # plt.draw()
    
    def save_map(self, image_path='trajectory_map.png', data_path='trajectory_data.pkl'):
        """
        Saves the plotted map as an image and the trajectory data.

        Args:
            image_path (str): Filename for the saved map image.
            data_path (str): Filename for the saved trajectory data.
        """
        # Save the plot as an image
        self.figure.savefig(image_path)
        print(f"Map image saved as '{image_path}'.")
        
        # Save the trajectory data using pickle
        with open(data_path, 'wb') as f:
            pickle.dump({
                'actions': self.actions,
                'trajectory_x': self.trajectory_x,
                'trajectory_y': self.trajectory_y
            }, f)
        print(f"Trajectory data saved as '{data_path}'.")
    
    def load_map(self, image_path='trajectory_map.png', data_path='trajectory_data.pkl'):
        """
        Loads the trajectory data and renders the map.

        Args:
            image_path (str): Filename of the saved map image.
            data_path (str): Filename of the saved trajectory data.
        """
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.actions = data['actions']
                self.trajectory_x = data['trajectory_x']
                self.trajectory_y = data['trajectory_y']
            self.draw_map()
            print(f"Map loaded from '{image_path}' and '{data_path}'.")
        else:
            tm.load_actions(self.actions_json_filepath)
    
            tm.compute_trajectory()

            tm.draw_map()

            tm.save_map()

    def show_target(self, index):
        self.ax.plot(self.trajectory_x[index], self.trajectory_y[index], 'g*')
        self.ax.annotate(f'Target-{index}', (self.trajectory_x[index], self.trajectory_y[index]),
                         textcoords="offset points", xytext=(0,10), ha='center', color='green')
        plt.draw()
        plt.pause(0.01)

    def show_dot(self, index, ):
        """
        Highlights a specific point on the trajectory.

        Args:
            index (int): The index of the action/image to highlight.
        """
        if index < 0 or index >= len(self.trajectory_x):
            print("Error: Index out of range.")
            return

        if self.current_dot or self.current_annotation:
            self.current_dot.remove()
            self.current_annotation.remove()
        
        # Highlight the specific point
        self.current_dot, = self.ax.plot(self.trajectory_x[index], self.trajectory_y[index], 'ro')  # Red dot
        self.current_annotation = self.ax.annotate(f'{index}', (self.trajectory_x[index], self.trajectory_y[index]),
                         textcoords="offset points", xytext=(0,10), ha='center', color='red')
        # self.current_dot = self.ax.plot(self.trajectory_x[index], self.trajectory_y[index], 'ro')  # Red dot
        # self.current_annotation = self.ax.annotate(f'{index}', (self.trajectory_x[index], self.trajectory_y[index]),
        #                  textcoords="offset points", xytext=(0,10), ha='center', color='red')
        plt.draw()
        plt.pause(0.1)

        # Revert back
        self.ax.plot(self.trajectory_x[index], self.trajectory_y[index], 'ro')  # Red dot
        self.ax.annotate(f'{index}', (self.trajectory_x[index], self.trajectory_y[index]),
                         textcoords="offset points", xytext=(0,10), ha='center', color='red')
        print(f"Highlighted point at index {index}.")

        

# Example Usage
if __name__ == "__main__":
    # Specify the path to your JSON file
    filepath = '/home/zhiquancao/master2yr/perception/vis_nav_player/image_actions.json'  # Update as needed
    
    tm = TrajectoryMap()
    
    tm.load_map()
    
    tm.show_dot(1000)

    

    plt.show(block=True)
    
