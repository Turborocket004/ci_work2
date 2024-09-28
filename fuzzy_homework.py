import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Input Variables
light_intensity = ctrl.Antecedent(np.arange(0, 1001, 1), 'light_intensity')  # 0 - 1000 lux
battery_level = ctrl.Antecedent(np.arange(0, 101, 1), 'battery_level')  # 0 - 100%

# Output Variable
light_status = ctrl.Consequent(np.arange(0, 2, 1), 'light_status')  # 0: Off, 1: On

# Define membership functions for light intensity
light_intensity['low'] = fuzz.trapmf(light_intensity.universe, [0, 0, 100, 300])
light_intensity['medium'] = fuzz.trimf(light_intensity.universe, [200, 500, 800])
light_intensity['high'] = fuzz.trapmf(light_intensity.universe, [600, 800, 1000, 1000])

# Define membership functions for battery level
battery_level['low'] = fuzz.trapmf(battery_level.universe, [0, 0, 20, 40])
battery_level['medium'] = fuzz.trimf(battery_level.universe, [30, 50, 80])
battery_level['high'] = fuzz.trapmf(battery_level.universe, [60, 80, 100, 100])

# Define membership functions for light status
light_status['off'] = fuzz.trimf(light_status.universe, [0, 0, 0])
light_status['on'] = fuzz.trimf(light_status.universe, [1, 1, 1])

# Define Fuzzy Rules
rule1 = ctrl.Rule(light_intensity['low'] & battery_level['high'], light_status['on'])
rule2 = ctrl.Rule(light_intensity['low'] & battery_level['low'], light_status['on'])
rule3 = ctrl.Rule(light_intensity['low'] & battery_level['medium'], light_status['on']) 
rule4 = ctrl.Rule(light_intensity['medium'] & battery_level['high'], light_status['on'])
rule5 = ctrl.Rule(light_intensity['medium'] & battery_level['low'], light_status['off'])  
rule6 = ctrl.Rule(light_intensity['medium'] & battery_level['medium'], light_status['off'])
rule7 = ctrl.Rule(light_intensity['high'], light_status['off'])

# Control System Creation and Simulation
light_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5,rule6,rule7])
light_sim = ctrl.ControlSystemSimulation(light_ctrl)

# Function to test the system and apply defuzzification (thresholding)
def test_system(light_val, battery_val):
    light_sim.input['light_intensity'] = light_val
    light_sim.input['battery_level'] = battery_val

    # Compute the result
    light_sim.compute()
    result = light_sim.output['light_status']
    
    # Defuzzification (Thresholding)
    if result > 0.5:
        return "On"  # If fuzzy result is more than 0.5, turn on
    else:
        return "Off"  # If fuzzy result is less than or equal to 0.5, turn off

# Example Testing
print(f"Light ON/OFF Status (Light: 100 lux, Battery: 90%): {test_system(100, 90)}")
print(f"Light ON/OFF Status (Light: 700 lux, Battery: 50%): {test_system(700, 50)}")
print(f"Light ON/OFF Status (Light: 400 lux, Battery: 30%): {test_system(400, 30)}")
print(f"Light ON/OFF Status (Light: 200 lux, Battery: 20%): {test_system(200, 20)}")
print(f"Light ON/OFF Status (Light: 300 lux, Battery: 25%): {test_system(300, 25)}")
print(f"Light ON/OFF Status (Light: 500 lux, Battery: 80%): {test_system(500, 80)}")
print(f"Light ON/OFF Status (Light: 150 lux, Battery: 60%): {test_system(150, 60)}")

