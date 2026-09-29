import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

K_P = 5
K_I = 0.1
K_D = 0.1
 
STEPS = 550
 
car = make_car(desired_v=20.0, dt=0.1)

#WRITE CODE HERE
velocities = []
errors = []
times = []

for i in range(STEPS):
    desired_accleration, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
    throttle = acceleration_to_throttle_percentage(desired_accleration)

    update(car, throttle)

    velocities.append(car["v"])
    errors.append(error)
    times.append(car["t"])

print(car)
plt.plot(times, velocities)
plt.show()
plt.plot(times, errors)
plt.show()