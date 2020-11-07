"""
Root to attempt1.py
New Tasks are defined here
/Lucas
"""

# import stuff

import math
import time

sim = 0
if sim:
    import cfsim.crtp as crtp
    from cfsim.crazyflie.log import LogConfig
    from cfsim.crazyflie.swarm import Swarm, CachedCfFactory
    from cfsim.crazyflie.syncLogger import SyncLogger
else:
    import cflib.crtp as crtp
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.swarm import Swarm, CachedCfFactory
    from cflib.crazyflie.syncLogger import SyncLogger


# ad logg data
logdata = {}

# set up URIs to connect to Crazyflies
uris = [
    "radio://0/90/2M/E7E7E7E703",
    "radio://0/90/2M/E7E7E7E704",
    "radio://0/100/2M/E7E7E7E705",
    "radio://0/100/2M/E7E7E7E706",
]

# set agent 1 (the bozz)

agent_1 = uris[0]

# set up other agent

agent_2 = uris[1]
agent_3 = uris[2]
agent_4 = uris[3]

# Sequence for the first crazyflie to fly
#    x    y    z

sequence = [
    (0.5, 0, 0.7),
    (0.5, -0.5, 0.7),
    (0.5, 0, 0.7),
    (0, 0, 0.7),
    (-1, 0, 0.7),
    (-1, 0, 0.7),
    (0, 0, 1.2),
    (0.5, 0, 1),
    (1, 0, 0.1),
]
# Starting Positions dict, keyed by uri
#    x    y    z
initialPositions = {
    agent_1: (1.0, 0.0, 0),
    agent_2: (-0, 0.0, 0.0),
    agent_3: (-1, 0.0, 0.0),
    agent_4: (-2, 0.0, 0.0),
}

set_work_dist = {
    agent_1: (0.0, 0.0, 0),
    agent_2: (-0.5, 0.3, 0.0),
    agent_3: (-0.5, -0.3, 0.0),
    agent_4: (-1, 0, 0.0),
}
side_mision = {
    agent_1: (0.0, 0.0, 0.0),
    agent_2: (0, 0.0, 0),
    agent_3: (1, -0.5, 0.1),
    agent_4: (0, -0.5, 0.1),
}

# Task are specified here (maby put futher up?)
tasks = {
    agent_1: "main_sequence",
    agent_2: "follow",
    agent_3: "changes",
    agent_4: "changes",
}


# Used to keep track of current position of the crazyflies
currentPositions = initialPositions.copy()  # shallow copy

""" SETUP FUNCTIONS (comand C comand V fron exempel) """


def wait_for_position_estimator(scf):
    print(scf.cf.link_uri, ": Waiting for estimator to find position...")

    log_config = LogConfig(name="Kalman Variance", period_in_ms=500)
    log_config.add_variable("kalman.varPX", "float")
    log_config.add_variable("kalman.varPY", "float")
    log_config.add_variable("kalman.varPZ", "float")

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data["kalman.varPX"])
            var_x_history.pop(0)
            var_y_history.append(data["kalman.varPY"])
            var_y_history.pop(0)
            var_z_history.append(data["kalman.varPZ"])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            if (
                (max_x - min_x) < threshold
                and (max_y - min_y) < threshold
                and (max_z - min_z) < threshold
            ):
                break
    print(scf.cf.link_uri, ": Position found.")


def set_initial_position(scf):
    initial = initialPositions[scf.cf.link_uri]
    scf.cf.param.set_value("kalman.initialX", initial[0])
    scf.cf.param.set_value("kalman.initialY", initial[1])
    scf.cf.param.set_value("kalman.initialZ", initial[2])


def reset_estimator(scf):
    scf.cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    scf.cf.param.set_value("kalman.resetEstimation", "0")
    wait_for_position_estimator(scf)


def wait_for_param_download(scf):
    print(scf.cf.link_uri, ": Waiting for parameters to be downloaded...")
    while not scf.cf.param.is_updated:
        time.sleep(1.0)
    print(scf.cf.link_uri, ": Parameters downloaded.")


# Start saving position data
def start_position_callback(scf):
    # Set position logging configuration
    log_config = LogConfig(name="Position", period_in_ms=50)
    log_config.add_variable("kalman.stateX", "float")
    log_config.add_variable("kalman.stateY", "float")
    log_config.add_variable("kalman.stateZ", "float")

    def position_callback(timestamp, data, log_config):
        x = data["kalman.stateX"]
        y = data["kalman.stateY"]
        z = data["kalman.stateZ"]
        currentPositions[scf.cf.link_uri] = (x, y, z)
        # print(scf.cf.link_uri, ': Position ', (x,y,z))
        logdata[scf.cf.link_uri]["x"].append(x)
        logdata[scf.cf.link_uri]["y"].append(y)
        logdata[scf.cf.link_uri]["z"].append(z)

    scf.cf.log.add_config(log_config)
    log_config.data_received_cb.add_callback(position_callback)
    log_config.start()


""" How it flys given it position and gaol position and  """


def fly_to(cf, position, K):
    # Compute velocity (P controller)
    vmax = 0.4  # Maximum velocity
    # K = 1  # Controller gain

    xe = currentPositions[cf.link_uri][0] - position[0]
    ye = currentPositions[cf.link_uri][1] - position[1]
    ze = currentPositions[cf.link_uri][2] - position[2]
    d = math.sqrt(xe ** 2 + ye ** 2 + ze ** 2)

    if d > 1e-4:
        v = K * d
        v = min(vmax, v)
        vx = -v * xe / d
        vy = -v * ye / d
        vz = -v * ze / d
    else:
        vx = 0.0
        vy = 0.0
        vz = 0.0

    # Send velocity
    cf.commander.send_velocity_world_setpoint(vx, vy, vz, 0)


""" Set upp agnet 1 task """

# Main loop each crazyflie runs until completion
def run_task(scf):
    try:
        cf = scf.cf
        task = tasks[cf.link_uri]
        print(cf.link_uri, ": Running task [%s]..." % (task))
        TimeZero = time.time()

        # Loop to retry the conditions if tasks have changed
        should_restart = True
        while should_restart:
            should_restart = False

            if task == "main_sequence":
                startTime = time.time()
                sequencePos = -1
                position = sequence[0]
                while True:
                    # Determine position reference based on time
                    relativeTime = time.time() - startTime
                    if (
                        relativeTime > (sequencePos + 1) * 5
                    ):  # Fly to each point for 5 seconds
                        sequencePos += 1

                        if sequencePos >= len(sequence):
                            break
                        position = sequence[sequencePos]
                        print(cf.link_uri, ": Setting position {}".format(position))

                    # Desired position
                    initial = initialPositions[cf.link_uri]
                    x = position[0] + initial[0]
                    y = position[1] + initial[1]
                    z = position[2] + initial[2]

                    # Gain
                    K = 1

                    fly_to(cf, (x, y, z), K)
                    time.sleep(0.01)

            elif task == "follow":
                while True:
                    if tasks[uris[0]] == "landing":
                        break

                    initial_target = initialPositions[uris[0]]
                    pos_target = currentPositions[uris[0]]
                    work_position = set_work_dist[cf.link_uri]
                    x = pos_target[0] + work_position[0]
                    y = pos_target[1] + work_position[1]
                    z = pos_target[2] + work_position[2]

                    # Gain
                    K = 1

                    fly_to(cf, (x, y, z), K)
                    time.sleep(0.1)

            elif task == "follow2":
                while True:
                    if tasks[uris[0]] == "landing":
                        break

                    pos_target1 = currentPositions[uris[0]]
                    pos_target2 = currentPositions[uris[1]]
                    pos_target3 = currentPositions[uris[2]]

                    if cf.link_uri == uris[1]:
                        work_position = set_work_dist[cf.link_uri]
                        x = pos_target1[0] + work_position[0]
                        y = pos_target1[1] + work_position[1]
                        z = pos_target1[2] + work_position[2]
                        K = 1.3
                    elif cf.link_uri == uris[2]:
                        work_position = set_work_dist[cf.link_uri]
                        x = pos_target2[0] + work_position[0]
                        y = pos_target2[1] + work_position[1]
                        z = pos_target2[2] + work_position[2]
                        K = 1.6
                    elif cf.link_uri == uris[3]:
                        work_position = set_work_dist[cf.link_uri]
                        x = pos_target3[0] + work_position[0]
                        y = pos_target3[1] + work_position[1]
                        z = pos_target3[2] + work_position[2]
                        K = 2

                    fly_to(cf, (x, y, z), K)
                    time.sleep(0.1)

            elif task == "changes":
                while True:
                    if tasks[uris[0]] == "landing":
                        break

                    # time since the start
                    TimeSince = time.time() - TimeZero
                    if TimeSince > 30:
                        tasks[cf.link_uri] = "fix point"
                        task = tasks[cf.link_uri]
                        should_restart = True
                        break

                    pos_target = currentPositions[uris[0]]
                    work_position = set_work_dist[cf.link_uri]
                    x = pos_target[0] + work_position[0]
                    y = pos_target[1] + work_position[1]
                    z = pos_target[2] + work_position[2]
                    # Gain
                    K = 1
                    fly_to(cf, (x, y, z), K)
                    time.sleep(0.1)

            elif task == "parralel":
                while True:
                    if tasks[uris[0]] == "landing":
                        break

                    initial_self = initialPositions[cf.link_uri]
                    initial_target = initialPositions[uris[0]]
                    pos_target = currentPositions[uris[0]]
                    x = pos_target[0] + initial_target[0] + initial_self[0]
                    y = pos_target[1] + initial_target[1] + initial_self[1]
                    z = pos_target[2] + initial_target[2] + initial_self[2]
                    K = 1
                    fly_to(cf, (x, y, z), K)
                    time.sleep(0.1)

            elif task == "fix point":
                while True:
                    if tasks[uris[0]] == "landing":
                        break

                    position = side_mision[cf.link_uri]
                    x = position[0]
                    y = position[1]
                    z = position[2]
                    K = 1.5
                    fly_to(cf, (x, y, z), K)
                    time.sleep(0.01)

            else:
                print(cf.link_uri, ": Task [%s] not defined" % (task))

        tasks[cf.link_uri] = "landing"
        print(cf.link_uri, ": Landing")
        time_landing = (
            currentPositions[cf.link_uri][2] - initialPositions[cf.link_uri][2]
        ) / 0.1
        for i in range(math.floor(10 * time_landing)):
            cf.commander.send_velocity_world_setpoint(0, 0, -0.1, 0)
            time.sleep(0.1)
        cf.commander.send_stop_setpoint()
        # Make sure that the last packet leaves before the link is closed
        # since the message queue is not flushed before closing
        time.sleep(1.0)

        print(cf.link_uri, ": Finished task [%s]." % (task))
    except Exception as e:
        print(e)


if __name__ == "__main__":
    for uri in uris:
        logdata[uri] = {"x": [], "y": [], "z": []}

    crtp.init_drivers(enable_debug_driver=False)

    with Swarm(uris, factory=CachedCfFactory(rw_cache="./cache")) as swarm:
        # Set initial position and reset kalman filters
        swarm.parallel_safe(set_initial_position)
        swarm.parallel_safe(reset_estimator)
        # Wait for all crazyflies
        swarm.parallel_safe(wait_for_param_download)
        # Start position callback
        swarm.parallel_safe(start_position_callback)
        print("Starting in 3 seconds...")
        time.sleep(3)
        swarm.parallel_safe(run_task)
        print(tasks)
        print("END")

    # Plot
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    for uri in uris:
        ax.plot(logdata[uri]["x"], logdata[uri]["y"], logdata[uri]["z"])
    plt.show()