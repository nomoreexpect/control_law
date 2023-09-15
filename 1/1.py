#!/usr/bin/env python3

import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityNedYaw,PositionNedYaw)

# Constants
KC = 20
KA = 30
DT = 0.02

def initialize_weights():
    return np.zeros((11, 3))


async def telemetry_own_position(drone):
    """Retrieve the drone's own position."""
    async for ned in drone.telemetry.position_velocity_ned():
        return ned.position.north_m, ned.position.east_m, ned.position.down_m


async def telemetry_own_heading(drone):
    """Retrieve the drone's own yaw."""
    async for head in drone.telemetry.heading():
        return head.heading_deg


async def critic_nn(exi):
    """Neural network for the critic."""
    d = np.array([
        [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
        [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
        [30, 20, 10, 5, 1, 0, -1, -5, -10, -20, -30],
        [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
        [10, 8, 5, 2, 1, 0, -1, -2, -5, -8, -10],
        [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]
    ])

    wc1 = initialize_weights()
    s1 = np.array([np.exp(-np.linalg.norm(exi - d[:, i]) ** 2 / 2) for i in range(11)])
    wc1_k = wc1 - KC * (np.outer(s1, s1)) @ wc1 * DT
    sr = s1

    wa = initialize_weights()
    wa_k = wa - np.outer(sr, sr) @ (KA * (wa - wc1_k) + KC * wc1_k) * DT
    wt = -0.5 * np.dot(wa_k.T, s1)
    return wt[0], wt[1], wt[2]


async def compute_errors(drone, target_north, target_east, target_altitude):
    """Compute position errors."""
    x0, y0, z0 = await telemetry_own_position(drone)
    xd = target_north - x0
    yd = target_east - y0
    zd = target_altitude - z0
    return xd, yd, zd


async def control_drone(drone, position, exi):
    """Control the drone based on target positions and tracking error."""
    target_north, target_east, target_altitude, target_yaw = position
  
    while True:
        north_error, east_error, altitude_error = await compute_errors(drone, target_north, target_east,
                                                                       target_altitude)

        combin_vector = np.column_stack((north_error, east_error, altitude_error))
        combin_state = np.column_stack((await telemetry_own_position(drone)))

        exi = np.column_stack((combin_state, combin_vector))
        yaw_error = target_yaw - await telemetry_own_heading(drone)
        nnx, nny, nnz = await critic_nn(exi)
        # Apply limits to nnx, nny, and nnz
        nnx = max(-0.1, min(0.1, nnx))
        nny = max(-0.1, min(0.1, nny))
        nnz = max(-0.1, min(0.1, nnz))

        vx = 0.5 * north_error + nnx
        vx = max(-0.3, min(0.3, vx))

        await drone.offboard.set_velocity_ned(VelocityNedYaw(
            vx,
            0.8 * east_error + nny,
            0.3 * altitude_error + nnz,
            yaw_error
        ))

        # Break loop if drone is close enough to target
        # if all(abs(err) < 0.1 for err in [altitude_error, north_error, east_error]):
        #     break
        if abs(north_error) < 0.3 and abs(east_error) < 0.3 and abs(altitude_error) < 0.4:
            break


# ... (The rest of the code remains largely unchanged)

async def main():
    """Main control loop."""
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyUSB0:921600")

    # Establish connection
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    # Ensure global position estimate is available
    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    # Arm the drone
    print("-- Arming")
    try:
        await drone.action.arm()
    except Exception as e:
        print(f"Failed to arm the drone: {e}")

    # Set initial setpoint
    print("-- Setting initial setpoint")
    initial_yaw = await telemetry_own_heading(drone)
    x_initial, y_initial, z_initial = await telemetry_own_position(drone)
    print("1")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0,0.0, 0.0))
    print("2")
    #z_error = abs(z_initial-2)
        #if z_error < 0.3:
            #print("--Ready!")
            #break

    # Start offboard mode
    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    #take off
    print("--Go up 2 m")
    await drone.offboard.set_position_ned(
            PositionNedYaw(x_initial, y_initial,z_initial -2.0, initial_yaw))
    await asyncio.sleep(10)
    print("--Ready,next!")
#Define target positions
    target_positions = [
        (x_initial, y_initial, z_initial-2.0 , initial_yaw+30.0),
        (x_initial, y_initial, z_initial-2.0 , initial_yaw),
        #(x_initial, y_initial, z_initial-2.0 , initial_yaw),
        (x_initial + 2.0, y_initial, z_initial - 2.0, initial_yaw)
    ]
#     relative_positions = [
#             (0.0, 0.0, -3.0, 0.0),
#             (0.0, 0.0, -3.0, 90.0),
#             (0.0, 0.0, -3.0, 0.0),
#             (0.0, 0.0, -3.0, -90.0),
#             (0.0, 0.0, -3.0, 0.0),
#             (3.0, 0.0, -3.0, 0.0),
#         ]
    exi = np.zeros((6, 1))
#     positions = [PositionNedYaw(x_initial + rel_n, y_initial + rel_e,
#                                 z_initial + rel_d, initial_yaw + rel_y)
#                 for rel_n, rel_e, rel_d, rel_y in relative_positions]
#     for relative_positions,position in zip(relative_positions, positions):

            # positions = await telemetry_own_position(drone)
            # yaw = await telemetry_own_heading(drone)

    # Control the drone towards target positions
    for target in target_positions:
        await control_drone(drone, target, exi)
        await asyncio.sleep(5)

    # Stop offboard mode
    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed with error code: {error._result.result}")

    # Land the drone
    print("-- Landing")
    await drone.action.land()
    await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
