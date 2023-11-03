#!/usr/bin/env python3
import asyncio
import numpy as np
import time
import matplotlib.pyplot as plt
from mavsdk import System
from mavsdk.offboard import (OffboardError,PositionNedYaw,VelocityNedYaw)
# Constants
KC = 30
KA = 50
DT = 0.02

# Initialize global storage for wc and wa
global_wc = []
global_wa = []

def initialize_weights_c():
    # return np.zeros((11, 3))
    return np.full((11, 3), 0.5)
def initialize_weights_a():
    # return np.zeros((11, 3))
    return np.full((11, 3), 0.8)
wc1 = initialize_weights_c()
wa = initialize_weights_a()
# [1, 0.8, 0.6, , 1, 0, -1, -2, -3, -4, -5],
        # [0.2, 0.15, 0.1, 0.05, 0.03, 0, -0.03, -0.05, -0.1, -0.15, -0.2],
        # [3, 2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2, -3],
        # [2, 1.5, 1, 0.5, 0.3, 0, -0.3, -0.5, -1, -1.5, -2],
        # [0.2, 0.15, 0.1, 0.05, 0.03, 0, -0.03, -0.05, -0.1, -0.15, -0.2],
        # [0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5]
async def critic_nn(exi):
    """Neural network for the critic."""
    d = np.array([
        np.arange(1,-1,0.1),
        np.arange(1,-1,0.1),
        np.arange(1,-1,0.1),
        np.arange(1,-1,0.1),
        np.arange(1,-1,0.1),
        np.arange(1,-1,0.1)
    ])

    
    s1 = np.array([np.exp(-np.linalg.norm(exi - d[:, i]) ** 2 / 2) for i in range(11)])
    wc1 = wc1 - KC * (np.outer(s1, s1)) @ wc1 * DT
    sr = s1

    wa = wa - np.outer(sr, sr) @ (KA * (wa - wc1) + KC * wc1) * DT
    wt = -0.5 * np.dot(wa.T, s1)
    # Append current wc and wa to the global lists
    global_wc.append(wc1.flatten())
    global_wa.append(wa.flatten())
    return wt[0], wt[1], wt[2]



async def control(drone,PositionNedYaw, initial_position, exi):
    pd=PositionNedYaw
    
    kpx=0.5
    kpy=0.8
    kpz=0.3#observer gain
    start_time = time.time()  # 记录控制开始时间

    async for p in drone.telemetry.position_velocity_ned():
  
        x=p.position.north_m
        y=p.position.east_m
        z=p.position.down_m

        # 计算当前位置与初始位置之间的距离and计算当前位置与目标位置之间的距离
        target_position = [pd.north_m, pd.east_m, pd.down_m]
        distance_to_initial = np.linalg.norm(np.array([x, y, z]) - np.array(initial_position))
        # distance_to_target = np.linalg.norm(np.array([x, y, z]) - np.array(target_position))
        distance_to_x = pd.north_m-x
        # 如果距离目标点小于一定阈值，启动计时器
        if distance_to_x < 0.5 :
            print("success")
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, pd.yaw_deg))
            return


        # 如果距离超过5米，触发退出并降落
        if distance_to_initial > 5.0:
            print("Drone is too far from the initial position. Exiting and landing...")
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, pd.yaw_deg))
            await drone.action.land()
            return
        ex = pd.north_m-x
        ey = pd.east_m-y
        ez = pd.down_m-z
        combin_vector = np.column_stack((ex,ey,ez))
        combin_state = np.column_stack((x,y,z))
        exi = np.column_stack((combin_state, combin_vector))
        nnx, nny, nnz = await critic_nn(exi)
        # nnx = max(-0.1, min(0.1, nnx))
        # nny = max(-0.1, min(0.1, nny))
        # nnz = max(-0.1, min(0.1, nnz))
        #position control
        ux=kpx*ex + nnx
        uy=kpy*ey + nny
        uz=kpz*ez + nnz
        ux = max(-0.2,min(0.2,ux))
        uy = max(-0.2,min(0.2,uy))
        uz = max(-0.2,min(0.2,uz))

        await drone.offboard.set_velocity_ned(VelocityNedYaw(ux, uy, uz, pd.yaw_deg))


async def run():
    """ Does Offboard control using velocity NED coordinates. """

    drone = System()
    await drone.connect(system_address="serial:///dev/ttyUSB0:921600")
    # await drone.connect(system_address="udp://:921600s")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break
    
    await drone.telemetry.set_rate_position_velocity_ned(50)
    await asyncio.sleep(1)


    print("-- Arming")
    await drone.action.arm()

    async for p_v_ini in drone.telemetry.position_velocity_ned():
        x0=p_v_ini.position.north_m
        y0=p_v_ini.position.east_m
        z0=p_v_ini.position.down_m
        break
    
    async for yaw_ini in drone.telemetry.heading():
        yaw0=yaw_ini.heading_deg
        break
    initial_position = [x0, y0, z0]
    print("-- Setting initial setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(x0,y0,z0,yaw0))
    print(initial_position)

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: \
              {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    await drone.offboard.set_position_ned(
            PositionNedYaw(x0,y0,z0-3.0,yaw0))
    await asyncio.sleep(10)
    print("takeoff finish")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw0))
    print("start control")
    exi = np.zeros((6, 1))
    await control(drone,PositionNedYaw(x0+2.0,y0+0,z0-3.0,yaw0+0),initial_position,exi)
    await asyncio.sleep(4)
    print("over")
    # # Calculate and print the 2-norms after control loop ends
    # wc_norm = np.linalg.norm(np.array(global_wc))
    # wa_norm = np.linalg.norm(np.array(global_wa))
    # print(f"wc Norm: {wc_norm}, wa Norm: {wa_norm}")

    global_wc_array = np.array(global_wc)
    global_wa_array = np.array(global_wa)
    
    wc_norm = np.linalg.norm(global_wc_array, axis=1)
    wa_norm = np.linalg.norm(global_wa_array, axis=1)
    # 绘制趋势图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(wc_norm)
    plt.title('Trend of wc Norms')
    plt.xlabel('Step')
    plt.ylabel('2-Norm')

    plt.subplot(1, 2, 2)
    plt.plot(wa_norm)
    plt.title('Trend of wa Norms')
    plt.xlabel('Step')
    plt.ylabel('2-Norm')

    plt.tight_layout()
    plt.show()
    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed with error code: \
              {error._result.result}")

    # Land the drone
    print("-- Landing")
    await drone.action.land()
    await asyncio.sleep(5)

    
if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())
