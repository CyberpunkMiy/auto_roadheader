import time
import mujoco
import mujoco.viewer
from digging_system_mesh import MeshDiggingSystem

def main():
    xml_path = "../output/merged_result0.xml"
    # xml_path = "../assets/robot.xml"
    print(f"🚀 正在加载模型: {xml_path}")
    
    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 2. 初始化挖掘系统
    # ==========================================
    # 👇 TODO: 请在这里修改为你 Robot 真实的命名 👇
    # ==========================================
    YOUR_CUTTER_MESH_NAME = "jiegetou_link"    # 对应 <mesh name="...">
    YOUR_CUTTER_BODY_NAME = "jiegetou_link"    # 对应绑定该 mesh 的 <body name="...">
    
    print("🔧 初始化挖掘系统...")
    try:
        digging = MeshDiggingSystem(
            model, 
            data, 
            mesh_name=YOUR_CUTTER_MESH_NAME,
            scene_body_name="voxel_target", # 自动寻找我们在合并时创建的容器
            clean_threshold=[3.0, 0.65, 0.65]
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("请检查 run_simulation.py 中的 YOUR_CUTTER_MESH_NAME 是否正确！")
        return

    # 3. 启动 Viewer
    print("🎥 启动模拟器...")
    print("💡 提示：在右侧菜单的 'Joints' 栏手动拖动滑块，控制机器人去撞击墙壁！")
    
    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     # 设置一个稍微远一点的视角以便观察
    #     viewer.cam.lookat[:] = [1.0, 0, 1.0]
    #     viewer.cam.distance = 5.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- 重新配置摄像机最佳交互视角 ---
        viewer.cam.lookat[:] = [1.2, 1.6, 1.2]  # 视点中心偏向 [1.2, 1.6, 1.2]，聚焦在截割头与墙体交互的特定区域
        viewer.cam.distance = 13.0               # 将距离拉远到 13.0，获得更广阔的全局视野，确保庞大的机身和墙体完整入画
        viewer.cam.azimuth = 50               # 方位角设为 50 度，提供一个从右侧偏前方观察的视角
        viewer.cam.elevation = -25              # 俯仰角设为 -25 度，微微向下俯视，清晰观察空间三维挖掘动作
        
        while viewer.is_running():
            step_start = time.time()

            # --- 核心逻辑：执行挖掘 ---
            # 这行代码会检测截割头是否碰到了体素，如果碰到，体素就会消失
            reward = digging.perform_cutting(YOUR_CUTTER_BODY_NAME)
            
            if reward > 0:
                print(f"⛏️ 挖掘中! 消除体素数量: {reward}")

            # --- 物理步进 ---
            mujoco.mj_step(model, data)
            viewer.sync()

            # 保持实时帧率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()