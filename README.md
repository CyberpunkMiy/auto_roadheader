# Auto Roadheader RL Excavation

这是一个基于 MuJoCo 和强化学习的悬臂式掘进机仿真训练项目。项目目标是在截割头末端线速度固定的条件下，让智能体学习如何选择截割头运动方向，从而尽可能快、尽可能高效地清除整个体素化掘进区域，用来模拟山体/巷道截割过程。

## 项目内容

- 将掘进机 URDF/MJCF 模型、截割头 mesh 和体素目标区域整合到 MuJoCo 场景中。
- 使用 Gymnasium 环境封装掘进任务。
- 使用 Stable-Baselines3 PPO 训练 agent。
- 动作空间为截割头末端运动方向，环境内部保持固定步长，近似固定末端线速度。
- 体素清除由截割头 mesh 与目标体素空间关系计算。
- 支持无可视化后台训练、TensorBoard 监控、确定性评估、MuJoCo 可视化回放和录屏。

## 目录结构

```text
asset/                       纹理资源
config/                      模型配置
launch/                      ROS/Gazebo 相关启动文件
meshes/                      掘进机 STL mesh
src/
  assets/                    MuJoCo 基础 XML 资源
  control/                   IK 控制与交互控制
  core/                      场景合并、体素切削系统、仿真工具
  output/                    合并后的 MuJoCo 场景 XML
  rl_train/
    rl_env.py                强化学习环境
    train.py                 PPO 训练入口
    test_model.py            模型可视化测试入口
    models/                  最终筛选模型和结果说明
    scripts/                 评估、对比、可视化和录屏脚本
urdf/                        原始模型文件
```

## 环境

本项目使用 Conda 环境：

```powershell
conda activate urdf2mjcf_env
```

主要依赖：

- MuJoCo
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy / SciPy
- TensorBoard
- imageio / imageio-ffmpeg，用于录屏输出

如果使用命令行直接运行，推荐通过：

```powershell
conda run -n urdf2mjcf_env python ...
```

这样可以避免 Windows 下某些底层库加载差异。

## 最终模型

正式训练后评估多个 checkpoint，当前可复现的确定性最佳模型为：

```text
src/rl_train/models/roadheader_best_selected.zip
src/rl_train/models/vecnormalize_best_selected.pkl
```

评估结果记录在：

```text
src/rl_train/models/BEST_SELECTED_RESULT.txt
```

当前最佳确定性评估结果：

- 候选 checkpoint：`2400k`
- 最大步数：`2000`
- 剩余体素：`149 / 5313`
- 清除率：`97.1956%`
- 评估局数：`3`
- 全清除成功：`0 / 3`

训练过程中曾出现随机采样最好剩余 `132` 个体素，但确定性复现评估中 `2400k` checkpoint 最稳定。

## 可视化最佳模型

运行最佳模型并打开 MuJoCo viewer：

```powershell
conda run -n urdf2mjcf_env python src/rl_train/scripts/visualize_best_selected_policy.py --max-steps 2000 --render-pause 0.025 --spin-speed 28.0
```

半速播放：

```powershell
conda run -n urdf2mjcf_env python src/rl_train/scripts/visualize_best_selected_policy.py --max-steps 2000 --render-pause 0.05 --spin-speed 28.0
```

说明：训练时 agent 控制的是截割头末端运动方向；`spin-speed` 只用于可视化时给截割头增加自转效果，方便观察掘进机动作表现。

## 录屏

生成半速可视化评测视频：

```powershell
conda run -n urdf2mjcf_env python src/rl_train/scripts/record_best_selected_policy.py --max-steps 2000 --fps 20 --width 640 --height 480 --spin-speed 28.0
```

已生成的视频：

```text
src/rl_train/logs/best_selected_visual_eval_half_speed.mp4
```

## 后台训练

从零开始训练：

```powershell
cd src/rl_train
conda run -n urdf2mjcf_env python train.py --timesteps 3000000 --n-envs 4 --vec-env subproc --n-steps 1024 --batch-size 256 --save-freq 100000 --eval-freq 100000 --eval-episodes 1 --max-steps 2000 --control-mode kinematic --tensorboard
```

TensorBoard：

```powershell
conda run -n urdf2mjcf_env python -m tensorboard.main --logdir src/rl_train/logs --host 127.0.0.1 --port 6006
```

打开：

```text
http://127.0.0.1:6006
```

## 续训

训练脚本支持从 checkpoint 续训，并加载对应 VecNormalize：

```powershell
cd src/rl_train
conda run -n urdf2mjcf_env python train.py --resume models/roadheader_ppo_3000000_steps.zip --resume-vecnormalize models/roadheader_ppo_vecnormalize_3000000_steps.pkl --timesteps 1000000 --n-envs 4 --vec-env subproc --target-k 20 --learning-rate 0.00008 --tensorboard
```

后续如果要专门优化最后残余体素，可以尝试：

- 减小 `--target-k`，让目标点更贴近稀疏残余体素。
- 提高 `--empty-cut-penalty`，减少无效空切。
- 提高 `--completion-bonus` 和 `--early-finish-bonus-scale`，鼓励尽快全清除。

## 评估候选模型

对比多个候选 checkpoint：

```powershell
conda run -n urdf2mjcf_env python src/rl_train/scripts/compare_candidate_policies.py --episodes 3 --max-steps 2000
```

评估指定模型：

```powershell
conda run -n urdf2mjcf_env python src/rl_train/scripts/evaluate_checkpoint_policy.py --model src/rl_train/models/roadheader_best_selected.zip --vecnormalize src/rl_train/models/vecnormalize_best_selected.pkl --episodes 3 --max-steps 2000
```

## 备注

- 中间训练 checkpoint 和 TensorBoard 事件文件没有纳入 Git，只保留最终筛选模型。
- `src/output/merged_result.xml` 是当前训练和评测使用的 MuJoCo 场景。
- 当前任务尚未达到稳定全体素清除，但已经完成一个可运行、可视化、可复现评估的强化学习掘进机训练流程。
