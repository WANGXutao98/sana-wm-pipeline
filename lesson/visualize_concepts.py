#!/usr/bin/env python3
"""可视化脚本：帮助理解视频归一化和相机几何的核心概念

运行方式：
    python visualize_concepts.py

需要的依赖：
    pip install numpy matplotlib opencv-python
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import cv2


def demo1_video_normalization():
    """演示1：视频归一化过程（缩放+裁剪）"""
    print("\n" + "="*60)
    print("演示1: 视频归一化 - 缩放与裁剪")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始视频
    original_w, original_h = 1920, 1080
    target_w, target_h = 1280, 720

    # 计算缩放比例（force_original_aspect_ratio=increase）
    scale_x = target_w / original_w
    scale_y = target_h / original_h
    scale = max(scale_x, scale_y)  # 选择更大的比例，确保覆盖目标尺寸

    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)

    # 计算裁剪偏移（center crop）
    crop_x = (scaled_w - target_w) // 2
    crop_y = (scaled_h - target_h) // 2

    print(f"\n原始尺寸: {original_w}x{original_h}")
    print(f"目标尺寸: {target_w}x{target_h}")
    print(f"缩放比例: {scale:.3f}")
    print(f"缩放后尺寸: {scaled_w}x{scaled_h}")
    print(f"裁剪偏移: x={crop_x}, y={crop_y}")

    # 1. 原始视频
    ax1 = axes[0]
    ax1.add_patch(Rectangle((0, 0), original_w, original_h,
                             fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax1.text(original_w/2, original_h/2, f'{original_w}x{original_h}\n原始视频',
             ha='center', va='center', fontsize=12, weight='bold')
    ax1.set_xlim(-100, 2100)
    ax1.set_ylim(-100, 1400)
    ax1.set_aspect('equal')
    ax1.set_title('步骤1: 原始视频', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. 缩放后
    ax2 = axes[1]
    ax2.add_patch(Rectangle((0, 0), scaled_w, scaled_h,
                             fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2))
    # 绘制目标框
    ax2.add_patch(Rectangle((crop_x, crop_y), target_w, target_h,
                             fill=False, edgecolor='red', linewidth=3, linestyle='--'))
    ax2.text(scaled_w/2, scaled_h/2, f'{scaled_w}x{scaled_h}\n缩放后',
             ha='center', va='center', fontsize=12, weight='bold')
    ax2.text(scaled_w/2, scaled_h/2 - 100, '红色虚线=裁剪区域',
             ha='center', va='center', fontsize=10, color='red')
    ax2.set_xlim(-100, 2100)
    ax2.set_ylim(-100, 1400)
    ax2.set_aspect('equal')
    ax2.set_title(f'步骤2: 缩放 (x{scale:.2f})', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 裁剪后
    ax3 = axes[2]
    ax3.add_patch(Rectangle((0, 0), target_w, target_h,
                             fill=True, facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax3.text(target_w/2, target_h/2, f'{target_w}x{target_h}\n最终输出',
             ha='center', va='center', fontsize=12, weight='bold')
    ax3.set_xlim(-100, 1500)
    ax3.set_ylim(-100, 900)
    ax3.set_aspect('equal')
    ax3.set_title('步骤3: Center Crop', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/afs/davidwang/workspace/sana_wm_pipeline/lesson/demo1_normalization.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 图片已保存: lesson/demo1_normalization.png")
    plt.show()


def demo2_camera_intrinsics():
    """演示2：相机内参的影响"""
    print("\n" + "="*60)
    print("演示2: 相机内参矩阵")
    print("="*60)

    # 创建一个简单的3D场景
    fig = plt.figure(figsize=(15, 5))

    # 3D点（世界坐标系）
    points_3d = np.array([
        [0, 0, 5],
        [1, 0, 5],
        [0, 1, 5],
        [1, 1, 5],
        [0.5, 0.5, 5],
    ])

    # 不同的相机内参
    intrinsics_list = [
        {'fx': 800, 'fy': 800, 'cx': 640, 'cy': 360, 'name': '标准镜头'},
        {'fx': 1200, 'fy': 1200, 'cx': 640, 'cy': 360, 'name': '长焦镜头 (fx↑)'},
        {'fx': 400, 'fy': 400, 'cx': 640, 'cy': 360, 'name': '广角镜头 (fx↓)'},
    ]

    for idx, K_params in enumerate(intrinsics_list):
        ax = fig.add_subplot(1, 3, idx+1)

        # 构建内参矩阵
        K = np.array([
            [K_params['fx'], 0, K_params['cx']],
            [0, K_params['fy'], K_params['cy']],
            [0, 0, 1]
        ])

        print(f"\n{K_params['name']}:")
        print(f"K = \n{K}")

        # 投影到2D
        points_2d = []
        for p3d in points_3d:
            p_cam = p3d  # 假设相机坐标=世界坐标
            p_img_homo = K @ p_cam
            p_img = p_img_homo[:2] / p_img_homo[2]  # 透视除法
            points_2d.append(p_img)

        points_2d = np.array(points_2d)

        # 绘制投影结果
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=100, c='red', marker='o')
        ax.plot([points_2d[0, 0], points_2d[1, 0]],
                [points_2d[0, 1], points_2d[1, 1]], 'b-', linewidth=2)
        ax.plot([points_2d[0, 0], points_2d[2, 0]],
                [points_2d[0, 1], points_2d[2, 1]], 'b-', linewidth=2)

        # 绘制主点
        ax.scatter(K_params['cx'], K_params['cy'], s=200, c='green', marker='x', linewidth=3)
        ax.text(K_params['cx']+50, K_params['cy']+50, '主点(cx,cy)', fontsize=10, color='green')

        # 绘制图像边界
        ax.add_patch(Rectangle((0, 0), 1280, 720, fill=False, edgecolor='gray', linestyle='--'))

        ax.set_xlim(-100, 1380)
        ax.set_ylim(820, -100)  # 翻转Y轴（图像坐标系）
        ax.set_aspect('equal')
        ax.set_title(f'{K_params["name"]}\nfx={K_params["fx"]}', fontsize=12, weight='bold')
        ax.set_xlabel('u (像素)', fontsize=10)
        ax.set_ylabel('v (像素)', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/afs/davidwang/workspace/sana_wm_pipeline/lesson/demo2_intrinsics.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 图片已保存: lesson/demo2_intrinsics.png")
    plt.show()


def demo3_camera_pose():
    """演示3：相机位姿（外参）"""
    print("\n" + "="*60)
    print("演示3: 相机位姿 - c2w vs w2c")
    print("="*60)

    fig = plt.figure(figsize=(12, 10))

    # 世界坐标系中的3D点
    cube_points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 底面
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # 顶面
    ])

    # 定义两个相机位姿
    cameras = [
        {
            'position': np.array([3, 3, 2]),
            'look_at': np.array([0.5, 0.5, 0.5]),
            'name': '相机1',
            'color': 'red'
        },
        {
            'position': np.array([-2, 2, 2]),
            'look_at': np.array([0.5, 0.5, 0.5]),
            'name': '相机2',
            'color': 'blue'
        }
    ]

    ax = fig.add_subplot(111, projection='3d')

    # 绘制立方体
    for i in range(4):
        # 底面边
        p1, p2 = cube_points[i], cube_points[(i+1)%4]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=2)
        # 顶面边
        p1, p2 = cube_points[i+4], cube_points[(i+1)%4+4]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=2)
        # 竖边
        ax.plot([cube_points[i, 0], cube_points[i+4, 0]],
                [cube_points[i, 1], cube_points[i+4, 1]],
                [cube_points[i, 2], cube_points[i+4, 2]], 'k-', linewidth=2)

    # 绘制世界坐标系
    origin = np.array([0, 0, 0])
    ax.quiver(origin[0], origin[1], origin[2], 1.5, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
    ax.quiver(origin[0], origin[1], origin[2], 0, 1.5, 0, color='green', arrow_length_ratio=0.2, linewidth=2)
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1.5, color='blue', arrow_length_ratio=0.2, linewidth=2)
    ax.text(1.5, 0, 0, 'X', fontsize=12, weight='bold', color='red')
    ax.text(0, 1.5, 0, 'Y', fontsize=12, weight='bold', color='green')
    ax.text(0, 0, 1.5, 'Z', fontsize=12, weight='bold', color='blue')

    # 绘制相机
    for cam in cameras:
        pos = cam['position']
        look_at = cam['look_at']

        # 相机位置
        ax.scatter(pos[0], pos[1], pos[2], s=200, c=cam['color'], marker='o')
        ax.text(pos[0], pos[1], pos[2]+0.3, cam['name'], fontsize=12, weight='bold', color=cam['color'])

        # 相机朝向
        direction = look_at - pos
        direction = direction / np.linalg.norm(direction) * 0.8
        ax.quiver(pos[0], pos[1], pos[2],
                 direction[0], direction[1], direction[2],
                 color=cam['color'], arrow_length_ratio=0.3, linewidth=2)

        # 计算并打印c2w矩阵
        forward = look_at - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        R_c2w = np.column_stack([right, up, -forward])
        t_c2w = pos

        print(f"\n{cam['name']} 位姿:")
        print(f"位置: {pos}")
        print(f"朝向: {look_at}")
        print(f"c2w 旋转矩阵 R:\n{R_c2w}")
        print(f"c2w 平移向量 t: {t_c2w}")

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('相机位姿示意图\n（红/蓝球=相机位置，箭头=朝向）', fontsize=14, weight='bold')
    ax.set_box_aspect([1,1,1])

    plt.savefig('/mnt/afs/davidwang/workspace/sana_wm_pipeline/lesson/demo3_camera_pose.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 图片已保存: lesson/demo3_camera_pose.png")
    plt.show()


def demo4_crop_effect_on_intrinsics():
    """演示4：裁剪对相机内参的影响"""
    print("\n" + "="*60)
    print("演示4: Center Crop 如何影响相机内参")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 原始内参
    original_w, original_h = 1920, 1080
    fx_orig, fy_orig = 1000, 1000
    cx_orig, cy_orig = original_w / 2, original_h / 2

    # 缩放和裁剪参数
    target_w, target_h = 1280, 720
    scale = max(target_w / original_w, target_h / original_h)
    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)
    crop_x = (scaled_w - target_w) // 2
    crop_y = (scaled_h - target_h) // 2

    # 新内参
    fx_new = fx_orig * scale
    fy_new = fy_orig * scale
    cx_new = cx_orig * scale - crop_x
    cy_new = cy_orig * scale - crop_y

    print(f"\n原始内参:")
    print(f"  fx={fx_orig}, fy={fy_orig}")
    print(f"  cx={cx_orig}, cy={cy_orig}")
    print(f"  分辨率: {original_w}x{original_h}")

    print(f"\n经过 scale={scale:.3f} + crop=({crop_x},{crop_y}) 后:")
    print(f"  fx={fx_new:.1f}, fy={fy_new:.1f}")
    print(f"  cx={cx_new:.1f}, cy={cy_new:.1f}")
    print(f"  分辨率: {target_w}x{target_h}")

    # 左图：原始
    ax1 = axes[0]
    ax1.add_patch(Rectangle((0, 0), original_w, original_h,
                             fill=False, edgecolor='blue', linewidth=2))
    ax1.scatter(cx_orig, cy_orig, s=300, c='red', marker='+', linewidth=3)
    ax1.text(cx_orig+100, cy_orig+50, f'主点\n({cx_orig:.0f}, {cy_orig:.0f})',
             fontsize=11, weight='bold', color='red')

    # 绘制视场角示意
    fov_lines = [
        [[cx_orig, cy_orig], [0, 0]],
        [[cx_orig, cy_orig], [original_w, 0]],
        [[cx_orig, cy_orig], [0, original_h]],
        [[cx_orig, cy_orig], [original_w, original_h]],
    ]
    for line in fov_lines:
        ax1.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                'g--', alpha=0.5, linewidth=1)

    ax1.set_xlim(-100, 2100)
    ax1.set_ylim(1200, -100)
    ax1.set_aspect('equal')
    ax1.set_title(f'原始图像\n{original_w}x{original_h}, fx={fx_orig}',
                  fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('u (像素)', fontsize=10)
    ax1.set_ylabel('v (像素)', fontsize=10)

    # 右图：裁剪后
    ax2 = axes[1]
    ax2.add_patch(Rectangle((0, 0), target_w, target_h,
                             fill=False, edgecolor='orange', linewidth=2))
    ax2.scatter(cx_new, cy_new, s=300, c='red', marker='+', linewidth=3)
    ax2.text(cx_new+80, cy_new+40, f'主点\n({cx_new:.0f}, {cy_new:.0f})',
             fontsize=11, weight='bold', color='red')

    # 绘制视场角示意（缩小了）
    fov_lines_new = [
        [[cx_new, cy_new], [0, 0]],
        [[cx_new, cy_new], [target_w, 0]],
        [[cx_new, cy_new], [0, target_h]],
        [[cx_new, cy_new], [target_w, target_h]],
    ]
    for line in fov_lines_new:
        ax2.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                'g--', alpha=0.5, linewidth=1)

    ax2.set_xlim(-100, 1500)
    ax2.set_ylim(850, -100)
    ax2.set_aspect('equal')
    ax2.set_title(f'Crop后图像\n{target_w}x{target_h}, fx={fx_new:.0f}',
                  fontsize=13, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('u (像素)', fontsize=10)
    ax2.set_ylabel('v (像素)', fontsize=10)

    plt.tight_layout()
    plt.savefig('/mnt/afs/davidwang/workspace/sana_wm_pipeline/lesson/demo4_crop_intrinsics.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 图片已保存: lesson/demo4_crop_intrinsics.png")
    plt.show()


def main():
    """运行所有演示"""
    print("\n" + "="*60)
    print("视频处理与相机几何 - 交互式可视化")
    print("="*60)
    print("\n本脚本将生成4组可视化图片，帮助理解:")
    print("  1. 视频归一化过程（缩放+裁剪）")
    print("  2. 相机内参矩阵的作用")
    print("  3. 相机位姿（c2w/w2c）")
    print("  4. 裁剪对内参的影响")

    try:
        demo1_video_normalization()
        demo2_camera_intrinsics()
        demo3_camera_pose()
        demo4_crop_effect_on_intrinsics()

        print("\n" + "="*60)
        print("✅ 所有演示完成！")
        print("="*60)
        print("\n生成的图片:")
        print("  - lesson/demo1_normalization.png")
        print("  - lesson/demo2_intrinsics.png")
        print("  - lesson/demo3_camera_pose.png")
        print("  - lesson/demo4_crop_intrinsics.png")
        print("\n请查看这些图片，配合教程文档学习！")

    except Exception as e:
        print(f"\n❌ 出错了: {e}")
        print("请确保已安装: numpy matplotlib opencv-python")
        print("安装命令: pip install numpy matplotlib opencv-python")


if __name__ == '__main__':
    main()
