import os
import json
import numpy as np
from PIL import Image
from scipy import ndimage
import random
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from utils_3d import load_labels, parse_label, rgb_to_instance_index, read_exr, encode_np_image_to_base64

def process_area3(image_path, annot_path, xyz_path, depth_path, semantic_labels, model_name, client, output_path):
    image = Image.open(image_path)
    pano_image = np.array(image)
    h, w, _ = pano_image.shape
    annot_image = np.array(Image.open(annot_path))
    xyz_image = read_exr(xyz_path)
    depth_image = np.array(Image.open(depth_path))
    semantic_nums = len(semantic_labels)
    # map (r,g,b) form annot_image to instance_index
    instance_index = rgb_to_instance_index(annot_image)
    instance_ids = np.unique(instance_index)
    instance_ids = instance_ids[instance_ids < semantic_nums]
    # concentrate instances information
    instance_info = []
    for iid in instance_ids:
        label = parse_label(semantic_labels[int(iid)])
        cur_info = dict()
        cur_info['class_name'] = label['instance_class']
        # only choose core object
        if cur_info['class_name'] == '<UNK>' or cur_info['class_name'] == 'floor' or cur_info['class_name'] == 'ceiling' or cur_info['class_name'] == 'beam':
            continue
        cur_info['mask'] = (instance_index == iid)
        cur_info['points'] = xyz_image[cur_info['mask']]
        instance_info.append(cur_info)
    # get object position in grid
    update_instance_info_with_grid_coordinates(instance_info)
    # get camera map
    p_camera = estimate_camera_position_from_equirectangular(xyz_image, depth_image, pano_image)
    # get cognitive map
    id_map = dict()
    for inst_id in instance_ids:
        # map instance name in instance_id
        label = parse_label(semantic_labels[int(inst_id)])
        id_map[inst_id] = label["instance_class"]
    cog_map = generate_cognitive_map(instance_index, xyz_image, id_map, p_camera)
    # generate data
    basic_prompt = build_basic_prompt()
    image_id = image_path.split('/')[-1]
    reason_data, messages = multiturn_converstaion(cog_map, image_id, instance_info, model_name, client, turn=4, basic_prompt=basic_prompt)
    # save reason data and messages
    output = {
        "dataset": "area3",
        "image_id": image_id,
        "prompt": basic_prompt,
        "reasoning": reason_data,
    }
    # save the basic information of the data
    with open(os.path.join(output_path, f"area3_{image_id}_nag.json"), "w") as f:
        json.dump(output, f, indent=2)
    # save the conversation information
    with open(os.path.join(output_path, f"area3_{image_id}_nag_conv.json"), "w") as f:
        json.dump(messages, f, indent=2)
        
    return len(reason_data)
    
def multiturn_converstaion(cog_map, image_id, instance_info, model_name, client, turn, basic_prompt):
    if instance_info is None:
        return None, None
    non_wall_instances = [inst for inst in instance_info if inst["class_name"] != "wall"]
    if instance_info is None:
        return None, None
    messages = []
    reason_data = []
    messages.append({"role": "user", "content": basic_prompt})
    for i in range(turn):
        selected_instances = random.sample(non_wall_instances, 2)
        print(selected_instances[0].keys())
        start_point = selected_instances[0]["grid_coord"]
        end_point = selected_instances[1]["grid_coord"]
        start_class = selected_instances[0]["class_name"]
        end_class = selected_instances[1]["class_name"]
        cog_map_se = plot_cog_map(cog_map, start_point, end_point, image_id)
        base64_image = encode_np_image_to_base64(cog_map_se, image_format='.png')
        messages.append({
        "role": "user", "content": [
            {"type": "text", "text": f"The start class is {start_class}, the end class is {end_class}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]})
        # generate description for each object.
        object_description = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.8,
            max_tokens=2048,
        )
        messages.append({"role": "assistant", "content": object_description.choices[0].message.content})
        # construct data
        sample = dict()
        sample["question"] = "Starting at the yellow point and facing toward the camera, what is the path to reach the green point? You may only use the actions: move forward, turn left, turn right, and turn around."
        sample["object"] = f"Start Point:{ndimage.center_of_mass(selected_instances[0]['mask'])}, Class:{selected_instances[0]['class_name']}. End Point:{ndimage.center_of_mass(selected_instances[1]['mask'])}, Class:{selected_instances[1]['class_name']}."
        sample["grid_position"] = f"Start Point:{start_point}. End Point:{end_point}."
        sample["answer"] = object_description.choices[0].message.content
        reason_data.append(sample)  
    return reason_data, messages
        
    
def plot_cog_map(cog_map, start_point, end_point, image_id, grid_size=(20, 20)):
    cog_map[start_point[0]][start_point[1]].append("Start")
    cog_map[end_point[0]][end_point[1]].append("End") 
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, grid_size[1] + 1))
    ax.set_yticks(np.arange(0, grid_size[0] + 1))
    ax.grid(True)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            items = cog_map[i][j]
            if items:
                label = '\n'.join(sorted(set(items)))
                if "Camera" in label:
                    color = 'red'
                elif "Start" in label:
                    color = 'gold'
                elif "End" in label:
                    color = 'green'
                else:
                    color = 'black'
                ax.text(j + 0.5, i + 0.5, label, ha='center', va='center', fontsize=8, color=color)

    ax.set_title("Spatial Cognitive Map with Start and End", fontsize=16)
    plt.tight_layout()
    
    # save cognitive map and load it.
    fig.savefig(f"{image_id}_cognitive_map.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    img_array = np.array(Image.open(f"{image_id}_cognitive_map.png"))
    # remove this turn information
    cog_map[start_point[0]][start_point[1]].remove("Start")
    cog_map[end_point[0]][end_point[1]].remove("End") 
    
    return img_array
    
def build_basic_prompt():
    line = []
    line.append("In each round of dialogue, you will be given a spatial cognitive map corresponding to a panoramic image.")
    line.append("The cognitive map is a 20×20 grid. Movement is restricted to one grid at a time.")
    line.append("This map represents a top-down layout of a scene. The camera’s position is marked in red. ")
    line.append("The yellow marker indicates your starting point, and the green marker indicates your destination.")
    line.append("Object class information will be provided in the accompanying text.")
    line.append("Your task is to generate a simple path plan from the starting point to the destination.")
    line.append("Your initial position is at the starting point, facing toward the camera.")
    line.append("You may only use the following movement instructions to describe the path: move forward, turn left, turn right, and turn around.")
    line.append("Do not collide with any objects during the path.")
    line.append("Do not include any extra description. Only output the planned path.")
    return "\n".join(line)
    
def update_instance_info_with_grid_coordinates(instance_info, grid_size=(20, 20), min_points=400):
    """
    输入:
        instance_info: list[dict]，每个物体一个字典，包含 mask 和 3D points
        grid_size: (rows, cols)，认知图大小
        min_points: 少于该点数的物体将跳过 grid 坐标计算

    功能:
        在每个物体字典中添加 'grid_coord': (gy, gx)
    """
    centers_3d = []
    new_instance_info = []

    # 第一步：先提取所有有效中心点用于计算归一化范围
    for info in instance_info:
        points = info["points"]
        if points.shape[0] < min_points:
            continue
        center = np.median(points, axis=0)
        info["_center3d"] = center
        centers_3d.append(center)
        new_instance_info.append(info)
    # 只保留符合大小条件的目标
    instance_info[:] = new_instance_info

    if not centers_3d:
        raise ValueError("没有足够的实例参与 grid 坐标计算")

    # 第二步：计算 xz 范围用于归一化
    centers_3d = np.array(centers_3d)
    min_xz = np.min(centers_3d[:, [0, 2]], axis=0)
    max_xz = np.max(centers_3d[:, [0, 2]], axis=0)

    # 第三步：遍历每个物体，计算 grid 坐标
    for info in instance_info:
        if "_center3d" not in info:
            continue  # 被跳过的物体
        x, _, z = info["_center3d"]
        norm_x = (x - min_xz[0]) / (max_xz[0] - min_xz[0] + 1e-6)
        norm_z = (z - min_xz[1]) / (max_xz[1] - min_xz[1] + 1e-6)

        grid_x = min(int(norm_x * grid_size[1]), grid_size[1] - 1)
        grid_y = min(int(norm_z * grid_size[0]), grid_size[0] - 1)

        info["grid_coord"] = (grid_y, grid_x)

        # 可选：删除临时字段
        del info["_center3d"]

# loacate the camera position
def estimate_camera_position_from_equirectangular(
    xyz_image: np.ndarray,
    depth_map_raw: np.ndarray,
    rgb_image: np.ndarray
) -> np.ndarray:
    """
    从 equirectangular 图像、深度图和 3D 点图中计算相机位置。

    参数:
        xyz_image: (H, W, 3) 的 float32 数组，表示每个像素的 3D 世界坐标。
        depth_map_raw: (H, W) 的 uint16 数组，深度图，每个像素值代表 1/512 米。
        rgb_image: (H, W, 3) 的 uint8 数组，用于尺寸验证。

    返回:
        camera_position: (3,) numpy 数组，表示估计的相机在世界坐标中的位置。
    """
    H, W = rgb_image.shape[:2]

    # Step 1: 将原始深度图转换为 float32 单位米
    depth_map = depth_map_raw.astype(np.float32) / 512.0

    # Step 2: 生成单位方向向量（从相机指向像素方向）
    u = np.linspace(0, 1, W, endpoint=False)
    v = np.linspace(0, 1, H, endpoint=False)
    uu, vv = np.meshgrid(u, v)

    theta = 2 * np.pi * (uu - 0.5)  # 水平 [-π, π]
    phi = np.pi * (vv - 0.5)        # 垂直 [-π/2, π/2]

    ray_dirs = np.stack([
        np.cos(phi) * np.sin(theta),  # x
        np.sin(phi),                  # y
        np.cos(phi) * np.cos(theta)   # z
    ], axis=-1)  # shape: (H, W, 3)

    # Step 3: 创建有效像素掩码（排除无效深度和 NaN）
    valid = (~np.isnan(xyz_image).any(axis=-1)) & (depth_map_raw != 65535)
    valid_xyz = xyz_image[valid]
    valid_depth = depth_map[valid]
    valid_dirs = ray_dirs[valid]

    if len(valid_xyz) == 0:
        raise ValueError("没有有效像素用于估计相机位置，请检查输入。")

    # Step 4: 对每个像素使用反投影公式反推相机位置
    cam_candidates = valid_xyz - valid_depth[:, np.newaxis] * valid_dirs

    # Step 5: 取平均以增强鲁棒性
    camera_position = np.mean(cam_candidates, axis=0)

    return camera_position

def generate_cognitive_map(instance_mask, xyz_map, id_to_class, p_camera, grid_size=(20, 20), min_points=400):
    """
    参数:
        instance_mask: H x W 的 numpy 数组，实例分割图，每个像素是实例ID
        xyz_map: H x W x 3 的 numpy 数组，存储每个像素的世界坐标 (x, y, z)
        id_to_class: 字典 {instance_id: class_name}
        p_camera: 相机位置 (3,)
        grid_size: 网格大小 (rows, cols)
        min_points: 每个实例最少多少个像素点

    返回:
        显示空间认知图（以 x-z 平面为投影）
    """
    instance_positions = {}
    for instance_id in np.unique(instance_mask):
        if instance_id == 0:
            continue  # 忽略背景
        mask = instance_mask == instance_id
        coords = xyz_map[mask]
        coords = coords[~np.isnan(coords).any(axis=1)]
        if len(coords) < min_points:
            continue
        center = np.median(coords, axis=0)  # 得到物体中心
        instance_positions[instance_id] = np.array(center, dtype=np.float64).reshape(3,)
    
    # 加入相机坐标，id=-1
    # p_camera = np.array(p_camera, dtype=np.float64).reshape(3,)
    instance_positions[-1] = p_camera

    # 计算所有中心点在 XZ 平面的最小/最大值
    all_centers = np.array(list(instance_positions.values()))
    min_xz = np.min(all_centers[:, [0, 2]], axis=0)
    max_xz = np.max(all_centers[:, [0, 2]], axis=0)

    # 构建网格
    grid = [[[] for _ in range(grid_size[1])] for _ in range(grid_size[0])]

    for inst_id, center in instance_positions.items():
        x, z = center[0], center[2]
        norm_x = (x - min_xz[0]) / (max_xz[0] - min_xz[0] + 1e-6)
        norm_z = (z - min_xz[1]) / (max_xz[1] - min_xz[1] + 1e-6)

        grid_x = min(int(norm_x * grid_size[1]), grid_size[1] - 1)
        grid_y = min(int(norm_z * grid_size[0]), grid_size[0] - 1)

        label = id_to_class.get(inst_id, "Camera" if inst_id == -1 else f"id{inst_id}")
        # beam(横梁)\ceiling(天花板)\floor(地板)不纳入考虑
        if label == "beam" or label == "ceiling" or label == "floor":
            continue
        grid[grid_y][grid_x].append(label)
    return grid