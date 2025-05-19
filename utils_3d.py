import os
import re
import cv2
import json
import array
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import OpenEXR
import Imath
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import random
import base64
from io import BytesIO
from scipy.ndimage import center_of_mass
import psutil

# 定义 4 种颜色（RGB）
colors = [(255, 0, 0),    # red
            (0, 255, 0),    # green
            (0, 0, 255),    # blue
            (255, 255, 0)]  # yellow
colors_name = ['red', 'green', 'blue', 'yellow']

def process_area3(image_path, annot_path, xyz_path, semantic_labels, model_name, client, output_path):
    image = Image.open(image_path)
    np_image = np.array(image)
    h, w, _ = np_image.shape
    annot_image = np.array(Image.open(annot_path))
    xyz_image = read_exr(xyz_path)
    semantic_nums = len(semantic_labels)
    # print_mem()
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
        if cur_info['class_name'] == '<UNK>' or cur_info['class_name'] == 'wall' or cur_info['class_name'] == 'floor' or cur_info['class_name'] == 'ceiling':
            continue
        cur_info['mask'] = (instance_index == iid)
        # Nx3 area
        cur_info['points'] = xyz_image[cur_info['mask']]
        instance_info.append(cur_info)
    # multi-turn conversation
    basic_prompt = build_prompt_from_instances(h, w)
    reasoning_data, messages = multiturn_conversation(image_path, model_name, client, turn=4, basic_prompt=basic_prompt, instance_info=instance_info)
    image_id = image_path.split('/')[-1].split('.')[0]
    output = {
        "dataset": "area3",
        "image_id": image_id,
        "prompt": basic_prompt,
        "reasoning": reasoning_data,
    }
    # save the basic information of the data
    with open(os.path.join(output_path, f"area3_{image_id}.json"), "w") as f:
        json.dump(output, f, indent=2)
    # save the conversation information
    with open(os.path.join(output_path, f"area3_{image_id}_conversation.json"), "w") as f:
        json.dump(messages, f, indent=2)
        
    return len(reasoning_data)

def compute_center_3d(points):
    return np.mean(points, axis=0)

def multiturn_conversation(image_path, model_name, client, turn, basic_prompt, instance_info):
    # inference with point
    selected_instance = select_instance_groups_area3(instance_info, k=turn)
    points = [(center_of_mass(instance["2d_mask"])) for instance in selected_instance]
    image = mark_points_on_image(image_path, points, colors)
    # basic message
    # base64_image = encode_image_to_base64(image_path)
    base64_image = encode_np_image_to_base64(image, image_format='.png')
    messages = []
    messages.append({
        "role": "user", "content": [
            {"type": "text", "text": basic_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]})
    reasoning_data = []
    # multi-turn conversation
    for i in range(turn):
        # current object information to generate data
        cur_content = []
        # split the generation process to two part: position location + spatial reasoning
        cur_content.append(f"The object information is shown below. The key point of the object have been marked in the image with dots of {colors_name[i]} color.")
        # select objects for current turn conversation
        # selected_instance = select_instance_groups_area3(instance_info, k=1)
        for idx, cur_instance in enumerate(selected_instance):
            label = cur_instance["class_name"]
            center_3d = cur_instance["center"]
            cy, cx = center_of_mass(cur_instance["2d_mask"])
            cur_content.append(f"- Object: class = \"{label}\", 2D center(h,w) = ({int(cy), int(cx)}), 3D center(x,y,z) = ({int(center_3d[0])}, {int(center_3d[1])}, {int(center_3d[2])})")
        cur_content.append("Please generate a short, natural language description for the above object. Each description should be unique and clearly distinguishable.")
        cur_content.append("For clutter objects, infer a more specific category if possible.")
        cur_content = "\n".join(cur_content)
        messages.append({"role": "user", "content": cur_content})
        # generate description for each object.
        object_description = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.8,
            max_tokens=2048,
        )
        messages.append({"role": "assistant", "content": object_description.choices[0].message.content})
        # generate spatial reasoning data using generated descriptions
        messages.append({"role": "user", "content": "Now, using only the descriptions and 2d 3d location information above, generate a spatial reasoning problem that involve this object and explores its spatial relationships with other objects in the room. Do not include object IDs or coordinates. Use only the natural descriptions from above."})
        response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=2048,
                )
        # extract format data from response
        gpt_output = response.choices[0].message.content
        pattern = re.compile(
        r"Question:\s*\n(.*?)\n+"
        r"Reasoning Chain:\s*\n(.*?)\n+"
        r"Final Answer:\s*\n(.*?)(?=\n---|\Z)",
        re.DOTALL
        )
        for match in pattern.finditer(gpt_output):
            question = match.group(1).strip()
            reasoning_chain = match.group(2).strip()
            final_answer = match.group(3).strip()

            reasoning_data.append({
                "question": question,
                "reasoning_chain": reasoning_chain,
                "final_answer": final_answer
            })
        # update message
        messages.append({"role": "assistant", "content": gpt_output})
    return reasoning_data, messages

def build_prompt_from_instances(h, w):
    """
    build gpt prompt to generate spatial reasoning task.
    input: instance_info list, h, w
    output: prompt string
    """
    lines = [f"You are a language model tasked with generating spatial reasoning questions based on a panoramic scene with (h, w)={h,w}."]
    # lines.append("You are given the following object information:")
    lines.append("\nYour task is to generate complex spatial reasoning tasks using given object information, such as:")
    lines.append("- An object that satisfies multiple spatial constraints at once.")
    lines.append("- An object positioned between two others.")
    lines.append("- Comparison tasks involving sorting or ranking (by distance, height, etc.)")
    lines.append("- An object spatial position related to other objects.")
    lines.append("""The data format I need is a question, a chain of reasoning and the final answer. The format is shown below:
    Question:
    xxx
    Reasoning Chain:
    xxx
    Final Answer:
    xxx""")
    lines.append("Use natural and diverse language. Do not include coordinate values or measurements in your output.")

    return "\n".join(lines)

def mark_points_on_image(image_path: str, points: list, colors: list, radius: int = 3) -> np.ndarray:
    """
    mark key points on image.

    Args:
        image_path: string
        points (tuple): (y, x)
        colors: colors used in different points
        radius (int): radius

    Returns:
        np.ndarray: image with key points
    """
    image = np.array(Image.open(image_path))
    for i, point in enumerate(points):
        y, x = point
        cv2.circle(image, (int(x), int(y)), radius, colors[i], thickness=-1)

    return image

def select_instance_groups_area3(instance_info, k=15, min_points=100):
    """
    select instance groups for area3.
    select k objects from instance_info and claculate central-point.
    each object contains at least min_points.
    return element list in instance_info.
    """
    # extract available objects
    valid_instances = [
        {
            "id": idx,
            "class_name": inst["class_name"],
            "2d_mask": inst["mask"],
            "center": compute_center_3d(inst["points"]),
        }
        for idx, inst in enumerate(instance_info)
        if inst["points"].shape[0] >= min_points
    ]

    if len(valid_instances) < k:
        return None

    selected = random.sample(valid_instances, k)
    return selected

""" Label functions """
def load_labels( label_file ):
    """ Convenience function for loading JSON labels """
    with open( label_file ) as f:
        return json.load( f )

def parse_label( label ):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split( "_" )
    res[ 'instance_class' ] = clazz
    res[ 'instance_num' ] = int( instance_num )
    res[ 'room_type' ] = room_type
    res[ 'room_num' ] = int( room_num )
    res[ 'area_num' ] = int( area_num )
    return res

def get_index(color):
    """ Parse a color as a base-256 number and return the index """
    r, g, b = map(int, color)  # 👈 转换为 Python int 类型，避免 uint8 溢出
    return r * 256 * 256 + g * 256 + b

def rgb_to_instance_index(rgb_image):
    """
    将 (H, W, 3) 的 RGB 实例图转换为 (H, W) 的整数实例ID图。
    """
    rgb_image = rgb_image.astype(np.uint32)  # 防止 uint8 溢出
    return (rgb_image[:, :, 0] << 16) + (rgb_image[:, :, 1] << 8) + rgb_image[:, :, 2]

# 读取exr文件
def read_exr( image_fpath ):
    """ Reads an openEXR file into an RGB matrix with floats """
    f = OpenEXR.InputFile( image_fpath )
    dw = f.header()['dataWindow']
    w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)    
    im = np.empty( (h, w, 3) )

    # Read in the EXR
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = f.channels( ["R", "G", "B"], FLOAT )
    for i, channel in enumerate( channels ):
        im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
    return im

# 可视化3D场景
def plot_xyz_pointcloud(xyz_map, step=10):
    h, w, _ = xyz_map.shape
    pts = xyz_map[::step, ::step].reshape(-1, 3)  # 降采样
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=pts[:, 2], cmap='viridis')
    ax.set_title("3D Point Cloud from Global XYZ")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    plt.show()
    
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def encode_pil_image_to_base64(pil_image, image_format="PNG"):
    buffer = BytesIO()
    pil_image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def encode_np_image_to_base64(np_image, image_format=".jpg"):
    # 将 numpy 图像编码为内存中的图像二进制
    success, encoded_image = cv2.imencode(image_format, np_image)
    if not success:
        raise ValueError("Image encoding failed")
    
    # 转为 base64 字符串
    return base64.b64encode(encoded_image.tobytes()).decode("utf-8")
    
def print_mem():
    pid = os.getpid()
    py = psutil.Process(pid)
    mem = py.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {mem:.2f} MB")