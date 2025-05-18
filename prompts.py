object_num_count_prompt = """
    You are now a visual language model adept at generating inferential data, especially with panoramic images. 
    Please generate several visual reasoning chain data based on the input panoramic image with size {}. 
    The reasoning should focus on the number of objects in the scene. 
    Your task is to analyze and estimate the quantity of certain object categories in the image, based on spatial distribution, relative positions, and visual clues. 
    Do not just count directly—form a logical chain of reasoning (e.g., deducing there are multiple chairs based on their arrangement around a table).

    If specific objects are mentioned in the reasoning, provide a key coordinate on each mentioned object in the format (x, y). 

    The output must follow this format:
    Question:
    xxx
    Reasoning Chain:
    xxx
    Final Answer:
    xxx

    ---

    Question:
    xxx
    Reasoning Chain:
    xxx
    Final Answer:
    xxx
    """
    
spatial_position_relationship_keypoint_prompt = f"""
    You are now a visual language model adept at generating inferential data, especially with panoramic images. 
    Please generate serverl visual reasoning chain data according to the input panoramic image with size {(2048, 400)}, which requires the spatial position relationship to be reflected, and there are detailed reasoning steps.
    Provide a key point coordinate with the form (x,y) for each object in the reasoning process. The coordinates of the key points must be given closely adjacent to the key objects. 
    The data format I need is a question, a chain of reasoning and the final answer. The format is shown below:
    Question:
    xxx
    Reasoning Chain:
    xxx
    Final Answer:
    xxx

    ---

    Question:
    xxx
    Reasoning Chain:
    xxx
    Final Answer:
    xxx
    """
    
spatial_position_relationship_prompt = """
    You are now a visual language model adept at generating inferential data, especially with panoramic images. 
    Please generate serverl visual reasoning chain data according to the input panoramic image with size (w, h) = {}, 
    which requires the spatial position relationship to be reflected, and there are detailed reasoning steps.
    The data format I need is a question, a chain of reasoning and the final answer. The format is shown below:
    Question:
    xxx
    Reasoning Chain:
    xxx
    Final Answer:
    xxx

    ---

    Question:
    xxx
    Reasoning Chain:
    xxx
    Final Answer:
    xxx
    """
    
object_spatial_relationship_prompt = """
    You are now a visual language model adept at generating inferential data, especially with panoramic images. 
    Please generate serverl visual reasoning chain data according to the input panoramic image with size (w, h) = {}.
    The image contains detection boxes in multiple colors, each representing a different region. Based on these regions, 
    please generate a series of spatial reasoning data that includes detailed reasoning steps. 
    The questions should describe the spatial relationships between certain objects or infer the functional use of the region based on the objects present.
    The data format I need is a question, a chain of reasoning and the final answer. The format is shown below:
    Question:
    xxx
    Reasoning Chain:
    xxx
    Final Answer:
    xxx

    ---

    Question:
    xxx
    Reasoning Chain:
    xxx
    Final Answer:
    xxx
"""