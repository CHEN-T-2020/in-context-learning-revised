import os

family = "gpt2"
n_embd = 256
n_layer = 8
n_dims = 5
n_positions = 2 * n_dims + 1

date = "4July"
name = f"{date}_{family}_{n_dims}dim_{n_layer}layer_{n_embd}_RL"

# 指定文件夹路径
folder_path = "/home/tianqi/in-context-learning-main/src/conf/cust_models"


def generate_config_file(folder_path, model_config, training_config):
    # 创建文件夹（如果不存在）
    os.makedirs(folder_path, exist_ok=True)

    # 生成model配置文档
    model_config_str = "model:\n"
    for key, value in model_config.items():
        model_config_str += f"    {key}: {value}\n"

    # 生成training配置文档
    training_config_str = "training:\n"
    for key, value in training_config.items():
        if isinstance(value, dict):
            training_config_str += f"    {key}:\n"
            for k, v in value.items():
                training_config_str += f"        {k}: {v}\n"
        else:
            training_config_str += f"    {key}: {value}\n"

    # 生成完整的配置文档内容
    config_content = model_config_str + "\n" + training_config_str

    # 生成文件路径
    file_path = os.path.join(
        folder_path,
        f"{name}.yaml",
    )

    # 将配置文档内容写入文件
    with open(file_path, "w") as file:
        file.write(config_content)


# 模型配置
model_config = {
    "family": family,
    "n_embd": n_embd,
    "n_layer": n_layer,
    "n_head": 8,
    "n_dims": n_dims,
    "n_positions": n_positions,
}

# 训练配置
training_config = {
    "resume_id": name,
    "task": "linear_regression",
    "data": "gaussian",
    "task_kwargs": "{" + "}",
    "batch_size": 64,
    "learning_rate": 0.0001,
    "save_every_steps": 1000,
    "keep_every_steps": 100000,
    "train_steps": 50001,
    "curriculum": {
        "dims": {"start": n_dims, "end": n_dims, "inc": 1, "interval": 2000},
        "points": {
            "start": n_positions,
            "end": n_positions,
            "inc": 2,
            "interval": 2000,
        },
    },
    "out_dir": "../models/linear_regression",
    "wandb": {
        "name": name,
        "project": "in-context-training",
        "entity": "tianqi_chen",
        "notes": "",
        "log_every_steps": 100,
    },
}

# 生成配置文件
generate_config_file(folder_path, model_config, training_config)
