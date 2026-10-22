import hashlib
import json
import sys
from pathlib import Path


def compact_bytes(value, *, sorted_keys=False):
    return json.dumps(value, sort_keys=sorted_keys, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def main(config_path, checkpoint_path):
    config = {
        "dt": 0.01,
        "steps": 100,
        "seed": 19027,
        "boundary": "box",
        "particles": [
            {"id": "p-a", "mass": 1.0, "radius": 0.1, "x": 0.25, "y": 0.5, "vx": 0.4, "vy": 0.0},
            {"id": "p-b", "mass": 2.0, "radius": 0.1, "x": 0.75, "y": 0.5, "vx": -0.2, "vy": 0.0},
        ],
        "mass_checks": [
            {"check_id": "C-A", "coefficients": {"p-a": 1, "p-b": 0}, "total": 1.0},
            {"check_id": "C-B", "coefficients": {"p-a": 0, "p-b": 1}, "total": 2.0},
        ],
    }
    config_bytes = compact_bytes(config) + b"\n"
    Path(config_path).write_bytes(config_bytes)

    particles = config["particles"]
    initial_energy = sum(0.5 * p["mass"] * (p["vx"] ** 2 + p["vy"] ** 2) for p in particles)
    initial_momentum = [
        sum(p["mass"] * p["vx"] for p in particles),
        sum(p["mass"] * p["vy"] for p in particles),
    ]
    checkpoint = {
        "version": 1,
        "step": 0,
        "seed": config["seed"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "initial_energy": initial_energy,
        "initial_momentum": initial_momentum,
        "max_relative_energy_error": 0.0,
        "collision_count": 0,
        "particles": particles,
        "state_sha256": "",
    }
    payload = {
        "particles": particles,
        "initial_energy": initial_energy,
        "initial_momentum": initial_momentum,
        "max_relative_energy_error": 0.0,
        "collision_count": 0,
    }
    checkpoint["state_sha256"] = hashlib.sha256(compact_bytes(payload, sorted_keys=True)).hexdigest()
    Path(checkpoint_path).write_bytes(compact_bytes(checkpoint) + b"\n")
    source = Path(config_path).parent / "DATA_SOURCE.txt"
    source.write_text(
        "这批参数来自实验室对封闭盒碰撞恢复流程的例行复核。"
        "生成脚本使用固定种子 19027，按同一份交接规则写入两颗粒子的质量、位置和速度，"
        "覆盖两种质量比例，目的是复现一次不等质量正碰及第 0 步检查点。"
        "文件只用于接口验收，不包含真实人员、设备或现场测量信息。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
