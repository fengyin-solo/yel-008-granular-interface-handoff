import pytest

import copy
import hashlib
import json
import math
import os
import resource
import subprocess
import tempfile
from pathlib import Path

SCRIPT = Path(os.environ.get("SIM_SCRIPT", "/app/simulate.py"))
CONFIG = Path(os.environ.get("SIM_CONFIG", "/data/simulation.json"))
GENERATED_CHECKPOINT = Path(os.environ.get("SIM_GENERATED_CHECKPOINT", "/output/state.chk"))
PARTICLE_ORDER = ["id", "mass", "radius", "x", "y", "vx", "vy"]
PARTICLE_KEYS = set(PARTICLE_ORDER)
CHECKPOINT_ORDER = [
    "version", "step", "seed", "config_sha256", "initial_energy",
    "initial_momentum", "max_relative_energy_error", "collision_count",
    "particles", "state_sha256",
]
PROCESS_TIMEOUT = 30
MAX_JSON_BYTES = 65536
MAX_STEPS = 20000
MAX_SEED = 9223372036854775807
MAX_ID_BYTES = 128

DIAGNOSTIC_ORDER = [
    "initial_energy", "final_energy", "max_relative_energy_error",
    "initial_momentum", "final_momentum", "collision_count",
]


def compact(value, *, sort_keys=False):
    return json.dumps(value, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def pass_case(case_id):
    print("CASE PASS " + case_id, flush=True)


def base_config(**changes):
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value.update(changes)
    return value


def run(config, checkpoint=None, output=None):
    directory = Path(tempfile.mkdtemp(prefix="sim-case-"))
    config_path = directory / "config.json"
    output_path = output or directory / "run.json"
    checkpoint_path = checkpoint or directory / "state.chk"
    config_path.write_bytes(compact(config) + b"\n")
    if checkpoint is None:
        checkpoint_path.write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        ["python3", str(SCRIPT), "--config", str(config_path), "--output", str(output_path), "--checkpoint", str(checkpoint_path)],
        capture_output=True, text=True, timeout=PROCESS_TIMEOUT,
    )
    return result, output_path, checkpoint_path


def energy(particles):
    return sum(0.5 * p["mass"] * (p["vx"] ** 2 + p["vy"] ** 2) for p in particles)


def momentum(particles):
    return [sum(p["mass"] * p["vx"] for p in particles), sum(p["mass"] * p["vy"] for p in particles)]


def checkpoint_payload(checkpoint):
    return {
        "particles": checkpoint["particles"],
        "initial_energy": checkpoint["initial_energy"],
        "initial_momentum": checkpoint["initial_momentum"],
        "max_relative_energy_error": checkpoint["max_relative_energy_error"],
        "collision_count": checkpoint["collision_count"],
    }


def seal_checkpoint(config_path, config, particles, *, step, initial_energy=None, initial_momentum=None, max_error=0.0, collision_count=0):
    checkpoint = {
        "version": 1,
        "step": step,
        "seed": config["seed"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "initial_energy": energy(config["particles"]) if initial_energy is None else initial_energy,
        "initial_momentum": momentum(config["particles"]) if initial_momentum is None else initial_momentum,
        "max_relative_energy_error": max_error,
        "collision_count": collision_count,
        "particles": copy.deepcopy(particles),
        "state_sha256": "",
    }
    checkpoint["state_sha256"] = hashlib.sha256(compact(checkpoint_payload(checkpoint), sort_keys=True)).hexdigest()
    return checkpoint


def run_with_raw_config(raw, *, output_bytes=b"old output\n", checkpoint_bytes=b"old checkpoint\n"):
    directory = Path(tempfile.mkdtemp(prefix="sim-raw-"))
    config = directory / "config.json"
    output = directory / "run.json"
    checkpoint = directory / "state.chk"
    config.write_bytes(raw)
    output.write_bytes(output_bytes)
    checkpoint.write_bytes(checkpoint_bytes)
    result = subprocess.run(
        ["python3", str(SCRIPT), "--config", str(config), "--output", str(output), "--checkpoint", str(checkpoint)],
        capture_output=True, text=True, timeout=PROCESS_TIMEOUT,
    )
    return result, output, checkpoint


# Entrypoint and complete input contract.
def test_entrypoint_exists_and_runs():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    # CASE LEG-001-A LEG-001-B LEG-009-A
    assert SCRIPT.is_file(), "入口脚本缺失"
    result, output, checkpoint = run(base_config(steps=2))
    assert result.returncode == 0, result.stderr
    assert output.is_file() and checkpoint.is_file()


def test_config_rejects_non_object_invalid_utf8_and_duplicate_keys_without_publish():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    # CASE LEG-009-B
    valid = compact(base_config(steps=0)) + b"\n"
    raws = [
        b"[]\n", b"null\n", b'"config"\n', b"{\n", valid[:-1] + b"\xff\n",
        b'{"dt":0.01,"dt":0.02,"steps":0,"seed":1,"boundary":"box","particles":[]}\n',
        b'{"dt":0.01,"steps":0,"seed":1,"boundary":"box","particles":[{"id":"p","mass":1,"radius":0.1,"x":0.2,"y":0.2,"vx":0,"vx":1,"vy":0}]}\n',
        b'{"dt":1,"steps":0,"seed":0,"boundary":"box","particles":[{"id":"\\ud800","mass":1,"radius":0.01,"x":0,"y":0,"vx":0,"vy":0}]}\n',
        b" " * (MAX_JSON_BYTES + 1),
    ]
    for index, raw in enumerate(raws):
        result, output, checkpoint = run_with_raw_config(raw)
        assert result.returncode != 0
        assert output.read_bytes() == b"old output\n" and checkpoint.read_bytes() == b"old checkpoint\n"
        pass_case(f"config-raw-{index}")
    exact = compact(base_config(steps=0))
    exact += b" " * (MAX_JSON_BYTES - len(exact))
    assert len(exact) == MAX_JSON_BYTES
    result, output, checkpoint = run_with_raw_config(exact)
    assert result.returncode == 0, result.stderr
    assert output.is_file() and checkpoint.is_file()
    pass_case("config-raw-valid-exact-byte-limit")


def test_config_read_failure_exits_before_publish():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    directory = Path(tempfile.mkdtemp(prefix="sim-missing-config-"))
    output = directory / "run.json"; checkpoint = directory / "state.chk"
    output.write_bytes(b"old output\n"); checkpoint.write_bytes(b"old checkpoint\n")
    result = subprocess.run(
        ["python3", str(SCRIPT), "--config", str(directory / "missing.json"), "--output", str(output), "--checkpoint", str(checkpoint)],
        capture_output=True, text=True, timeout=PROCESS_TIMEOUT,
    )
    assert result.returncode != 0
    assert output.read_bytes() == b"old output\n" and checkpoint.read_bytes() == b"old checkpoint\n"


def test_config_requires_exact_top_level_and_particle_fields():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    config = base_config(steps=0)
    untouched_targets = []
    for field in ("dt", "steps", "seed", "boundary", "particles"):
        candidate = copy.deepcopy(config); candidate.pop(field)
        result, output, checkpoint = run(candidate)
        assert result.returncode != 0, field
        untouched_targets.append((output, checkpoint))
        pass_case("config-missing-" + field)
    result, output, checkpoint = run(dict(config, extra=1))
    assert result.returncode != 0
    untouched_targets.append((output, checkpoint))
    pass_case("config-extra-field")
    for field in PARTICLE_ORDER:
        particle = copy.deepcopy(config["particles"][0]); particle.pop(field)
        result, output, checkpoint = run(dict(config, particles=[particle]))
        assert result.returncode != 0, field
        untouched_targets.append((output, checkpoint))
        pass_case("config-particle-missing-" + field)
    particle = copy.deepcopy(config["particles"][0]); particle["extra"] = 1
    result, output, checkpoint = run(dict(config, particles=[particle]))
    assert result.returncode != 0
    untouched_targets.append((output, checkpoint))
    pass_case("config-particle-extra-field")
    assert all(not output.exists() for output, _ in untouched_targets)
    assert all(checkpoint.read_text() == "{}\n" for _, checkpoint in untouched_targets)


def test_config_types_ranges_boundary_and_unique_ids():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    # CASE LEG-004-A
    config = base_config(steps=0)
    invalid = [
        ("dt-zero", dict(config, dt=0)),
        ("dt-below", dict(config, dt=9.99e-10)),
        ("dt-above", dict(config, dt=100.000001)),
        ("dt-nan", dict(config, dt=float("nan"))),
        ("dt-bool", dict(config, dt=True)),
        ("dt-string", dict(config, dt="0.1")),
        ("steps-below", dict(config, steps=-1)),
        ("steps-above", dict(config, steps=MAX_STEPS + 1)),
        ("steps-bool", dict(config, steps=True)),
        ("steps-float", dict(config, steps=1.0)),
        ("seed-negative", dict(config, seed=-1)),
        ("seed-above", dict(config, seed=MAX_SEED + 1)),
        ("seed-bool", dict(config, seed=False)),
        ("seed-float", dict(config, seed=1.0)),
        ("boundary-enum", dict(config, boundary="reflective")),
        ("boundary-type", dict(config, boundary=1)),
        ("particles-type", dict(config, particles={})),
    ]
    invalid_targets = []
    for case_id, candidate in invalid:
        result, output, checkpoint = run(candidate)
        assert result.returncode != 0, case_id
        invalid_targets.append((output, checkpoint))
        pass_case("config-" + case_id)
    invalid_particle_values = (
        ("id-empty", "id", ""), ("id-type", "id", 7),
        ("mass-below", "mass", 9.99e-5), ("mass-above", "mass", 10000.0001),
        ("radius-below", "radius", 9.99e-7), ("radius-above", "radius", 1.000001),
        ("x-below", "x", -1e-9), ("x-above", "x", 1.000000001),
        ("y-below", "y", -1e-9), ("y-above", "y", 1.000000001),
        ("vx-positive-above", "vx", 10000.0001), ("vx-negative-above", "vx", -10000.0001),
        ("vy-positive-above", "vy", 10000.0001), ("vy-negative-above", "vy", -10000.0001),
    )
    for case_id, key, value in invalid_particle_values:
        particle = copy.deepcopy(config["particles"][0]); particle[key] = value
        result, output, checkpoint = run(dict(config, particles=[particle]))
        assert result.returncode != 0, case_id
        invalid_targets.append((output, checkpoint))
        pass_case("config-particle-" + case_id)
    assert all(not output.exists() for output, _ in invalid_targets)
    assert all(checkpoint.read_text() == "{}\n" for _, checkpoint in invalid_targets)
    malformed_particles = []
    for key in PARTICLE_ORDER[1:]:
        for type_name, value in (("bool", True), ("string", "1"), ("infinite", float("inf"))):
            particle = copy.deepcopy(config["particles"][0]); particle[key] = value
            malformed_particles.append((key, type_name, run(dict(config, particles=[particle]))[0].returncode))
    assert all(returncode != 0 for _, _, returncode in malformed_particles)
    for key, type_name, _ in malformed_particles:
        pass_case(f"config-particle-{key}-{type_name}")
    duplicate = [copy.deepcopy(config["particles"][0]), copy.deepcopy(config["particles"][0])]
    assert run(dict(config, particles=duplicate))[0].returncode != 0
    pass_case("config-particle-duplicate-id")
    id_boundary = copy.deepcopy(config["particles"][0]); id_boundary["id"] = "a" * MAX_ID_BYTES
    assert run(dict(config, particles=[id_boundary]))[0].returncode == 0
    pass_case("config-particle-id-valid-128-bytes")
    id_too_long = copy.deepcopy(id_boundary); id_too_long["id"] += "a"
    assert run(dict(config, particles=[id_too_long]))[0].returncode != 0
    pass_case("config-particle-id-invalid-129-bytes")
    for steps in (0, MAX_STEPS):
        assert run(dict(config, steps=steps))[0].returncode == 0
        pass_case(f"config-steps-valid-{steps}")
    for seed in (0, MAX_SEED):
        assert run(dict(config, seed=seed))[0].returncode == 0
        pass_case(f"config-seed-valid-{seed}")
    assert run(dict(config, particles=[]))[0].returncode == 0
    pass_case("config-particles-valid-empty")
    for dt in (1e-9, 100):
        assert run(dict(config, dt=dt))[0].returncode == 0
        pass_case(f"config-dt-valid-{dt}")
    lower_particle = copy.deepcopy(config["particles"][0])
    lower_particle.update(id="lower", mass=1e-4, radius=1e-6, x=0, y=0, vx=-1e4, vy=-1e4)
    upper_particle = copy.deepcopy(config["particles"][0])
    upper_particle.update(id="upper", mass=1e4, radius=1, x=1, y=1, vx=1e4, vy=1e4)
    assert run(dict(config, particles=[lower_particle, upper_particle]))[0].returncode == 0
    pass_case("config-particle-all-valid-boundaries")
    maximum_particles = [dict(lower_particle, id=f"p-{index}", x=(index % 8) / 8 + 0.01, y=(index // 8) / 8 + 0.01, vx=0, vy=0) for index in range(64)]
    assert run(dict(config, particles=maximum_particles, steps=MAX_STEPS))[0].returncode == 0
    pass_case("config-particles-valid-64-at-max-steps")
    assert run(dict(config, particles=maximum_particles + [dict(lower_particle, id="p-64")]))[0].returncode != 0
    pass_case("config-particles-invalid-65")


def test_generated_data_is_deterministic_and_checkpoint_zero_state_is_valid():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    # CASE LEG-003-A LEG-004-B LEG-005-A LEG-006-A
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["seed"] == 19027 and len(config["particles"]) == 2
    assert config["particles"][0]["mass"] != config["particles"][1]["mass"]
    generated = json.loads(GENERATED_CHECKPOINT.read_text(encoding="utf-8"))
    assert list(generated) == CHECKPOINT_ORDER and generated["step"] == 0 and generated["seed"] == 19027
    assert generated["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert generated["state_sha256"] == hashlib.sha256(compact(checkpoint_payload(generated), sort_keys=True)).hexdigest()
    assert generated["initial_energy"] == energy(config["particles"])
    assert generated["initial_momentum"] == momentum(config["particles"])
    assert generated["max_relative_energy_error"] == 0 and generated["collision_count"] == 0
    assert [particle["id"] for particle in generated["particles"]] == [particle["id"] for particle in config["particles"]]
    result, output, checkpoint = run(config)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["steps"] == config["steps"]
    assert json.loads(checkpoint.read_text())["step"] == config["steps"]


# Output and serialization contract.
def test_output_checkpoint_schema_order_types_and_hashes():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    # CASE LEG-003-B LEG-005-B LEG-006-B
    result, output, checkpoint = run(base_config(steps=0))
    assert result.returncode == 0
    raw_output = output.read_bytes(); raw_checkpoint = checkpoint.read_bytes()
    assert raw_output.endswith(b"\n") and b"\n" not in raw_output[:-1]
    assert raw_checkpoint.endswith(b"\n") and b"\n" not in raw_checkpoint[:-1]
    data = json.loads(raw_output); saved = json.loads(raw_checkpoint)
    assert list(data) == ["steps", "final_state", "diagnostics", "checkpoint_sha256"]
    assert list(data["diagnostics"]) == DIAGNOSTIC_ORDER
    assert list(saved) == CHECKPOINT_ORDER
    assert all(list(particle) == PARTICLE_ORDER for particle in data["final_state"] + saved["particles"])
    assert raw_output == compact(data) + b"\n" and raw_checkpoint == compact(saved) + b"\n"
    assert data["checkpoint_sha256"] == hashlib.sha256(raw_checkpoint).hexdigest()
    assert saved["state_sha256"] == hashlib.sha256(compact(checkpoint_payload(saved), sort_keys=True)).hexdigest()
    for digest in (data["checkpoint_sha256"], saved["config_sha256"], saved["state_sha256"]):
        assert len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
    assert data["diagnostics"]["max_relative_energy_error"] == 0
    assert data["diagnostics"]["collision_count"] == 0


def test_output_preserves_input_order_and_all_diagnostics_are_finite():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    particles = [
        {"id": "z", "mass": 1.0, "radius": 0.02, "x": 0.2, "y": 0.2, "vx": 0.0, "vy": 0.0},
        {"id": "a", "mass": 2.0, "radius": 0.02, "x": 0.8, "y": 0.8, "vx": 0.0, "vy": 0.0},
    ]
    result, output, _ = run(base_config(particles=particles, steps=0)); assert result.returncode == 0
    data = json.loads(output.read_text()); diagnostics = data["diagnostics"]
    assert [particle["id"] for particle in data["final_state"]] == ["z", "a"]
    for key in ("initial_energy", "final_energy", "max_relative_energy_error"):
        assert type(diagnostics[key]) in (int, float) and math.isfinite(diagnostics[key])
    for key in ("initial_momentum", "final_momentum"):
        assert len(diagnostics[key]) == 2 and all(type(value) in (int, float) and math.isfinite(value) for value in diagnostics[key])
    assert type(diagnostics["collision_count"]) is int and diagnostics["collision_count"] >= 0


def test_extreme_finite_input_still_produces_finite_json_numbers():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    particles = [{"id": "large", "mass": 1e4, "radius": 1.0, "x": 0.1, "y": 0.2, "vx": 1e4, "vy": -1e4}]
    result, output, _ = run(base_config(particles=particles, boundary="periodic", dt=1e-9, steps=2))
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text()); diagnostics = data["diagnostics"]
    assert diagnostics["initial_energy"] == energy(particles)
    assert diagnostics["initial_momentum"] == momentum(particles)
    for key in ("initial_energy", "final_energy", "max_relative_energy_error"):
        assert math.isfinite(diagnostics[key])
    for key in ("initial_momentum", "final_momentum"):
        assert all(math.isfinite(value) for value in diagnostics[key])
    assert all(math.isfinite(particle[key]) for particle in data["final_state"] for key in PARTICLE_ORDER[1:])


def test_repeat_from_final_checkpoint_is_byte_stable():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    directory = Path(tempfile.mkdtemp(prefix="sim-repeat-"))
    config = directory / "config.json"; output = directory / "run.json"; checkpoint = directory / "state.chk"
    config.write_bytes(compact(base_config(steps=6)) + b"\n"); checkpoint.write_text("{}\n")
    args = ["python3", str(SCRIPT), "--config", str(config), "--output", str(output), "--checkpoint", str(checkpoint)]
    assert subprocess.run(args, capture_output=True, text=True, timeout=PROCESS_TIMEOUT).returncode == 0
    before = (output.read_bytes(), checkpoint.read_bytes())
    assert subprocess.run(args, capture_output=True, text=True, timeout=PROCESS_TIMEOUT).returncode == 0
    assert (output.read_bytes(), checkpoint.read_bytes()) == before


# Core challenge 1: coupled physics and geometry.
def test_box_boundary_reflects_once_and_across_multiple_walls_on_both_axes():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    simple = {"id": "edge", "mass": 1.0, "radius": 0.01, "x": 0.99, "y": 0.5, "vx": 1.0, "vy": 0.0}
    result, output, _ = run(base_config(particles=[simple], boundary="box", dt=0.2, steps=1)); assert result.returncode == 0
    particle = json.loads(output.read_text())["final_state"][0]
    assert math.isclose(particle["x"], 0.81, abs_tol=1e-12) and particle["vx"] < 0
    multiple = {"id": "multi", "mass": 1.0, "radius": 0.01, "x": 0.25, "y": 0.25, "vx": 10.0, "vy": -10.0}
    result, output, _ = run(base_config(particles=[multiple], boundary="box", dt=0.35, steps=1)); assert result.returncode == 0
    particle = json.loads(output.read_text())["final_state"][0]
    assert math.isclose(particle["x"], 0.25, abs_tol=1e-12) and math.isclose(particle["y"], 0.75, abs_tol=1e-12)
    assert math.isclose(particle["vx"], -10.0, abs_tol=1e-12) and math.isclose(particle["vy"], -10.0, abs_tol=1e-12)
    exact = {"id": "exact", "mass": 1.0, "radius": 0.01, "x": 0.25, "y": 0.75, "vx": 1.75, "vy": -1.75}
    result, output, _ = run(base_config(particles=[exact], boundary="box", dt=1.0, steps=1)); assert result.returncode == 0
    particle = json.loads(output.read_text())["final_state"][0]
    assert math.isclose(particle["x"], 0.0, abs_tol=1e-12) and math.isclose(particle["vx"], -1.75, abs_tol=1e-12)
    assert math.isclose(particle["y"], 1.0, abs_tol=1e-12) and math.isclose(particle["vy"], 1.75, abs_tol=1e-12)

    # JSON integers are part of the declared finite-number domain. Exercise an
    # integer-only exact-wall endpoint that previously reached int.is_integer().
    integer_exact = {"id": "integer-exact", "mass": 1, "radius": 0.01, "x": 0, "y": 0, "vx": 2, "vy": 0}
    result, output, _ = run(base_config(particles=[integer_exact], boundary="box", dt=1, steps=1)); assert result.returncode == 0, result.stderr
    particle = json.loads(output.read_text())["final_state"][0]
    assert math.isclose(particle["x"], 0.0, abs_tol=1e-12) and particle["vx"] == -2

    # Also cover an integer velocity crossing several walls and ending exactly
    # on a wall, so parity cannot be approximated by a single reflection.
    integer_multiwall = {"id": "integer-multiwall", "mass": 1, "radius": 0.01, "x": 0, "y": 0, "vx": 5, "vy": 0}
    result, output, _ = run(base_config(particles=[integer_multiwall], boundary="box", dt=1, steps=1)); assert result.returncode == 0, result.stderr
    particle = json.loads(output.read_text())["final_state"][0]
    assert math.isclose(particle["x"], 1.0, abs_tol=1e-12) and particle["vx"] == 5


def test_periodic_wrap_and_minimum_image_collisions_across_x_and_y_seams():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    # CASE LEG-002-A LEG-007-A
    mover = {"id": "edge", "mass": 1.0, "radius": 0.01, "x": 0.99, "y": 0.99, "vx": 1.0, "vy": 1.0}
    result, output, _ = run(base_config(particles=[mover], boundary="periodic", dt=0.2, steps=1)); assert result.returncode == 0
    particle = json.loads(output.read_text())["final_state"][0]
    assert math.isclose(particle["x"], 0.19, abs_tol=1e-12) and math.isclose(particle["y"], 0.19, abs_tol=1e-12)
    for axis in ("x", "y"):
        first = {"id": "a", "mass": 1.0, "radius": 0.03, "x": 0.5, "y": 0.5, "vx": 0.0, "vy": 0.0}
        second = {"id": "b", "mass": 1.0, "radius": 0.03, "x": 0.5, "y": 0.5, "vx": 0.0, "vy": 0.0}
        first[axis] = 0.98; second[axis] = 0.02
        velocity = "vx" if axis == "x" else "vy"
        first[velocity] = 1.0; second[velocity] = -1.0
        result, output, _ = run(base_config(particles=[first, second], boundary="periodic", dt=1e-6, steps=1)); assert result.returncode == 0
        data = json.loads(output.read_text()); state = {p["id"]: p for p in data["final_state"]}
        assert data["diagnostics"]["collision_count"] == 1
        assert math.isclose(state["a"][velocity], -1.0, abs_tol=1e-12)
        assert math.isclose(state["b"][velocity], 1.0, abs_tol=1e-12)
        pass_case("periodic-seam-" + axis)
    tie_dt = 2 ** -29
    tie = [
        {"id": "a", "mass": 1.0, "radius": 0.25, "x": 0.25 - tie_dt, "y": 0.5, "vx": 1.0, "vy": 0.0},
        {"id": "b", "mass": 1.0, "radius": 0.25, "x": 0.75 + tie_dt, "y": 0.5, "vx": -1.0, "vy": 0.0},
    ]
    result, output, _ = run(base_config(particles=tie, boundary="periodic", dt=tie_dt, steps=1)); assert result.returncode == 0
    data = json.loads(output.read_text()); state = {particle["id"]: particle for particle in data["final_state"]}
    assert data["diagnostics"]["collision_count"] == 1
    assert math.isclose(state["a"]["vx"], -1.0, abs_tol=1e-12) and math.isclose(state["b"]["vx"], 1.0, abs_tol=1e-12)
    pass_case("periodic-half-box-tie-keeps-raw-sign")

    negative = {"id": "negative", "mass": 1.0, "radius": 0.01, "x": 0.01, "y": 0.01, "vx": -1.0, "vy": -1.0}
    result, output, _ = run(base_config(particles=[negative], boundary="periodic", dt=0.2, steps=1)); assert result.returncode == 0
    particle = json.loads(output.read_text())["final_state"][0]
    assert math.isclose(particle["x"], 0.81, abs_tol=1e-12) and math.isclose(particle["y"], 0.81, abs_tol=1e-12)
    pass_case("periodic-negative-wrap-both-axes")
    diagonal = [
        {"id": "a", "mass": 1.0, "radius": 0.04, "x": 0.98, "y": 0.98, "vx": 1.0, "vy": 1.0},
        {"id": "b", "mass": 1.0, "radius": 0.04, "x": 0.02, "y": 0.02, "vx": -1.0, "vy": -1.0},
    ]
    result, output, _ = run(base_config(particles=diagonal, boundary="periodic", dt=1e-6, steps=1)); assert result.returncode == 0
    data = json.loads(output.read_text())
    assert data["diagnostics"]["collision_count"] == 1
    pass_case("periodic-diagonal-minimum-image")


def test_each_step_advances_then_applies_boundary_then_resolves_collisions():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    # CASE LEG-002-B LEG-007-B
    particles = [
        {"id": "a", "mass": 1.0, "radius": 0.06, "x": 0.75, "y": 0.5, "vx": 1.0, "vy": 0.0},
        {"id": "b", "mass": 1.0, "radius": 0.06, "x": 0.05, "y": 0.5, "vx": -1.0, "vy": 0.0},
    ]
    result, output, _ = run(base_config(particles=particles, boundary="periodic", dt=0.1, steps=1)); assert result.returncode == 0
    data = json.loads(output.read_text()); state = {particle["id"]: particle for particle in data["final_state"]}
    assert data["diagnostics"]["collision_count"] == 1
    assert math.isclose(state["a"]["x"], 0.85, abs_tol=1e-12) and math.isclose(state["b"]["x"], 0.95, abs_tol=1e-12)
    assert math.isclose(state["a"]["vx"], -1.0, abs_tol=1e-12) and math.isclose(state["b"]["vx"], 1.0, abs_tol=1e-12)


def test_unequal_mass_oblique_collision_conserves_energy_and_momentum():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    particles = [
        {"id": "heavy", "mass": 3.0, "radius": 0.16, "x": 0.42, "y": 0.42, "vx": 0.2, "vy": 0.1},
        {"id": "light", "mass": 1.0, "radius": 0.16, "x": 0.58, "y": 0.58, "vx": -0.4, "vy": -0.3},
    ]
    result, output, _ = run(base_config(particles=particles, dt=1e-6, steps=1)); assert result.returncode == 0
    data = json.loads(output.read_text()); diagnostics = data["diagnostics"]
    assert diagnostics["collision_count"] == 1
    assert math.isclose(diagnostics["initial_energy"], diagnostics["final_energy"], rel_tol=1e-12, abs_tol=1e-12)
    assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(diagnostics["initial_momentum"], diagnostics["final_momentum"]))
    state = {particle["id"]: particle for particle in data["final_state"]}
    advanced = [dict(particle, x=particle["x"] + particle["vx"] * 1e-6, y=particle["y"] + particle["vy"] * 1e-6) for particle in particles]
    dx = advanced[1]["x"] - advanced[0]["x"]; dy = advanced[1]["y"] - advanced[0]["y"]
    distance = math.hypot(dx, dy); nx = dx / distance; ny = dy / distance
    relative_normal = (advanced[1]["vx"] - advanced[0]["vx"]) * nx + (advanced[1]["vy"] - advanced[0]["vy"]) * ny
    impulse = -(1.0 + 1.0) * relative_normal / (1.0 / advanced[0]["mass"] + 1.0 / advanced[1]["mass"])
    expected = copy.deepcopy(advanced)
    expected[0]["vx"] -= impulse * nx / expected[0]["mass"]; expected[0]["vy"] -= impulse * ny / expected[0]["mass"]
    expected[1]["vx"] += impulse * nx / expected[1]["mass"]; expected[1]["vy"] += impulse * ny / expected[1]["mass"]
    comparisons = [
        math.isclose(state[expected_particle["id"]][key], expected_particle[key], rel_tol=1e-12, abs_tol=1e-12)
        for expected_particle in expected
        for key in ("vx", "vy")
    ]
    assert all(comparisons)


def test_utf8_pair_order_exact_overlap_and_once_per_step_are_deterministic():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    delta = 2 ** -29
    particles = [
        {"id": "Ω", "mass": 1.0, "radius": 0.4, "x": 0.5 + delta, "y": 0.5, "vx": -1.0, "vy": 0.0},
        {"id": "é", "mass": 1.0, "radius": 0.4, "x": 0.5, "y": 0.5, "vx": 0.0, "vy": 0.0},
        {"id": "a", "mass": 1.0, "radius": 0.4, "x": 0.5 - delta, "y": 0.5, "vx": 1.0, "vy": 0.0},
    ]
    result, output, _ = run(base_config(particles=particles, dt=delta, steps=1)); assert result.returncode == 0
    data = json.loads(output.read_text()); state = {particle["id"]: particle for particle in data["final_state"]}
    assert data["diagnostics"]["collision_count"] == 3
    assert math.isclose(state["a"]["vx"], -1.0, abs_tol=1e-12)
    assert math.isclose(state["é"]["vx"], 0.0, abs_tol=1e-12)
    assert math.isclose(state["Ω"]["vx"], 1.0, abs_tol=1e-12)
    pair = [
        {"id": "left", "mass": 1.0, "radius": 0.2, "x": 0.4, "y": 0.5, "vx": 1.0, "vy": 0.0},
        {"id": "right", "mass": 1.0, "radius": 0.2, "x": 0.6, "y": 0.5, "vx": -1.0, "vy": 0.0},
    ]
    result, output, _ = run(base_config(particles=pair, dt=0.1, steps=1)); assert result.returncode == 0
    data = json.loads(output.read_text()); state = {particle["id"]: particle for particle in data["final_state"]}
    assert data["diagnostics"]["collision_count"] == 1
    assert math.isclose(state["left"]["vx"], -1.0, abs_tol=1e-12) and math.isclose(state["right"]["vx"], 1.0, abs_tol=1e-12)


def test_overlapping_but_separating_particles_do_not_collide():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    particles = [
        {"id": "left", "mass": 1.0, "radius": 0.2, "x": 0.45, "y": 0.5, "vx": -1.0, "vy": 0.0},
        {"id": "right", "mass": 1.0, "radius": 0.2, "x": 0.55, "y": 0.5, "vx": 1.0, "vy": 0.0},
    ]
    result, output, _ = run(base_config(particles=particles, dt=1e-6, steps=1)); assert result.returncode == 0
    data = json.loads(output.read_text()); state = {particle["id"]: particle for particle in data["final_state"]}
    assert data["diagnostics"]["collision_count"] == 0
    assert math.isclose(state["left"]["vx"], -1.0, abs_tol=1e-12) and math.isclose(state["right"]["vx"], 1.0, abs_tol=1e-12)
    pass_case("collision-overlap-but-separating")
    apart = [
        {"id": "left", "mass": 1.0, "radius": 0.05, "x": 0.2, "y": 0.5, "vx": 1.0, "vy": 0.0},
        {"id": "right", "mass": 1.0, "radius": 0.05, "x": 0.8, "y": 0.5, "vx": -1.0, "vy": 0.0},
    ]
    result, output, _ = run(base_config(particles=apart, dt=1e-6, steps=1)); assert result.returncode == 0
    data = json.loads(output.read_text())
    assert data["diagnostics"]["collision_count"] == 0
    pass_case("collision-approaching-but-not-overlapping")


def test_long_periodic_run_bounds_energy_drift_and_tracks_intermediate_peak():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    particles = [
        {"id": "a", "mass": 3.0, "radius": 0.12, "x": 0.6723249016155505, "y": 0.7927944058937093, "vx": 0.5470759595557046, "vy": 0.009404437293050016},
        {"id": "b", "mass": 1.0, "radius": 0.12, "x": 0.12016192928484509, "y": 0.13292681904665996, "vx": 0.38577294638014736, "vy": 0.641816404412674},
        {"id": "c", "mass": 3.0, "radius": 0.12, "x": 0.23995033909991, "y": 0.7779881561332634, "vx": 0.016901105529250504, "vy": 0.058425322912603384},
        {"id": "d", "mass": 11.0, "radius": 0.12, "x": 0.7044056390762873, "y": 0.48171191818631354, "vx": 0.20852183884889397, "vy": -0.17035978510199934},
        {"id": "e", "mass": 11.0, "radius": 0.12, "x": 0.9970231952259788, "y": 0.3852014552588005, "vx": -0.8090461313769751, "vy": -0.09636396401314706},
    ]
    result, output, _ = run(base_config(particles=particles, boundary="periodic", dt=0.003, steps=3000)); assert result.returncode == 0
    diagnostics = json.loads(output.read_text())["diagnostics"]
    final_error = abs(diagnostics["final_energy"] - diagnostics["initial_energy"]) / max(abs(diagnostics["initial_energy"]), 1e-30)
    assert diagnostics["collision_count"] >= 50
    assert diagnostics["max_relative_energy_error"] <= 1e-10
    assert diagnostics["max_relative_energy_error"] > final_error + 1e-16


# Core challenge 2: complete checkpoint recovery.
def test_valid_checkpoint_reorders_particles_and_preserves_historical_diagnostics():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    config = base_config(steps=2, particles=[
        {"id": "first", "mass": 1.0, "radius": 0.02, "x": 0.2, "y": 0.5, "vx": 0.0, "vy": 0.0},
        {"id": "second", "mass": 2.0, "radius": 0.02, "x": 0.8, "y": 0.5, "vx": 0.0, "vy": 0.0},
    ])
    directory = Path(tempfile.mkdtemp(prefix="sim-cp-history-")); config_path = directory / "config.json"
    config_path.write_bytes(compact(config) + b"\n")
    saved = [copy.deepcopy(config["particles"][1]), copy.deepcopy(config["particles"][0])]
    saved[0]["x"] = 0.7; saved[1]["x"] = 0.3
    saved[0]["vx"] = 1e10; saved[0]["vy"] = 1e10
    saved[1]["vx"] = -1e10; saved[1]["vy"] = -1e10
    checkpoint_data = seal_checkpoint(config_path, config, saved, step=2, max_error=7e-11, collision_count=2)
    checkpoint_raw = compact(checkpoint_data)
    checkpoint_raw += b" " * (MAX_JSON_BYTES - len(checkpoint_raw) - 1) + b"\n"
    assert len(checkpoint_raw) == MAX_JSON_BYTES
    checkpoint = directory / "state.chk"; checkpoint.write_bytes(checkpoint_raw)
    output = directory / "run.json"
    result = subprocess.run(["python3", str(SCRIPT), "--config", str(config_path), "--output", str(output), "--checkpoint", str(checkpoint)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert [particle["id"] for particle in data["final_state"]] == ["first", "second"]
    assert [particle["x"] for particle in data["final_state"]] == [0.3, 0.7]
    assert math.isclose(data["final_state"][0]["vx"], -1e10, abs_tol=1e-12)
    assert math.isclose(data["final_state"][0]["vy"], -1e10, abs_tol=1e-12)
    assert math.isclose(data["final_state"][1]["vx"], 1e10, abs_tol=1e-12)
    assert math.isclose(data["final_state"][1]["vy"], 1e10, abs_tol=1e-12)
    assert data["diagnostics"]["initial_energy"] == energy(config["particles"])
    assert data["diagnostics"]["initial_momentum"] == momentum(config["particles"])
    assert math.isclose(data["diagnostics"]["max_relative_energy_error"], 7e-11, abs_tol=1e-15)
    assert data["diagnostics"]["collision_count"] == 2


def test_resume_after_unequal_mass_collision_matches_continuous_files_byte_for_byte():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    particles = [
        {"id": "a", "mass": 1.0, "radius": 0.11, "x": 0.4, "y": 0.5, "vx": 1.0, "vy": 0.0},
        {"id": "b", "mass": 2.0, "radius": 0.11, "x": 0.6, "y": 0.5, "vx": -0.5, "vy": 0.0},
    ]
    config = base_config(particles=particles, boundary="periodic", dt=1e-6, steps=6)
    directory = Path(tempfile.mkdtemp(prefix="sim-cp-combined-")); config_path = directory / "config.json"
    config_path.write_bytes(compact(config) + b"\n")
    full_output = directory / "full.json"; full_checkpoint = directory / "full.chk"; full_checkpoint.write_text("{}\n")
    full = subprocess.run(["python3", str(SCRIPT), "--config", str(config_path), "--output", str(full_output), "--checkpoint", str(full_checkpoint)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT)
    assert full.returncode == 0, full.stderr
    after_one = copy.deepcopy(particles)
    after_one[0]["x"] += after_one[0]["vx"] * config["dt"]
    after_one[1]["x"] += after_one[1]["vx"] * config["dt"]
    after_one[0]["vx"] = -1.0
    after_one[1]["vx"] = 0.5
    intermediate = seal_checkpoint(config_path, config, after_one, step=1, max_error=0.0, collision_count=1)
    resumed_checkpoint = directory / "resumed.chk"; resumed_checkpoint.write_bytes(compact(intermediate) + b"\n")
    resumed_output = directory / "resumed.json"
    resumed = subprocess.run(["python3", str(SCRIPT), "--config", str(config_path), "--output", str(resumed_output), "--checkpoint", str(resumed_checkpoint)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed_output.read_bytes() == full_output.read_bytes()
    assert resumed_checkpoint.read_bytes() == full_checkpoint.read_bytes()
    assert json.loads(resumed_output.read_text())["diagnostics"]["collision_count"] == 1


def test_resumed_collision_history_publishes_consistently_to_distinct_parents():
    """题面：「配置未找到、JSON 无法解析或字段校验失败时，入口须在触及目标前以非零状态退出，两个旧目标保持原样。`--output` 和 `--checkpoint` 先转为绝对路径、解析符号链接并折叠 `.`、`..`，归一化后的落点不得相同；发生冲突同样在写入前失败并保留旧项。真正发布前，两个目标各自的父目录要完整写好以 `.sim-` 开头的临时文件，然后才开始替换。一次发布被视为不可拆分的双文件事务：任一临时文件无法创建或写完时两个目标不动；替换任一侧失败时，普通文件恢复旧字节，原本没有的项继续不存在，目录仍保持目录。无论成功或失败，两个父目录都不能留下 `.sim-` 开头的残留。」。"""
    # CASE LEG-008-B
    particles = [
        {"id": "a", "mass": 1.0, "radius": 0.11, "x": 0.4, "y": 0.5, "vx": 1.0, "vy": 0.0},
        {"id": "b", "mass": 2.0, "radius": 0.11, "x": 0.6, "y": 0.5, "vx": -0.5, "vy": 0.0},
    ]
    config = base_config(particles=particles, boundary="periodic", dt=1e-6, steps=3)
    directory = Path(tempfile.mkdtemp(prefix="sim-all-core-")); config_path = directory / "config.json"
    output_dir = directory / "out"; checkpoint_dir = directory / "cp"; output_dir.mkdir(); checkpoint_dir.mkdir()
    config_path.write_bytes(compact(config) + b"\n")
    after_one = copy.deepcopy(particles)
    after_one[0]["x"] += after_one[0]["vx"] * config["dt"]
    after_one[1]["x"] += after_one[1]["vx"] * config["dt"]
    after_one[0]["vx"] = -1.0; after_one[1]["vx"] = 0.5
    checkpoint = checkpoint_dir / "state.chk"
    checkpoint.write_bytes(compact(seal_checkpoint(config_path, config, after_one, step=1, max_error=4e-12, collision_count=1)) + b"\n")
    output = output_dir / "run.json"
    result = subprocess.run(["python3", str(SCRIPT), "--config", str(config_path), "--output", str(output), "--checkpoint", str(checkpoint)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text()); saved = json.loads(checkpoint.read_text())
    assert data["diagnostics"]["collision_count"] == 1 and data["diagnostics"]["max_relative_energy_error"] >= 4e-12
    assert data["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert saved["collision_count"] == 1 and saved["max_relative_energy_error"] == data["diagnostics"]["max_relative_energy_error"]
    assert not list(output_dir.glob(".sim-*")) and not list(checkpoint_dir.glob(".sim-*"))


def test_checkpoint_rejects_missing_extra_bad_types_ranges_ids_and_tampered_state():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    config = base_config(steps=2, seed=0, particles=[
        {"id": "p", "mass": 1.0, "radius": 0.02, "x": 0.2, "y": 0.5, "vx": 0.1, "vy": 0.0},
        {"id": "q", "mass": 2.0, "radius": 0.03, "x": 0.8, "y": 0.5, "vx": -0.1, "vy": 0.0},
    ])
    baseline, baseline_output, _ = run(config); assert baseline.returncode == 0
    baseline_data = json.loads(baseline_output.read_text())
    directory = Path(tempfile.mkdtemp(prefix="sim-cp-invalid-")); config_path = directory / "config.json"; config_path.write_bytes(compact(config) + b"\n")
    saved = copy.deepcopy(config["particles"]); saved[0]["x"] = 0.8
    valid = seal_checkpoint(config_path, config, saved, step=1, max_error=1e-12, collision_count=1)
    mutations = []
    for field in CHECKPOINT_ORDER:
        candidate = copy.deepcopy(valid); candidate.pop(field); mutations.append(("missing-" + field, candidate))
    candidate = copy.deepcopy(valid); candidate["extra"] = 1; mutations.append(("extra", candidate))
    changes = [
        ("version", 2), ("version", True), ("step", -1), ("step", 3), ("step", "1"), ("step", True),
        ("seed", 1), ("seed", False), ("config_sha256", "A" * 64), ("config_sha256", "f" * 63),
        ("initial_energy", -1), ("initial_energy", True), ("initial_energy", float("nan")), ("initial_energy", energy(config["particles"]) + 1),
        ("initial_momentum", [0.0]), ("initial_momentum", [True, 0.0]), ("initial_momentum", [0.0, float("inf")]), ("initial_momentum", [1.0, 0.0]),
        ("max_relative_energy_error", -1), ("max_relative_energy_error", True),
        ("max_relative_energy_error", float("inf")), ("max_relative_energy_error", "0"),
        ("max_relative_energy_error", 1.0000001e-10),
        ("collision_count", -1), ("collision_count", True), ("collision_count", 1.5), ("collision_count", 2),
        ("particles", []), ("state_sha256", "f" * 64), ("state_sha256", "f" * 63),
    ]
    for index, (field, value) in enumerate(changes):
        candidate = copy.deepcopy(valid); candidate[field] = value; mutations.append((f"change-{index}-{field}", candidate))
    particle_mutations = []
    for field in PARTICLE_ORDER:
        candidate = copy.deepcopy(valid); candidate["particles"][0].pop(field); particle_mutations.append(("particle-missing-" + field, candidate))
    candidate = copy.deepcopy(valid); candidate["particles"][0]["extra"] = 1; particle_mutations.append(("particle-extra", candidate))
    for case_id, field, value in (
        ("id-empty", "id", ""), ("id-type", "id", 7),
        ("mass-below", "mass", 9.99e-5), ("mass-above", "mass", 10000.0001),
        ("radius-below", "radius", 9.99e-7), ("radius-above", "radius", 1.000001),
        ("x-below", "x", -1e-9), ("x-above", "x", 1.000000001),
        ("y-below", "y", -1e-9), ("y-above", "y", 1.000000001),
        ("vx-bool", "vx", True), ("vy-infinite", "vy", float("inf")),
        ("vx-positive-above", "vx", 1e10 + 1), ("vx-negative-above", "vx", -1e10 - 1),
        ("vy-positive-above", "vy", 1e10 + 1), ("vy-negative-above", "vy", -1e10 - 1),
    ):
        candidate = copy.deepcopy(valid); candidate["particles"][0][field] = value; particle_mutations.append(("particle-bad-" + case_id, candidate))
    duplicate_ids = copy.deepcopy(valid); duplicate_ids["particles"][1]["id"] = duplicate_ids["particles"][0]["id"]; particle_mutations.append(("particle-duplicate-id", duplicate_ids))
    for name, candidate in particle_mutations:
        candidate["state_sha256"] = hashlib.sha256(compact(checkpoint_payload(candidate), sort_keys=True)).hexdigest()
        mutations.append((name, candidate))
    wrong_mass = copy.deepcopy(valid); wrong_mass["particles"][0]["mass"] = 1.5; wrong_mass["state_sha256"] = hashlib.sha256(compact(checkpoint_payload(wrong_mass), sort_keys=True)).hexdigest(); mutations.append(("wrong-mass-for-id", wrong_mass))
    wrong_radius = copy.deepcopy(valid); wrong_radius["particles"][0]["radius"] = 0.025; wrong_radius["state_sha256"] = hashlib.sha256(compact(checkpoint_payload(wrong_radius), sort_keys=True)).hexdigest(); mutations.append(("wrong-radius-for-id", wrong_radius))
    wrong_id = copy.deepcopy(valid); wrong_id["particles"][0]["id"] = "other"; wrong_id["state_sha256"] = hashlib.sha256(compact(checkpoint_payload(wrong_id), sort_keys=True)).hexdigest(); mutations.append(("wrong-id", wrong_id))
    tampered_history = copy.deepcopy(valid); tampered_history["collision_count"] += 1; mutations.append(("tampered-history", tampered_history))
    for name, candidate in mutations:
        checkpoint = directory / (name + ".chk"); output = directory / (name + ".json")
        checkpoint.write_bytes(compact(candidate) + b"\n")
        result = subprocess.run(["python3", str(SCRIPT), "--config", str(config_path), "--output", str(output), "--checkpoint", str(checkpoint)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT)
        assert result.returncode == 0, name
        assert json.loads(output.read_text())["final_state"] == baseline_data["final_state"], name
        assert json.loads(output.read_text())["diagnostics"]["collision_count"] == baseline_data["diagnostics"]["collision_count"], name
        pass_case("checkpoint-invalid-" + name)


def test_checkpoint_invalid_utf8_and_duplicate_keys_fall_back_silently():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    config = base_config(steps=2); baseline, baseline_output, _ = run(config); assert baseline.returncode == 0
    expected = json.loads(baseline_output.read_text())["final_state"]
    directory = Path(tempfile.mkdtemp(prefix="sim-cp-raw-"))
    raws = [
        b'', b'[]\n', b'null\n', b'{\n',
        b'{"version":1}\xff\n',
        b'{"version":1,"version":1,"step":0,"seed":19027,"config_sha256":"x","initial_energy":0,"initial_momentum":[0,0],"max_relative_energy_error":0,"collision_count":0,"particles":[],"state_sha256":"x"}\n',
        b'{"version":1,"step":0,"seed":19027,"config_sha256":"x","initial_energy":0,"initial_momentum":[0,0],"max_relative_energy_error":0,"collision_count":0,"particles":[{"id":"p","mass":1,"radius":0.1,"x":0.2,"y":0.2,"vx":0,"vy":0,"vy":1}],"state_sha256":"x"}\n',
        b" " * (MAX_JSON_BYTES + 1),
    ]
    for index, raw in enumerate(raws):
        checkpoint = directory / f"raw-{index}.chk"; checkpoint.write_bytes(raw)
        result, output, _ = run(config, checkpoint=checkpoint)
        assert result.returncode == 0 and json.loads(output.read_text())["final_state"] == expected
        pass_case(f"checkpoint-raw-invalid-{index}")
    missing = directory / "missing.chk"
    result, output, _ = run(config, checkpoint=missing)
    assert result.returncode == 0 and json.loads(output.read_text())["final_state"] == expected
    pass_case("checkpoint-missing-falls-back")


# Core challenge 3: two-target transactional publish.
def test_invalid_config_preserves_both_existing_targets():
    """题面：「入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。」。"""
    directory = Path(tempfile.mkdtemp(prefix="sim-invalid-publish-")); output = directory / "run.json"; checkpoint = directory / "state.chk"
    output.write_bytes(b"old output\n"); checkpoint.write_bytes(b"old checkpoint\n")
    result, _, _ = run(dict(base_config(), steps=True), output=output, checkpoint=checkpoint)
    assert result.returncode != 0
    assert output.read_bytes() == b"old output\n" and checkpoint.read_bytes() == b"old checkpoint\n"
    config_path = directory / "valid.json"; config_path.write_bytes(compact(base_config(steps=0)) + b"\n")
    same = directory / "same.json"; same.write_bytes(b"old same\n")
    result = subprocess.run(["python3", str(SCRIPT), "--config", str(config_path), "--output", str(same), "--checkpoint", str(same)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT)
    assert result.returncode != 0 and same.read_bytes() == b"old same\n"
    alias_dir = directory / "alias"; alias_dir.mkdir()
    alias = alias_dir / ".." / "same.json"
    result = subprocess.run(["python3", str(SCRIPT), "--config", str(config_path), "--output", str(same), "--checkpoint", str(alias)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT)
    assert result.returncode != 0 and same.read_bytes() == b"old same\n"
    symlink = directory / "same-link.json"
    symlink.symlink_to(same.name)
    result = subprocess.run(["python3", str(SCRIPT), "--config", str(config_path), "--output", str(same), "--checkpoint", str(symlink)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT)
    assert result.returncode != 0 and same.read_bytes() == b"old same\n" and symlink.is_symlink()
    pass_case("publish-output-checkpoint-symlink-alias")


def test_temp_creation_failure_in_either_parent_preserves_targets_and_cleans_up():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    for blocked_target in ("output", "checkpoint"):
        directory = Path(tempfile.mkdtemp(prefix="sim-temp-create-")); os.chmod(directory, 0o777)
        output_dir = directory / "out"; checkpoint_dir = directory / "cp"; output_dir.mkdir(); checkpoint_dir.mkdir()
        os.chmod(output_dir, 0o777); os.chmod(checkpoint_dir, 0o777)
        output = output_dir / "run.json"; checkpoint = checkpoint_dir / "state.chk"
        output.write_bytes(b"old output"); checkpoint.write_bytes(b"old checkpoint")
        blocked = output_dir if blocked_target == "output" else checkpoint_dir
        os.chmod(blocked, 0o555)
        config = directory / "config.json"; config.write_bytes(compact(base_config(steps=1)) + b"\n"); os.chmod(config, 0o644)
        result = subprocess.run(["python3", str(SCRIPT), "--config", str(config), "--output", str(output), "--checkpoint", str(checkpoint)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT, user="nobody")
        assert result.returncode != 0, blocked_target
        assert output.read_bytes() == b"old output" and checkpoint.read_bytes() == b"old checkpoint"
        assert not list(output_dir.glob(".sim-*")) and not list(checkpoint_dir.glob(".sim-*"))
        pass_case("publish-temp-create-failure-" + blocked_target)


def test_temp_write_failure_preserves_targets_and_cleans_up():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    directory = Path(tempfile.mkdtemp(prefix="sim-temp-write-")); output = directory / "run.json"; checkpoint = directory / "state.chk"
    output.write_bytes(b"old output"); checkpoint.write_bytes(b"old checkpoint")
    config = directory / "config.json"; config.write_bytes(compact(base_config(steps=1)) + b"\n")
    def limit_file_size():
        resource.setrlimit(resource.RLIMIT_FSIZE, (1, 1))
    result = subprocess.run(["python3", str(SCRIPT), "--config", str(config), "--output", str(output), "--checkpoint", str(checkpoint)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT, preexec_fn=limit_file_size)
    assert result.returncode != 0
    assert output.read_bytes() == b"old output" and checkpoint.read_bytes() == b"old checkpoint"
    assert not list(directory.glob(".sim-*"))


def test_first_replacement_failure_keeps_directory_and_other_file_unchanged():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    directory = Path(tempfile.mkdtemp(prefix="sim-first-replace-")); output = directory / "output-dir"; output.mkdir(); checkpoint = directory / "state.chk"
    checkpoint.write_bytes(b"old checkpoint")
    result, _, _ = run(base_config(steps=1), output=output, checkpoint=checkpoint)
    assert result.returncode != 0
    assert output.is_dir() and checkpoint.read_bytes() == b"old checkpoint"
    checkpoint.unlink()
    result, _, _ = run(base_config(steps=1), output=output, checkpoint=checkpoint)
    assert result.returncode != 0
    assert output.is_dir() and not checkpoint.exists()
    assert not list(directory.glob(".sim-*"))


def test_second_replacement_failure_rolls_back_regular_and_missing_output_states():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    directory = Path(tempfile.mkdtemp(prefix="sim-second-replace-")); checkpoint = directory / "checkpoint-dir"; checkpoint.mkdir()
    output = directory / "run.json"; output.write_bytes(b"old output")
    result, _, _ = run(base_config(steps=1), output=output, checkpoint=checkpoint)
    assert result.returncode != 0 and output.read_bytes() == b"old output" and checkpoint.is_dir()
    output.unlink()
    result, _, _ = run(base_config(steps=1), output=output, checkpoint=checkpoint)
    assert result.returncode != 0 and not output.exists() and checkpoint.is_dir()
    assert not list(directory.glob(".sim-*"))


def test_second_replacement_permission_failure_restores_both_regular_files():
    """题面：「随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。」。"""
    directory = Path(tempfile.mkdtemp(prefix="sim-second-permission-")); os.chmod(directory, 0o777)
    output_dir = directory / "out"; sticky_dir = directory / "sticky"; output_dir.mkdir(); sticky_dir.mkdir()
    os.chmod(output_dir, 0o777); os.chmod(sticky_dir, 0o1777)
    config = directory / "config.json"; config.write_bytes(compact(base_config(steps=1)) + b"\n"); os.chmod(config, 0o644)
    output = output_dir / "run.json"; checkpoint = sticky_dir / "state.chk"
    output.write_bytes(b"old output"); checkpoint.write_bytes(b"old checkpoint"); os.chmod(output, 0o666)
    result = subprocess.run(["python3", str(SCRIPT), "--config", str(config), "--output", str(output), "--checkpoint", str(checkpoint)], capture_output=True, text=True, timeout=PROCESS_TIMEOUT, user="nobody")
    assert result.returncode != 0
    assert output.read_bytes() == b"old output" and checkpoint.read_bytes() == b"old checkpoint"
    assert not list(output_dir.glob(".sim-*")) and not list(sticky_dir.glob(".sim-*"))


def test_successful_publish_leaves_no_temporary_files_in_distinct_parents():
    """题面：「配置未找到、JSON 无法解析或字段校验失败时，入口须在触及目标前以非零状态退出，两个旧目标保持原样。`--output` 和 `--checkpoint` 先转为绝对路径、解析符号链接并折叠 `.`、`..`，归一化后的落点不得相同；发生冲突同样在写入前失败并保留旧项。真正发布前，两个目标各自的父目录要完整写好以 `.sim-` 开头的临时文件，然后才开始替换。一次发布被视为不可拆分的双文件事务：任一临时文件无法创建或写完时两个目标不动；替换任一侧失败时，普通文件恢复旧字节，原本没有的项继续不存在，目录仍保持目录。无论成功或失败，两个父目录都不能留下 `.sim-` 开头的残留。」。"""
    # CASE LEG-008-A
    directory = Path(tempfile.mkdtemp(prefix="sim-clean-success-")); output_dir = directory / "out"; checkpoint_dir = directory / "cp"; output_dir.mkdir(); checkpoint_dir.mkdir()
    result, output, checkpoint = run(base_config(steps=1), output=output_dir / "run.json", checkpoint=checkpoint_dir / "state.chk")
    assert result.returncode == 0, result.stderr
    assert output.is_file() and checkpoint.is_file()
    assert not list(output_dir.glob(".sim-*")) and not list(checkpoint_dir.glob(".sim-*"))


def run_all():
    tests = [(name, value) for name, value in globals().items() if name.startswith("test_") and callable(value)]
    failures = []
    for name, test in tests:
        try:
            test()
            print("PASS " + name)
        except Exception as error:
            failures.append((name, error))
            print("FAIL " + name + ": " + repr(error))
    if failures:
        raise AssertionError(str(len(failures)) + " test(s) failed: " + ", ".join(name for name, _ in failures))


CASE_IDS_BY_TEST = {'test_entrypoint_exists_and_runs': ['LEG-001-A', 'LEG-001-B', 'LEG-009-A'], 'test_config_rejects_non_object_invalid_utf8_and_duplicate_keys_without_publish': ['LEG-009-B'], 'test_config_read_failure_exits_before_publish': [], 'test_config_requires_exact_top_level_and_particle_fields': [], 'test_config_types_ranges_boundary_and_unique_ids': ['LEG-004-A'], 'test_generated_data_is_deterministic_and_checkpoint_zero_state_is_valid': ['LEG-003-A', 'LEG-004-B', 'LEG-005-A', 'LEG-006-A'], 'test_output_checkpoint_schema_order_types_and_hashes': ['LEG-003-B', 'LEG-005-B', 'LEG-006-B'], 'test_output_preserves_input_order_and_all_diagnostics_are_finite': [], 'test_extreme_finite_input_still_produces_finite_json_numbers': [], 'test_repeat_from_final_checkpoint_is_byte_stable': [], 'test_box_boundary_reflects_once_and_across_multiple_walls_on_both_axes': [], 'test_periodic_wrap_and_minimum_image_collisions_across_x_and_y_seams': ['LEG-002-A', 'LEG-007-A'], 'test_each_step_advances_then_applies_boundary_then_resolves_collisions': ['LEG-002-B', 'LEG-007-B'], 'test_unequal_mass_oblique_collision_conserves_energy_and_momentum': [], 'test_utf8_pair_order_exact_overlap_and_once_per_step_are_deterministic': [], 'test_overlapping_but_separating_particles_do_not_collide': [], 'test_long_periodic_run_bounds_energy_drift_and_tracks_intermediate_peak': [], 'test_valid_checkpoint_reorders_particles_and_preserves_historical_diagnostics': [], 'test_resume_after_unequal_mass_collision_matches_continuous_files_byte_for_byte': [], 'test_resumed_collision_history_publishes_consistently_to_distinct_parents': ['LEG-008-B'], 'test_checkpoint_rejects_missing_extra_bad_types_ranges_ids_and_tampered_state': [], 'test_checkpoint_invalid_utf8_and_duplicate_keys_fall_back_silently': [], 'test_invalid_config_preserves_both_existing_targets': [], 'test_temp_creation_failure_in_either_parent_preserves_targets_and_cleans_up': [], 'test_temp_write_failure_preserves_targets_and_cleans_up': [], 'test_first_replacement_failure_keeps_directory_and_other_file_unchanged': [], 'test_second_replacement_failure_rolls_back_regular_and_missing_output_states': [], 'test_second_replacement_permission_failure_restores_both_regular_files': [], 'test_successful_publish_leaves_no_temporary_files_in_distinct_parents': ['LEG-008-A']}

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when == 'call':
        for case_id in CASE_IDS_BY_TEST.get(item.name, []):
            print(('PASS ' if report.passed else 'FAIL ') + case_id, flush=True)

if __name__ == "__main__":
    run_all()
