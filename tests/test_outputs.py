"""Acceptance tests for the anonymous-mass collision simulator."""

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


SCRIPT = Path(os.environ.get("SIM_SCRIPT", "/app/simulate.py"))
CONFIG = Path(os.environ.get("SIM_CONFIG", "/data/simulation.json"))
GENERATED_CHECKPOINT = Path(
    os.environ.get("SIM_GENERATED_CHECKPOINT", "/output/state.chk")
)
PROCESS_TIMEOUT = 30
TRUSTED_PYTHON = Path("/usr/local/bin/python3")
SETPRIV = Path("/usr/bin/setpriv")
PARTICLE_KEYS = ["id", "mass", "radius", "x", "y", "vx", "vy"]
CHECKPOINT_KEYS = [
    "version",
    "step",
    "seed",
    "config_sha256",
    "initial_energy",
    "initial_momentum",
    "max_relative_energy_error",
    "collision_count",
    "particles",
    "state_sha256",
]


def candidate_argv(*arguments):
    values = [str(value) for value in arguments]
    if SETPRIV.is_file() and TRUSTED_PYTHON.is_file():
        return [
            str(SETPRIV),
            "--reuid=65534",
            "--regid=65534",
            "--clear-groups",
            str(TRUSTED_PYTHON),
            "-I",
            "-B",
            *values,
        ]
    return [sys.executable, "-I", "-B", *values]


def expose_to_candidate(path):
    if not SETPRIV.is_file():
        return
    path.chmod(0o777)
    for parent in path.parents:
        if parent == Path("/tmp"):
            break
        parent.chmod(parent.stat().st_mode | 0o001)


def invoke_candidate(path, *arguments, timeout=PROCESS_TIMEOUT):
    expose_to_candidate(Path(path))
    return subprocess.run(
        candidate_argv(*arguments),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def compact(value, *, sort_keys=False):
    return json.dumps(
        value,
        sort_keys=sort_keys,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def base_config(**changes):
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value.pop("mass_checks", None)
    value.update(changes)
    return value


def identity_mass_checks(particles):
    identifiers = [particle["id"] for particle in particles]
    return [
        {
            "check_id": f"C{index:02d}",
            "coefficients": {
                identifier: 1 if identifier == particle["id"] else 0
                for identifier in identifiers
            },
            "total": particle["mass"],
        }
        for index, particle in enumerate(particles, 1)
    ]


def run_config(config, *, checkpoint=None, auto_checks=True):
    directory = Path(tempfile.mkdtemp(prefix="sim-case-"))
    expose_to_candidate(directory)
    encoded = copy.deepcopy(config)
    if auto_checks and "mass_checks" not in encoded:
        encoded["mass_checks"] = identity_mass_checks(encoded["particles"])
    config_path = directory / "config.json"
    output = directory / "run.json"
    checkpoint_path = Path(checkpoint) if checkpoint else directory / "state.chk"
    config_path.write_bytes(compact(encoded) + b"\n")
    result = invoke_candidate(
        directory,
        SCRIPT,
        "--config",
        config_path,
        "--output",
        output,
        "--checkpoint",
        checkpoint_path,
    )
    return result, output, checkpoint_path, config_path


def energy(particles):
    return math.fsum(
        0.5 * item["mass"] * (item["vx"] ** 2 + item["vy"] ** 2)
        for item in particles
    )


def momentum(particles):
    return [
        math.fsum(item["mass"] * item[key] for item in particles)
        for key in ("vx", "vy")
    ]


def state_hash(payload):
    return hashlib.sha256(compact(payload, sort_keys=True)).hexdigest()


def test_entrypoint_and_output_contract():
    # CASE REQ-019-01 REQ-020-01 REQ-022-02 REQ-024-01 REQ-026-01 REQ-026-02 REQ-026-03
    """题面：「演算成功不向终端吐字，两个结果直接落盘。」「`run.json` 的根键依次为 `steps`、`final_state`、`diagnostics`、`checkpoint_sha256`。」"""
    result, output, checkpoint, _ = run_config(base_config(steps=0))
    document = json.loads(output.read_text(encoding="utf-8"))
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert result.stdout == "" and result.stderr == ""
    assert list(document) == ["steps", "final_state", "diagnostics", "checkpoint_sha256"]
    assert list(document["diagnostics"]) == [
        "initial_energy",
        "final_energy",
        "max_relative_energy_error",
        "initial_momentum",
        "final_momentum",
        "collision_count",
    ]
    assert list(saved) == CHECKPOINT_KEYS
    assert output.read_bytes().endswith(b"\n") and output.read_bytes().count(b"\n") == 1


def test_config_schema_and_numeric_validation():
    # CASE REQ-001-01 REQ-003-01 REQ-004-01 REQ-007-01 REQ-008-01 REQ-010-01 REQ-014-01 REQ-016-01 REQ-021-02 REQ-025-01 REQ-025-02 REQ-025-03
    """题面：「全部检查必须形成满秩系统，并给出唯一质量因子。」"""
    config = base_config(steps=0)
    missing = copy.deepcopy(config)
    missing.pop("mass_checks", None)
    result, output, checkpoint, _ = run_config(missing, auto_checks=False)
    assert result.returncode != 0 and not output.exists()
    complete = dict(config, mass_checks=identity_mass_checks(config["particles"]))
    malformed_checks = []
    shorter = copy.deepcopy(complete)
    shorter["mass_checks"] = shorter["mass_checks"][:-1]
    malformed_checks.append(shorter)
    extra = copy.deepcopy(complete)
    extra["mass_checks"][0]["extra"] = 1
    malformed_checks.append(extra)
    duplicate_id = copy.deepcopy(complete)
    duplicate_id["mass_checks"][1]["check_id"] = duplicate_id["mass_checks"][0]["check_id"]
    malformed_checks.append(duplicate_id)
    empty_id = copy.deepcopy(complete)
    empty_id["mass_checks"][0]["check_id"] = ""
    malformed_checks.append(empty_id)
    long_id = copy.deepcopy(complete)
    long_id["mass_checks"][0]["check_id"] = "x" * 129
    malformed_checks.append(long_id)
    missing_coefficient = copy.deepcopy(complete)
    missing_coefficient["mass_checks"][0]["coefficients"].pop("p-b")
    malformed_checks.append(missing_coefficient)
    assert all(
        run_config(item, auto_checks=False)[0].returncode != 0
        for item in malformed_checks
    )
    invalid = []
    for patch in (
        {"dt": 0},
        {"dt": True},
        {"steps": -1},
        {"steps": 1.0},
        {"seed": -1},
        {"boundary": "reflect"},
    ):
        invalid.append(run_config(dict(config, **patch))[0].returncode)
    assert all(code != 0 for code in invalid)


def test_dense_mass_checks_recover_physical_masses():
    # CASE REQ-001-02 REQ-003-02 REQ-004-02 REQ-005-01 REQ-006-01 REQ-007-02 REQ-008-02 REQ-009-01 REQ-010-02 REQ-012-01 REQ-014-02 REQ-016-02 REQ-017-01
    """题面：「全部检查必须形成满秩系统，并给出唯一质量因子。」"""
    particles = [
        {"id": "p-a", "mass": 1.0, "radius": 0.01, "x": 0.1, "y": 0.1, "vx": 1.0, "vy": 0.0},
        {"id": "p-b", "mass": 2.0, "radius": 0.01, "x": 0.9, "y": 0.9, "vx": -1.0, "vy": 0.0},
    ]
    config = base_config(
        steps=1,
        boundary="box",
        particles=particles,
        mass_checks=[
            {
                "check_id": "MC-SUM",
                "coefficients": {"p-a": 1, "p-b": 1},
                "total": 3.0,
            },
            {
                "check_id": "MC-DIFF",
                "coefficients": {"p-a": 1, "p-b": -1},
                "total": 0.0,
            },
        ],
    )
    result, output, _, _ = run_config(config, auto_checks=False)
    document = json.loads(output.read_text(encoding="utf-8"))
    masses = {item["id"]: item["mass"] for item in document["final_state"]}
    assert result.returncode == 0
    assert masses == pytest.approx({"p-a": 1.5, "p-b": 1.5})
    assert document["diagnostics"]["initial_energy"] == pytest.approx(1.5)


def test_mass_checks_require_full_rank():
    # CASE REQ-005-02 REQ-006-02 REQ-009-02 REQ-012-02 REQ-017-02
    """题面：「全部检查必须形成满秩系统，并给出唯一质量因子。」"""
    payload = base_config(steps=0)
    payload["mass_checks"] = [
        {
            "check_id": "MC-A",
            "coefficients": {"p-a": 1, "p-b": 0},
            "total": 1.0,
        },
        {
            "check_id": "MC-B",
            "coefficients": {"p-a": 1, "p-b": 0},
            "total": 1.0,
        },
    ]
    result, output, _, _ = run_config(payload, auto_checks=False)
    assert result.returncode != 0
    assert not output.exists()


def test_particle_mass_range_endpoints_are_accepted():
    # CASE REQ-027-01
    """题面：「质量权重与球的几何尺度分别限制在 `[1e-4,1e4]`、`[1e-6,1]`。」"""
    particles = [
        {"id": "light", "mass": 1e-4, "radius": 1e-6, "x": 0.2, "y": 0.2, "vx": 0.0, "vy": 0.0},
        {"id": "heavy", "mass": 1e4, "radius": 1.0, "x": 0.8, "y": 0.8, "vx": 0.0, "vy": 0.0},
    ]
    result, output, _, _ = run_config(
        base_config(particles=particles, boundary="box", steps=0)
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert [item["mass"] for item in document["final_state"]] == pytest.approx([1e-4, 1e4])


def test_box_boundary_reflects_once_and_across_multiple_walls():
    # CASE REQ-002-01
    """题面：「`box` 模式下越过 0 或 1 的坐标逐次镜像，对应速度每次镜像都反号。」"""
    single = {
        "id": "single",
        "mass": 1.0,
        "radius": 0.01,
        "x": 0.75,
        "y": 0.5,
        "vx": 1.0,
        "vy": 0.0,
    }
    result, output, _, _ = run_config(
        base_config(particles=[single], boundary="box", dt=0.5, steps=1)
    )
    final = json.loads(output.read_text(encoding="utf-8"))["final_state"][0]
    assert result.returncode == 0
    assert final["x"] == pytest.approx(0.75)
    assert final["vx"] == pytest.approx(-1.0)
    multi = dict(single, id="multi", x=0.25, vx=10.0)
    result, output, _, _ = run_config(
        base_config(particles=[multi], boundary="box", dt=0.35, steps=1)
    )
    final = json.loads(output.read_text(encoding="utf-8"))["final_state"][0]
    assert result.returncode == 0
    assert final["x"] == pytest.approx(0.25)
    assert final["vx"] == pytest.approx(-10.0)


def test_periodic_seam_collision():
    # CASE REQ-011-02 REQ-013-02
    """题面：「`periodic` 模式下坐标按模 1 收回 `[0,1)`。」「每轴球心差使用最短周期位移。」"""
    particles = [
        {"id": "a", "mass": 1.0, "radius": 0.02, "x": 0.99, "y": 0.5, "vx": 1.0, "vy": 0.0},
        {"id": "b", "mass": 1.0, "radius": 0.02, "x": 0.01, "y": 0.5, "vx": -1.0, "vy": 0.0},
    ]
    result, output, _, _ = run_config(
        base_config(particles=particles, boundary="periodic", dt=0.01, steps=1)
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert document["diagnostics"]["collision_count"] == 1


def test_touching_but_not_approaching_particles_do_not_collide():
    # CASE REQ-011-03
    """题面：「只有已经重叠且互相靠近的两球发生完全弹性二维碰撞」。"""
    particles = [
        {"id": "left", "mass": 1.0, "radius": 0.1, "x": 0.4, "y": 0.5, "vx": 0.0, "vy": 0.0},
        {"id": "right", "mass": 1.0, "radius": 0.1, "x": 0.6, "y": 0.5, "vx": 0.0, "vy": 0.0},
    ]
    result, output, _, _ = run_config(
        base_config(particles=particles, boundary="periodic", dt=0.01, steps=1)
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert document["diagnostics"]["collision_count"] == 0
    assert document["final_state"] == particles


def test_elastic_collision_conserves_energy_and_momentum():
    # CASE REQ-002-02 REQ-011-01 REQ-013-01
    """题面：「总动量和总动能必须守恒。」"""
    particles = [
        {"id": "heavy", "mass": 3.0, "radius": 0.16, "x": 0.42, "y": 0.42, "vx": 0.2, "vy": 0.1},
        {"id": "light", "mass": 1.0, "radius": 0.16, "x": 0.58, "y": 0.58, "vx": -0.4, "vy": -0.3},
    ]
    result, output, _, _ = run_config(
        base_config(particles=particles, boundary="periodic", dt=1e-6, steps=1)
    )
    diagnostics = json.loads(output.read_text(encoding="utf-8"))["diagnostics"]
    assert result.returncode == 0
    assert diagnostics["collision_count"] == 1
    assert diagnostics["initial_energy"] == pytest.approx(diagnostics["final_energy"])
    assert diagnostics["initial_momentum"] == pytest.approx(diagnostics["final_momentum"])


def test_checkpoint_resume_matches_continuous_run():
    # CASE REQ-015-01 REQ-018-01 REQ-023-01
    """题面：「任一检查失败时忽略续算文件，从配置起点重跑。」"""
    config = base_config(steps=20)
    fresh, fresh_output, _, _ = run_config(config)
    directory = Path(tempfile.mkdtemp(prefix="sim-resume-"))
    directory.chmod(0o777)
    resume_checkpoint = directory / "state.chk"
    resume_checkpoint.write_bytes(GENERATED_CHECKPOINT.read_bytes())
    resume_checkpoint.chmod(0o644)
    resumed, resumed_output, _, _ = run_config(
        config, checkpoint=resume_checkpoint
    )
    assert fresh.returncode == 0 and resumed.returncode == 0
    assert fresh_output.read_bytes() == resumed_output.read_bytes()


def test_invalid_checkpoint_falls_back_to_start():
    # CASE REQ-015-02 REQ-018-02 REQ-023-02
    """题面：「任一检查失败时忽略续算文件，从配置起点重跑。」"""
    directory = Path(tempfile.mkdtemp(prefix="sim-bad-checkpoint-"))
    directory.chmod(0o777)
    checkpoint = directory / "bad.chk"
    checkpoint.write_text("{}\n", encoding="utf-8")
    checkpoint.chmod(0o644)
    config = base_config(steps=3)
    fresh, fresh_output, _, _ = run_config(config)
    fallback, fallback_output, _, _ = run_config(config, checkpoint=checkpoint)
    assert fresh.returncode == 0 and fallback.returncode == 0
    assert fresh_output.read_bytes() == fallback_output.read_bytes()


def test_invalid_config_preserves_existing_targets(tmp_path):
    # CASE REQ-021-01
    """题面：「输入被拒时两个旧目标保持原样。」"""
    config = tmp_path / "config.json"
    output = tmp_path / "run.json"
    checkpoint = tmp_path / "state.chk"
    config.write_text("{broken\n", encoding="utf-8")
    output.write_bytes(b"old output\n")
    checkpoint.write_bytes(b"old checkpoint\n")
    result = invoke_candidate(
        tmp_path,
        SCRIPT,
        "--config",
        config,
        "--output",
        output,
        "--checkpoint",
        checkpoint,
    )
    assert result.returncode != 0
    assert "traceback" not in result.stderr.lower()
    assert output.read_bytes() == b"old output\n"
    assert checkpoint.read_bytes() == b"old checkpoint\n"


def test_repeat_run_is_byte_identical():
    # CASE REQ-019-02 REQ-020-02 REQ-022-01 REQ-024-02
    """题面：「交付给下一班的两个输出，在同一班次配置下重新演算时须逐字节一致。」"""
    config = base_config(steps=5)
    first, first_output, first_checkpoint, _ = run_config(config)
    first_bytes = (first_output.read_bytes(), first_checkpoint.read_bytes())
    second, second_output, second_checkpoint, _ = run_config(config)
    assert first.returncode == 0 and second.returncode == 0
    assert (second_output.read_bytes(), second_checkpoint.read_bytes()) == first_bytes


def test_generated_sample_checkpoint_is_valid():
    """题面：「随环境提供的 `environment/generate_data.py` 用固定 seed 19027 生成两球样本和第 0 步续算文件，只用于接口验收。」"""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checkpoint = json.loads(GENERATED_CHECKPOINT.read_text(encoding="utf-8"))
    assert config["seed"] == 19027
    assert len(config["particles"]) == 2
    assert len(config["mass_checks"]) == 2
    assert list(checkpoint) == CHECKPOINT_KEYS
    assert checkpoint["step"] == 0
    assert checkpoint["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
