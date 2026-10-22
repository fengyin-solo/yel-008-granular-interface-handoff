#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

PARTICLE_KEYS = ("id", "mass", "radius", "x", "y", "vx", "vy")
CONFIG_KEYS = ("dt", "steps", "seed", "boundary", "particles")
CHECKPOINT_KEYS = (
    "version", "step", "seed", "config_sha256", "initial_energy",
    "initial_momentum", "max_relative_energy_error", "collision_count",
    "particles", "state_sha256",
)
MAX_JSON_BYTES = 65536
MAX_STEPS = 20000
MAX_SEED = 9223372036854775807
MAX_ID_BYTES = 128
MAX_RELATIVE_ENERGY_ERROR = 1e-10


def finite_number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def strict_load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    raw = Path(path).read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("JSON input too large")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)


def compact_bytes(value, *, sorted_keys=False):
    return json.dumps(
        value,
        sort_keys=sorted_keys,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalize_particle(particle):
    return {key: particle[key] for key in PARTICLE_KEYS}


def kinetic_energy(particles):
    return math.fsum(
        0.5 * particle["mass"] * (particle["vx"] ** 2 + particle["vy"] ** 2)
        for particle in particles
    )


def momentum(particles):
    return [
        math.fsum(particle["mass"] * particle[key] for particle in particles)
        for key in ("vx", "vy")
    ]


def relative_error(value, initial):
    return abs(value - initial) / max(abs(initial), 1e-30)


def validate_particle(particle, identifiers, *, velocity_limit):
    if not isinstance(particle, dict) or set(particle) != set(PARTICLE_KEYS):
        raise ValueError("particle schema")
    identifier = particle["id"]
    if not isinstance(identifier, str) or not identifier or identifier in identifiers:
        raise ValueError("particle id")
    try:
        encoded_identifier = identifier.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("particle id Unicode") from error
    if len(encoded_identifier) > MAX_ID_BYTES:
        raise ValueError("particle id length")
    identifiers.add(identifier)
    for key in PARTICLE_KEYS[1:]:
        if not finite_number(particle[key]):
            raise ValueError("particle number")
    if not 1e-4 <= particle["mass"] <= 1e4:
        raise ValueError("particle mass")
    if not 1e-6 <= particle["radius"] <= 1:
        raise ValueError("particle radius")
    if abs(particle["vx"]) > velocity_limit or abs(particle["vy"]) > velocity_limit:
        raise ValueError("particle velocity")
    if not (0 <= particle["x"] <= 1 and 0 <= particle["y"] <= 1):
        raise ValueError("particle position")


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(CONFIG_KEYS):
        raise ValueError("config schema")
    if not finite_number(config["dt"]) or not 1e-9 <= config["dt"] <= 100:
        raise ValueError("dt")
    if type(config["steps"]) is not int or not 0 <= config["steps"] <= MAX_STEPS:
        raise ValueError("steps")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= MAX_SEED:
        raise ValueError("seed")
    if config["boundary"] not in ("box", "periodic"):
        raise ValueError("boundary")
    if not isinstance(config["particles"], list) or len(config["particles"]) > 64:
        raise ValueError("particles")
    identifiers = set()
    for particle in config["particles"]:
        validate_particle(particle, identifiers, velocity_limit=1e4)


def checkpoint_payload(checkpoint):
    return {
        "particles": checkpoint["particles"],
        "initial_energy": checkpoint["initial_energy"],
        "initial_momentum": checkpoint["initial_momentum"],
        "max_relative_energy_error": checkpoint["max_relative_energy_error"],
        "collision_count": checkpoint["collision_count"],
    }


def valid_checkpoint(checkpoint, config, config_hash):
    try:
        if not isinstance(checkpoint, dict) or set(checkpoint) != set(CHECKPOINT_KEYS):
            return False
        if type(checkpoint["version"]) is not int or checkpoint["version"] != 1:
            return False
        if type(checkpoint["step"]) is not int or not 0 <= checkpoint["step"] <= config["steps"]:
            return False
        if type(checkpoint["seed"]) is not int or checkpoint["seed"] != config["seed"]:
            return False
        if checkpoint["config_sha256"] != config_hash:
            return False
        if not isinstance(checkpoint["config_sha256"], str) or len(checkpoint["config_sha256"]) != 64:
            return False
        if not isinstance(checkpoint["state_sha256"], str) or len(checkpoint["state_sha256"]) != 64:
            return False
        if any(character not in "0123456789abcdef" for character in checkpoint["config_sha256"] + checkpoint["state_sha256"]):
            return False
        if not finite_number(checkpoint["initial_energy"]) or checkpoint["initial_energy"] < 0:
            return False
        if (not finite_number(checkpoint["max_relative_energy_error"])
                or not 0 <= checkpoint["max_relative_energy_error"] <= MAX_RELATIVE_ENERGY_ERROR):
            return False
        maximum_collisions = checkpoint["step"] * len(config["particles"]) * (len(config["particles"]) - 1) // 2
        if (type(checkpoint["collision_count"]) is not int
                or not 0 <= checkpoint["collision_count"] <= maximum_collisions):
            return False
        if not isinstance(checkpoint["initial_momentum"], list) or len(checkpoint["initial_momentum"]) != 2:
            return False
        if not all(finite_number(value) for value in checkpoint["initial_momentum"]):
            return False
        if checkpoint["initial_energy"] != kinetic_energy(config["particles"]):
            return False
        if checkpoint["initial_momentum"] != momentum(config["particles"]):
            return False
        if not isinstance(checkpoint["particles"], list):
            return False
        identifiers = set()
        for particle in checkpoint["particles"]:
            validate_particle(particle, identifiers, velocity_limit=1e10)
        if identifiers != {particle["id"] for particle in config["particles"]}:
            return False
        config_by_id = {particle["id"]: particle for particle in config["particles"]}
        if any(
            particle["mass"] != config_by_id[particle["id"]]["mass"]
            or particle["radius"] != config_by_id[particle["id"]]["radius"]
            for particle in checkpoint["particles"]
        ):
            return False
        return checkpoint["state_sha256"] == sha256(compact_bytes(checkpoint_payload(checkpoint), sorted_keys=True))
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        return False


def reflect_coordinate(particle, position_key, velocity_key):
    value = particle[position_key]
    if 0 <= value <= 1:
        return
    segment = math.floor(value)
    folded = value % 2.0
    particle[position_key] = folded if folded <= 1.0 else 2.0 - folded
    if segment % 2:
        particle[velocity_key] = -particle[velocity_key]


def step_particles(particles, dt, boundary):
    for particle in particles:
        particle["x"] += particle["vx"] * dt
        particle["y"] += particle["vy"] * dt
        if boundary == "periodic":
            particle["x"] %= 1.0
            particle["y"] %= 1.0
        else:
            for position_key, velocity_key in (("x", "vx"), ("y", "vy")):
                if particle[position_key] < 0:
                    particle[position_key] = -particle[position_key]
                    particle[velocity_key] = -particle[velocity_key]
                elif particle[position_key] > 1:
                    particle[position_key] = 2 - particle[position_key]
                    particle[velocity_key] = -particle[velocity_key]
    collisions = 0
    ordered = sorted(particles, key=lambda particle: particle["id"])
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            dx = right["x"] - left["x"]
            dy = right["y"] - left["y"]
            radius = left["radius"] + right["radius"]
            if dx * dx + dy * dy <= radius * radius:
                left["vx"], right["vx"] = right["vx"], left["vx"]
                left["vy"], right["vy"] = right["vy"], left["vy"]
                collisions += 1
    return collisions

def build_checkpoint(step, config, config_hash, particles, initial_energy, initial_momentum, maximum_error, collision_count):
    checkpoint = {
        "version": 1,
        "step": step,
        "seed": config["seed"],
        "config_sha256": config_hash,
        "initial_energy": initial_energy,
        "initial_momentum": initial_momentum,
        "max_relative_energy_error": maximum_error,
        "collision_count": collision_count,
        "particles": particles,
        "state_sha256": "",
    }
    checkpoint["state_sha256"] = sha256(compact_bytes(checkpoint_payload(checkpoint), sorted_keys=True))
    return checkpoint


def create_temp(target, data):
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".sim-", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return Path(name)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def snapshot(path):
    if path.is_file():
        return ("file", path.read_bytes())
    if path.exists():
        return ("other", None)
    return ("missing", None)


def restore(path, before):
    kind, data = before
    if kind == "file":
        path.write_bytes(data)
    elif kind == "missing":
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except FileNotFoundError:
            pass


def publish(output_path, output_bytes, checkpoint_path, checkpoint_bytes):
    output = Path(output_path)
    checkpoint = Path(checkpoint_path)
    output_temp = create_temp(output, output_bytes)
    checkpoint_temp = None
    try:
        os.replace(output_temp, output)
        output_temp = None
        checkpoint_temp = create_temp(checkpoint, checkpoint_bytes)
        os.replace(checkpoint_temp, checkpoint)
        checkpoint_temp = None
    finally:
        for temporary in (output_temp, checkpoint_temp):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    if Path(args.output).resolve(strict=False) == Path(args.checkpoint).resolve(strict=False):
        raise ValueError("output and checkpoint paths must differ")

    config_bytes = Path(args.config).read_bytes()
    config = strict_load(args.config)
    validate_config(config)
    config_hash = sha256(config_bytes)

    particles = [normalize_particle(particle) for particle in config["particles"]]
    start = 0
    initial_energy = kinetic_energy(particles)
    initial_momentum = momentum(particles)
    maximum_error = 0.0
    collision_count = 0
    try:
        checkpoint = strict_load(args.checkpoint)
        if valid_checkpoint(checkpoint, config, config_hash):
            by_id = {particle["id"]: particle for particle in checkpoint["particles"]}
            particles = [normalize_particle(by_id[particle["id"]]) for particle in config["particles"]]
            start = checkpoint["step"]
            initial_energy = kinetic_energy(particles)
            initial_momentum = momentum(particles)
            maximum_error = 0.0
            collision_count = 0
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        pass

    for _ in range(start, config["steps"]):
        collision_count += step_particles(particles, config["dt"], config["boundary"])
        maximum_error = max(maximum_error, relative_error(kinetic_energy(particles), initial_energy))

    final_energy = kinetic_energy(particles)
    final_momentum = momentum(particles)
    checkpoint = build_checkpoint(
        config["steps"], config, config_hash, particles,
        initial_energy, initial_momentum, maximum_error, collision_count,
    )
    checkpoint_bytes = compact_bytes(checkpoint) + b"\n"
    diagnostics = {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "max_relative_energy_error": maximum_error,
        "initial_momentum": initial_momentum,
        "final_momentum": final_momentum,
        "collision_count": collision_count,
    }
    result = {
        "steps": config["steps"],
        "final_state": particles,
        "diagnostics": diagnostics,
        "checkpoint_sha256": sha256(checkpoint_bytes),
    }
    publish(args.output, compact_bytes(result) + b"\n", args.checkpoint, checkpoint_bytes)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit("simulation failed: " + str(error))
