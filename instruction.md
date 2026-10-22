# 接缝巡检的碰撞复核

夜班交接前，接缝巡检员要从一份已保存的现场续算到班次结束。现有 `/app/simulate.py` 可以启动，却没有恢复被匿名读数隐去的真实质量，碰撞和续算结果不可信。请保留入口：

```bash
python3 /app/simulate.py --config /data/simulation.json --output /output/run.json --checkpoint /output/state.chk
```

## 现场配置

交接配置的原始 JSON 不得超过 64 KiB，必须为 UTF-8，解码时出现重复成员即作废；顶层恰好包含：

```text
dt
steps
seed
boundary
particles
mass_checks
```

`dt` 是 `[1e-9,100]` 内的有限数。
`steps` 是 0 到 20000 的整数。
`seed` 是 0 到 9223372036854775807 的整数。
`boundary` 只能是 `box` 或 `periodic`。
布尔值不算整数或数值。

`particles` 最多 64 个。每颗球恰好包含：

```text
id
mass
radius
x
y
vx
vy
```

`id` 非空、整表不可重复、UTF-8 表示不超过 128 字节。
所有数值必须有限。
质量权重与球的几何尺度分别限制在 `[1e-4,1e4]`、`[1e-6,1]`。
速度绝对值不超过 `1e4`，位置位于 `[0,1]`。

`mass` 只是基准权重，不是物理质量。`mass_checks` 在粒子非空时恰好有同等数量的对象，空粒子表时是空数组。每项恰好包含：

```text
check_id
coefficients
total
```

每条检查的 `check_id` 在列表里只能出现一次。该标识不能为空串。标识不得长于 128 个编码字节。
`coefficients` 的键集合必须恰好等于全部粒子 ID。
每个系数是 `[-1000,1000]` 内的整数。
令未知质量因子为 `q_i`，每条检查要求：

```text
sum(coefficients[particle_id] * mass[particle_id] * q_i) = total
```

全部检查必须形成满秩系统，并给出唯一质量因子。
实际质量等于 `mass * q_i`，仍须落在 `[1e-4,1e4]`。
恢复失败时程序只以非零状态退出。

## 推进规则

一个节拍包含位置推进、边界处理和碰撞结算三个阶段：坐标先增加 `dt` 对应的位移，边界随后生效，碰撞按 ID 的 UTF-8 字节序处理。只有已经重叠且互相靠近的两球发生完全弹性二维碰撞，恢复系数为 1。总动量和总动能必须守恒。两球心重合时，法向从字节序较小的 ID 指向较大 ID 的正 x 方向。

`box` 模式下越过 0 或 1 的坐标逐次镜像，对应速度每次镜像都反号。
`periodic` 模式下坐标按模 1 收回 `[0,1)`。
每轴球心差使用最短周期位移。
原始差大于 `0.5` 减 1，小于 `-0.5` 加 1，恰好为正负 `0.5` 时方向不变。

## 续算与输出

接班检查站从同名路径读取一份 UTF-8 JSON。这张续算单恰好有十个根键。它们按下列先后顺序写出：

```text
version
step
seed
config_sha256
initial_energy
initial_momentum
max_relative_energy_error
collision_count
particles
state_sha256
```

`version` 为整数 1；`step` 不超过配置步数；`seed` 与配置相同；`config_sha256` 是配置原文的小写 SHA-256。
`state_sha256` 的输入对象只含 `particles`、`initial_energy`、`initial_momentum`、`max_relative_energy_error`、`collision_count` 五个键。
该对象按键名排序后写成紧凑 UTF-8 JSON 再取 SHA-256。
续算粒子必须与配置中的 ID、物理质量和半径一致。
续算位置仍位于 `[0,1]`，速度绝对值不超过 `1e10`。
任一检查失败时忽略续算文件，从配置起点重跑。

演算成功不向终端吐字，两个结果直接落盘。
`run.json` 的根键依次为 `steps`、`final_state`、`diagnostics`、`checkpoint_sha256`。
`final_state` 按配置顺序保留全部球。
诊断键依次为 `initial_energy`、`final_energy`、`max_relative_energy_error`、`initial_momentum`、`final_momentum`、`collision_count`。
`run.json` 与 `state.chk` 都必须是无填充空格的 UTF-8 JSON，而且各自只以单个换行收尾。

输入被拒时两个旧目标保持原样。
拒收过程不得在 stdout 或 stderr 打印 traceback。
交付给下一班的两个输出，在同一班次配置下重新演算时须逐字节一致。

随环境提供的 `environment/generate_data.py` 用固定 seed 19027 生成两球样本和第 0 步续算文件，只用于接口验收。
