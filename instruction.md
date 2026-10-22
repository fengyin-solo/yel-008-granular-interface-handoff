# 交接班颗粒接缝复核

夜班交接前，接缝巡检要用已保存的碰撞现场续算到班次结束。现有 `/app/simulate.py` 虽能启动并产生文件，但斜碰、接缝接触、历史累计和双文件落盘在异常时会给出互相矛盾的结果。请保留命令行接口并修复这个复核器：

```bash
python3 /app/simulate.py --config /data/simulation.json --output /output/run.json --checkpoint /output/state.chk
```

入口先把配置读成 UTF-8 JSON 文档，原文最多 65536 字节，根节点必须为对象，重复键一律视为坏输入。对象只可含 `dt`、`steps`、`seed`、`boundary`、`particles` 五项：`dt` 取 `[1e-9,100]` 内有限数值；`steps` 为 0 到 20000 的 JSON 整数；`seed` 亦为 JSON 整数且落在 `[0,9223372036854775807]`，布尔值不能冒充整数；边界枚举只接受 `box` 与 `periodic`；粒子表的长度上限为 64，下界为零。

粒子对象由 `id`、`mass`、`radius`、`x`、`y`、`vx`、`vy` 七个键组成，不能少也不能多。`id` 非空且整张表中不可重复，UTF-8 表示最多 128 字节，并须排除 `U+D800` 至 `U+DFFF` 的代理码点；其余六项只能是有限 JSON 数值。质量限制在 `[1e-4,1e4]`，半径限制在 `[1e-6,1]`，横纵速度绝对值均不超过 `1e4`，位置初值均位于 `[0,1]`。合法配置的总能量与总动量必须能在 binary64 中保持有限；入口遇到合法数也不得悄悄截断。

推进节拍固定，单个节拍按“位置前进 `dt` → 边界处理 → 按 ID 的 UTF-8 字节序碰撞结算”这一顺序执行；同一对在同一节拍内不能重复结算。只有相互靠近且已经重叠的粒子才发生完全弹性二维碰撞，恢复系数为 1，总动量与总动能同时守恒。两球心完全重合时，法向从字节序较小的 ID 指向较大的 ID，并落在 x 轴正方向。`box` 模式中，坐标越过 0 或 1 要按逐次镜像处理，对应速度分量每次镜像都反号；一个节拍穿过多面墙也必须全部计入。`periodic` 模式将坐标按模 1 收回 `[0,1)`，两个轴上的球心差都取最短周期位移，因此 0/1 接缝远处的球也可能相撞；单轴原始差超过 `0.5` 时减去 1，低于 `-0.5` 时加上 1，恰好为正负 `0.5` 时方向符号不变。

报告中的 `diagnostics.initial_energy` 与 `diagnostics.initial_momentum` 永远取自配置起点，不能改成续算点。每个节拍完成推进、边界和碰撞后，用当前总动能相对起点能量求误差；`max_relative_energy_error` 覆盖起点及此前全部节拍的峰值，`collision_count` 则从第 0 步开始累计实际结算次数。若 `steps=0`，两项均为零。长跑即使发生多次碰撞，误差峰值也不得超过 `1e-10`。

续算文件同样是根为对象的 UTF-8 JSON，最多 65536 字节，出现重复键就失效。根键集合固定为 `version`、`step`、`seed`、`config_sha256`、`initial_energy`、`initial_momentum`、`max_relative_energy_error`、`collision_count`、`particles`、`state_sha256`。其中 `version` 只能为整数 1；`step` 为不超过配置上限的非负整数；`seed` 与配置一致；`config_sha256` 取配置原文的 64 位小写 SHA-256。续算粒子沿用配置中的字段、ID 与位置规则，速度绝对值放宽到 `1e10`，ID 集合必须和配置完全相同，同 ID 的 `mass`、`radius` 也必须逐值一致。`initial_energy` 为与配置起点相符的非负有限值，`initial_momentum` 为与起点相符的两个有限分量，`max_relative_energy_error` 位于 `[0,1e-10]`，`collision_count` 是不超过 `step*n*(n-1)/2` 的非负整数，`n` 表示粒子数。`state_sha256` 的输入对象仅含 `particles`、`initial_energy`、`initial_momentum`、`max_relative_energy_error`、`collision_count`，键名排序后写成 UTF-8 紧凑 JSON 再取 SHA-256。两个摘要都必须是 64 位小写十六进制。任何结构、类型、摘要、配置、种子、步数或 ID 检查失败，都从配置起点静默重跑，不采用该续算文件。

通过校验的续算必须继承文件里的起点能量、起点动量、误差峰值和碰撞累计，并依照配置表重排粒子。校验只检查公开的静态约束、摘要与范围，不要求为了证明轨迹可达而重放第 0 步。若中间文件确由本程序此前的成功运行产生，则续算后的 `run.json` 与最终续算文件应和“从零连续跑到同一终点”的两个文件逐字节一致；终点之前已经碰过球或创下误差峰值时也必须如此。

正常结束写出的 `run.json` 根键依次固定为 `steps`、`final_state`、`diagnostics`、`checkpoint_sha256`。`final_state` 按配置表原顺序保留全部球，单球键依次为 `id`、`mass`、`radius`、`x`、`y`、`vx`、`vy`。诊断对象键依次为 `initial_energy`、`final_energy`、`max_relative_energy_error`、`initial_momentum`、`final_momentum`、`collision_count`；除碰撞累计必须为非负整数外，能量、误差和动量分量都须为有限数值。`checkpoint_sha256` 对最终续算文件原文求 SHA-256。续算文件自身按前述根键次序书写，内部球的键序与 `final_state` 一致。两个结果都使用紧凑 UTF-8 JSON，并在末尾只放一个换行。

配置未找到、JSON 无法解析或字段校验失败时，入口须在触及目标前以非零状态退出，两个旧目标保持原样。`--output` 和 `--checkpoint` 先转为绝对路径、解析符号链接并折叠 `.`、`..`，归一化后的落点不得相同；发生冲突同样在写入前失败并保留旧项。真正发布前，两个目标各自的父目录要完整写好以 `.sim-` 开头的临时文件，然后才开始替换。一次发布被视为不可拆分的双文件事务：任一临时文件无法创建或写完时两个目标不动；替换任一侧失败时，普通文件恢复旧字节，原本没有的项继续不存在，目录仍保持目录。无论成功或失败，两个父目录都不能留下 `.sim-` 开头的残留。

随环境提供的 `environment/generate_data.py` 会用固定 seed 19027 写出起点配置和第 0 步续算文件，样本中两个粒子的质量不同。它是可重复的验收输入，不表示采集到了现场真实数据。
