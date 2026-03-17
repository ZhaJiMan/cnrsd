## 0.1.2 (ongoing)

- 增加对常量和站名更严格的检查
- 新增 `read_file` 函数，等价于 `RSD.from_file` 方法
- 新增 `gunn_kinzer_velocity` 函数，用经验公式计算液态雨滴的下落末速度
- `build_rsd_dataarray` 函数中若 `times` 或 `class_numbers` 中有重复的值，对应的粒子数现在会累加而不是直接覆盖

## 0.1.1 (2026-02-11)

第一版
