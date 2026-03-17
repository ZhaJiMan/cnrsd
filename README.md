# cnrsd

解析中国 BUFR 格式的雨滴谱文件

## 安装

```bash
pip install cnrsd

# 额外含有 pandas 和 xarray 依赖
pip install cnrsd[all]
```

## 基本用法

```python
import cnrsd

# 读取 BUFR 文件
filepath = "sample/Z_SURF_I_53691_20240809000000_O_AWS-RSD-MM_FTM.BIN"
rsd = cnrsd.read_file(filepath)

# 转换成 pandas 或 xarray 的数据类型，方便处理
df = rsd.to_dataframe()
da = rsd.to_dataarray()

# 批量转换类型
df = cnrsd.rsds_to_dataframe(rsds)
```

> 民间解读，直径速度可能有错，恳请指正。
> 以什么形式表示雨滴谱数据，方便保存和数据分析，也仍在探索当中，欢迎讨论。

示例：[examples/plot_station.py](examples/plot_station.py)

![plot_station.png](images/plot_station.png)

## TODO

- [ ] 样例文件
- [ ] 详细的 README
- [ ] 单元测试、CI
- [ ] 雨强、液态水含量

## 参考资料

- [雨滴谱观测数据BUFR编码格式（V1.0）](https://bbs.06climate.com/forum.php?mod=viewthread&tid=98567)
- WUSH-PW 型降水现象仪 技术手册
- 智能降水现象测量仪（双向）功能规格需求书
- [星小Py-017：以雨滴谱为例解析BUFR数据_2025.04.17](https://www.bilibili.com/video/BV1nw5azGEye/)
- [关于雨滴谱BURF数据解码的一些小心得](https://bbs.06climate.com/forum.php?mod=viewthread&tid=111686)
- [Operating instructions Present Weather Sensor OTT Parsivel 2](https://psl.noaa.gov/data/obs/instruments/OpticalDisdrometer.pdf)
- [Instruction for Use Laser Precipitation Monitor](https://ftp.cptec.inpe.br/chuva/read_me/alcantara/eq_disdrometer/esp_thies/thies_manual.pdf)
- [A Primer on Writing BUFR templates](https://www.eumetnet.eu/wp-content/uploads/2025/05/OPERA_BUFR_template_primer_V1.4.pdf)
