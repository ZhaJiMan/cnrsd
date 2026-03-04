"""读取一个 BUFR 文件，绘制 5 分钟累积的二维雨滴谱。"""

from __future__ import annotations

from pathlib import Path

import cmaps
import matplotlib.pyplot as plt

import cnrsd


def main() -> None:
    filepath = Path("sample/Z_SURF_I_53691_20240809000000_O_AWS-RSD-MM_FTM.BIN")
    rsd = cnrsd.read_file(filepath)
    da = rsd.to_dataarray()

    fig, ax = plt.subplots()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 12)

    pc = ax.pcolormesh(
        cnrsd.get_bin_edges(da.diameter_center, da.diameter_width),
        cnrsd.get_bin_edges(da.velocity_center, da.velocity_width),
        da.sum("time"),
        cmap=cmaps.WhiteBlueGreenYellowRed,
        ec=(0, 0, 0, 0.25),
        lw=0.5,
    )

    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"{rsd.station_id}\n{rsd.reference_time:%Y-%m-%d %H:%M}")
    fig.colorbar(pc)

    fig_dirpath = Path("images")
    fig_dirpath.mkdir(exist_ok=True)
    fig_filepath = fig_dirpath / f"{rsd.station_id}.png"
    fig.savefig(fig_filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
