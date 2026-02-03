import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    results: Dict[str, np.ndarray],
    title: str,
    outdir: Optional[str],
    show: bool,
    shock_day: int,
    shock_len: int,
) -> None:
    days = len(results["avg_tt"])
    x = np.arange(1, days + 1)

    fig = plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, results["avg_tt"])
    if shock_day > 0 and shock_len > 0:
        ax1.axvspan(shock_day, shock_day + shock_len - 1, color="red", alpha=0.12)
    ax1.set_title("Average Travel Time")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Time")

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x, results["route_switch"], label="route")
    ax2.plot(x, results["dep_switch"], label="departure")
    if shock_day > 0 and shock_len > 0:
        ax2.axvspan(shock_day, shock_day + shock_len - 1, color="red", alpha=0.12)
    ax2.set_title("#Switchings")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Count")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(x, results["rel_gap"])
    if shock_day > 0 and shock_len > 0:
        ax3.axvspan(shock_day, shock_day + shock_len - 1, color="red", alpha=0.12)
    ax3.set_title("Relative Gap")
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Gap")

    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(results["dep_counts"].T, aspect="auto", origin="lower")
    ax4.set_title("Departure Counts (window x day)")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Departure window")

    plt.suptitle(title)
    plt.tight_layout()

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        safe_title = "".join([c if c.isalnum() or c in "-_" else "_" for c in title])
        fig.savefig(os.path.join(outdir, f"{safe_title}.png"), dpi=200)

    if show:
        plt.show()

    plt.close(fig)

    daily_flow_global = results.get("daily_flow_global", None)
    od_offsets = results.get("od_offsets", None)
    od_sizes = results.get("od_sizes", None)
    if (
        isinstance(daily_flow_global, np.ndarray)
        and daily_flow_global.ndim == 2
        and isinstance(od_offsets, np.ndarray)
        and isinstance(od_sizes, np.ndarray)
        and od_offsets.size == od_sizes.size
        and daily_flow_global.shape[0] == days
    ):
        hhi_day = np.full(days, float("nan"), dtype=float)
        ent_day = np.full(days, float("nan"), dtype=float)

        for d in range(days):
            tot_w = 0.0
            hhi_w = 0.0
            ent_w = 0.0
            for off, sz in zip(od_offsets.tolist(), od_sizes.tolist()):
                off_i = int(off)
                sz_i = int(sz)
                if sz_i <= 0:
                    continue
                f = daily_flow_global[d, off_i : off_i + sz_i]
                tot = float(np.sum(f))
                if not np.isfinite(tot) or tot <= 0.0:
                    continue

                p = f / tot
                p = np.where(p > 0.0, p, 0.0)
                hhi = float(np.sum(p * p))

                if sz_i <= 1:
                    ent = 0.0
                else:
                    m = p > 0.0
                    ent_raw = -float(np.sum(p[m] * np.log(p[m]))) if np.any(m) else 0.0
                    ent = ent_raw / float(np.log(float(sz_i)))

                hhi_w += hhi * tot
                ent_w += ent * tot
                tot_w += tot

            if tot_w > 0.0:
                hhi_day[d] = hhi_w / tot_w
                ent_day[d] = ent_w / tot_w

        fig2 = plt.figure(figsize=(12, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(x, hhi_day)
        if shock_day > 0 and shock_len > 0:
            ax1.axvspan(shock_day, shock_day + shock_len - 1, color="red", alpha=0.12)
        ax1.set_title("HHI (concentration)")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("HHI")

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(x, ent_day)
        if shock_day > 0 and shock_len > 0:
            ax2.axvspan(shock_day, shock_day + shock_len - 1, color="red", alpha=0.12)
        ax2.set_title("Normalized entropy")
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Entropy")

        plt.suptitle(f"{title} (Herding)")
        plt.tight_layout()

        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            safe_title = "".join([c if c.isalnum() or c in "-_" else "_" for c in title])
            fig2.savefig(os.path.join(outdir, f"{safe_title}_herding.png"), dpi=200)

        if show:
            plt.show()

        plt.close(fig2)
