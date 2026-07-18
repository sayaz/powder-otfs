import numpy as np
from powder_otfs.visualization.plots import plot_otfs_pipeline
from powder_otfs.channel.awgn import add_awgn

from powder_otfs.modulation.qam import qam_modulate
from powder_otfs.otfs.grid import (
    create_delay_doppler_grid,
    insert_pilot_and_guards,
)
from powder_otfs.otfs.transforms import (
    heisenberg, 
    isfft,
    sfft,
    wigner,
)

from powder_otfs.visualization.plots import (
    plot_delay_doppler,
    plot_time_frequency,
    plot_waveform,
)


M = 32
N = 16

bits = np.random.randint(0, 2, M * N * 2)
symbols = qam_modulate(bits, order=4)

dd_grid = create_delay_doppler_grid(
    symbols,
    num_delay_bins=M,
    num_doppler_bins=N,
)

dd_grid = insert_pilot_and_guards(
    grid=dd_grid,
    pilot_delay_bin=M // 2,
    pilot_doppler_bin=N // 2,
    pilot_amplitude=2.0,
    guard_delay=1,
    guard_doppler=1,
)

pilot_delay_bin = M // 2
pilot_doppler_bin = N // 2

print(
    np.abs(
        dd_grid[
            pilot_delay_bin - 1 : pilot_delay_bin + 2,
            pilot_doppler_bin - 1 : pilot_doppler_bin + 2,
        ]
    )
)


tf_grid = isfft(dd_grid)
waveform = heisenberg(tf_grid)

rx_waveform = add_awgn(
    waveform,
    snr_db = 20.0,
)

recovered_tf_grid = wigner(
    rx_waveform,
    num_subcarriers = M,
    num_time_slots = N,
)

recovered_dd_grid = sfft(recovered_tf_grid)

print(
    "Maximum reconstruction error:",
    np.max(np.abs(dd_grid - recovered_dd_grid)),
)

plot_otfs_pipeline(
    dd_grid,
    tf_grid,
    waveform,
    recovered_dd_grid,
    recovered_tf_grid,
    rx_waveform,
)