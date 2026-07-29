# OTFS Fundamentals

This document gives a simple introduction to Orthogonal Time Frequency Space
(OTFS) and explains the signal flow used in POWDER-OTFS.

## Why OTFS when OFDM already exists?

OFDM sends QAM symbols on many subcarriers. It works very well when the wireless
channel changes slowly.

When a transmitter or receiver moves quickly, the received signal experiences
Doppler shifts. The channel can then change during an OFDM symbol, causing
inter-carrier interference and making channel estimation more difficult.

OTFS handles the data differently:

- OFDM places data directly in the **time-frequency domain**.
- OTFS first places data in the **delay-Doppler domain**.

Delay represents how late a reflected signal arrives. Doppler represents the
frequency shift caused by motion. These quantities describe the physical paths
between the transmitter and receiver.

OTFS still produces a time-domain waveform for transmission. The
delay-Doppler grid is an additional signal-processing layer that makes a
time-varying multipath channel easier to represent and estimate.

## Complete OTFS link

![Complete OTFS transmitter, channel, and receiver](images/otfs-system-model.svg)

## Step 1: Bits become QPSK symbols

The transmitter begins with binary information:

```text
00  01  11  10
```

QPSK maps every two bits to one complex symbol:

```text
00 -> (+1 + j) / sqrt(2)
01 -> (+1 - j) / sqrt(2)
11 -> (-1 - j) / sqrt(2)
10 -> (-1 + j) / sqrt(2)
```

The real part is the in-phase component and the imaginary part is the
quadrature component. The division by `sqrt(2)` gives the constellation unit
average power.

## Step 2: QPSK symbols fill the delay-Doppler grid

The QPSK symbols are placed into a two-dimensional array:

```text
X_DD has shape M x N
```

- `M` is the number of delay bins.
- `N` is the number of Doppler bins.
- Most bins contain QPSK data.
- One bin contains a known pilot.
- Bins around the pilot are set to zero and form the guard region.

The pilot is stronger than a normal data symbol. After the channel shifts and
scales it, the receiver uses the resulting pilot copies to estimate the channel
paths.

![Mapping bits to the delay-Doppler grid](images/otfs-grid-mapping.svg)

## Step 3: Delay-Doppler becomes time-frequency

The delay-Doppler grid is not transmitted directly. The ISFFT converts it into
a time-frequency grid.

In this project, the ISFFT applies:

1. an FFT along the delay axis;
2. an inverse FFT along the Doppler axis.

The result has the same `M x N` shape, but its meaning has changed:

- each row represents a subcarrier;
- each column represents a time slot.

Every delay-Doppler symbol contributes to multiple time-frequency samples. This
spreads each information symbol across the frame instead of assigning it to
only one time-frequency location.

## Step 4: Time-frequency becomes a waveform

The Heisenberg transform applies an inverse FFT across the subcarriers of every
time slot. This produces time-domain samples.

The columns are then placed one after another:

```text
time slot 0 samples
time slot 1 samples
...
time slot N-1 samples
```

The result is one complex waveform containing `M x N` samples. This waveform is
what the simulated channel receives and what a future USRP transmitter will
send.

## Step 5: The wireless channel changes the waveform

A received signal normally contains several copies of the transmitted
waveform. Each propagation path can have:

- a different delay;
- a different Doppler shift;
- a different complex gain.

The channel adds all path contributions and then adds complex AWGN:

```text
r(t) = sum of delayed, Doppler-shifted, scaled copies of s(t) + noise
```

The simulation uses circular integer-sample delays. This keeps the frame length
fixed and represents a cyclic-prefix-protected, timing-aligned frame. The OTA
waveform inserts a cyclic prefix before transmission and removes it at the
receiver.

### Fading

The path gains can use one of three models:

- **Fixed:** the configured complex gain does not change.
- **Rayleigh:** the gain changes randomly with no dominant line-of-sight path.
- **Rician:** a line-of-sight component is combined with random scattering.

For Rayleigh and Rician fading, the example generates new gains for every
frame.

## Step 6: The receiver returns to the delay-Doppler domain

The receiver reverses the transmitter transforms:

1. The **Wigner transform** converts the received waveform into a received
   time-frequency grid.
2. The **SFFT** converts the time-frequency grid into a received
   delay-Doppler grid.

The received grid is not yet the transmitted grid. It contains shifted and
scaled contributions from the channel paths, plus noise.

## Step 7: The embedded pilot estimates the paths

The receiver knows the transmitted pilot's position and complex value. It
examines only the configured observation region around that position. The zero
guard prevents data symbols from hiding the shifted pilot copies.

A channel path moves the pilot to another DD bin and multiplies it by a complex
gain. The receiver keeps bins whose magnitude satisfies:

```text
received magnitude >= threshold_factor * sqrt(noise variance)
```

For every retained bin, the receiver calculates:

```text
delay bin   = received delay position - pilot delay position
Doppler bin = received Doppler position - pilot Doppler position
```

The differences wrap around the grid boundaries. The received pilot is divided
by the known pilot value and the known OTFS phase term to estimate the complex
path coefficient. That coefficient contains both attenuation and phase.

The detected coefficients are stored in a sparse `M x N` delay-Doppler channel
response. Undetected positions remain zero. For example, three paths produce
three nonzero entries containing their estimated complex coefficients at their
estimated delay and Doppler positions.

The guard region is used to observe and detect the pilot copies. It is not the
final channel matrix.

## Step 8: The sparse response becomes the channel matrix

Equalization uses the vector model:

```text
y = Hx + n
```

The transmitter and receiver DD grids are flattened row by row:

```text
x = [X_DD[0,0], X_DD[0,1], ..., X_DD[M-1,N-1]]^T
y = [Y_DD[0,0], Y_DD[0,1], ..., Y_DD[M-1,N-1]]^T
```

Both vectors contain `MN` elements, so `H` has shape `MN x MN`. Each column of
`H` describes the received DD grid that would result from transmitting one
symbol at the corresponding position of `x`.

POWDER-OTFS constructs `H` using the reduced-cyclic-prefix OTFS structure:

1. For each detected delay tap, its `N` Doppler coefficients form the first
   column of an `N x N` circulant block.
2. Circularly shifting that first column creates the remaining columns of the
   block. This describes how that delay tap shifts symbols along the Doppler
   axis.
3. The block is placed between the appropriate input-delay and output-delay
   groups in `H`.
4. Known phase rotations are applied as the response moves across delay bins.
5. When a delay wraps around the end of the grid, the rectangular-pulse phase
   correction is applied.

Repeating this for every output delay and every detected path fills the complete
`MN x MN` matrix. The implementation follows the RCP-OTFS input-output
structure and validates the circulant construction against a basis-response
construction.

## Step 9: ZF or MMSE equalizes the channel

The received DD grid is flattened into `y`, and the equalizer estimates `x`.

Zero Forcing solves the least-squares problem:

```text
x_hat = arg min ||Hx - y||^2
```

ZF tries to invert the estimated channel directly. It can strongly amplify
noise when `H` is poorly conditioned.

MMSE solves:

```text
x_hat = (H^H H + (noise variance / symbol energy) I)^-1 H^H y
```

The extra noise-dependent term prevents unstable inversion and normally makes
MMSE more robust than ZF. The resulting vector is reshaped back into an `M x N`
DD grid. Only data positions are demodulated; the pilot and guard positions are
excluded.

## Integer and fractional delay-Doppler

The Doppler-bin spacing is:

```text
Doppler resolution = sample_rate / (M * N)
```

For the default `M=32`, `N=16`, and `sample_rate=1 MHz`, it is `1953.125 Hz`.

An integer Doppler path is an exact multiple of this resolution. An integer
delay is an exact sample delay. Such a path ideally produces one shifted pilot
copy at one DD bin.

A fractional path lies between grid positions. Its energy spreads across
neighboring delay or Doppler bins. The current estimator can interpret this
spread as several integer paths, producing an inaccurate `H`. Enlarging the
guard protects the pilot observation, but it does not remove fractional
spreading. Fractional delay-Doppler estimation is not yet implemented.

## Step 10: BER measures the recovered data

The bit error rate is:

```text
BER = incorrect recovered bits / total transmitted data bits
```

The pilot and guard bins are excluded because they do not carry user data.
The example accumulates errors over multiple OTFS frames.

Zero BER at high SNR is possible for the current grid-aligned simulation. It
does not guarantee zero BER with fractional Doppler, synchronization errors, or
real OTA hardware.

## OTFS and OFDM comparison

An OFDM baseline has not been implemented yet. A fair future comparison should
use the same:

- QAM order;
- bandwidth and sample rate;
- transmitted information bits;
- channel paths and noise realization;
- pilot overhead;
- number of frames.

The comparison should include BER versus SNR, BER versus Doppler or vehicle
speed, channel-estimation error, computational cost, and spectral efficiency.
