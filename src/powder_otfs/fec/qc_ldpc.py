"""Deterministic quasi-cyclic accumulate LDPC encoder and decoder."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


SUPPORTED_RATES = ("1/2", "2/3", "3/4")
BASE_CODEWORD_BLOCKS = 36


@dataclass(frozen=True, slots=True)
class QCLDPCCode:
    """A systematic QC-accumulate LDPC code."""

    rate: str
    lifting_size: int
    information_blocks: int
    parity_blocks: int
    shifts: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def information_length(self) -> int:
        return self.information_blocks * self.lifting_size

    @property
    def codeword_length(self) -> int:
        return BASE_CODEWORD_BLOCKS * self.lifting_size

    def encode(self, information_bits: np.ndarray) -> np.ndarray:
        """Systematically encode one information block."""

        bits = np.asarray(information_bits, dtype=np.uint8)
        if bits.shape != (self.information_length,):
            raise ValueError(
                f"information_bits must contain {self.information_length} bits."
            )
        information = bits.reshape(
            self.information_blocks,
            self.lifting_size,
        )
        parity = np.zeros(
            (self.parity_blocks, self.lifting_size),
            dtype=np.uint8,
        )

        previous = np.zeros(self.lifting_size, dtype=np.uint8)
        for check_block, connections in enumerate(self.shifts):
            syndrome = np.zeros(self.lifting_size, dtype=np.uint8)
            for information_block, shift in connections:
                syndrome ^= np.roll(information[information_block], shift)
            parity[check_block] = syndrome ^ previous
            previous = parity[check_block]

        return np.concatenate((bits, parity.reshape(-1)))

    def parity_check(self, codeword: np.ndarray) -> np.ndarray:
        """Return the binary parity-check syndrome."""

        bits = np.asarray(codeword, dtype=np.uint8)
        if bits.shape != (self.codeword_length,):
            raise ValueError(
                f"codeword must contain {self.codeword_length} bits."
            )
        information = bits[:self.information_length].reshape(
            self.information_blocks,
            self.lifting_size,
        )
        parity = bits[self.information_length:].reshape(
            self.parity_blocks,
            self.lifting_size,
        )
        syndrome = np.zeros(
            (self.parity_blocks, self.lifting_size),
            dtype=np.uint8,
        )
        for check_block, connections in enumerate(self.shifts):
            for information_block, shift in connections:
                syndrome[check_block] ^= np.roll(
                    information[information_block],
                    shift,
                )
            syndrome[check_block] ^= parity[check_block]
            if check_block:
                syndrome[check_block] ^= parity[check_block - 1]
        return syndrome.reshape(-1)

    def decode(self, llrs: np.ndarray, maximum_iterations: int = 30) -> np.ndarray:
        """Decode one codeword using normalized min-sum iterations."""

        channel_llrs = np.asarray(llrs, dtype=float)
        if channel_llrs.shape != (self.codeword_length,):
            raise ValueError(
                f"llrs must contain {self.codeword_length} values."
            )
        if maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")

        checks = self._check_adjacency()
        variable_edges: list[list[tuple[int, int]]] = [
            [] for _ in range(self.codeword_length)
        ]
        check_messages = [np.zeros(len(row), dtype=float) for row in checks]
        variable_messages = []
        for check_index, row in enumerate(checks):
            messages = channel_llrs[row].copy()
            variable_messages.append(messages)
            for edge_index, variable in enumerate(row):
                variable_edges[int(variable)].append((check_index, edge_index))

        posterior = channel_llrs.copy()
        normalization = 0.80
        for _ in range(maximum_iterations):
            for check_index, incoming in enumerate(variable_messages):
                signs = np.where(incoming < 0.0, -1.0, 1.0)
                magnitudes = np.abs(incoming)
                minimum_index = int(np.argmin(magnitudes))
                minimum = float(magnitudes[minimum_index])
                if len(magnitudes) > 1:
                    second = float(np.partition(magnitudes, 1)[1])
                else:
                    second = minimum
                total_sign = float(np.prod(signs))
                outgoing = check_messages[check_index]
                for edge_index in range(len(incoming)):
                    excluded_minimum = second if edge_index == minimum_index else minimum
                    outgoing[edge_index] = (
                        normalization
                        * total_sign
                        * signs[edge_index]
                        * excluded_minimum
                    )

            posterior = channel_llrs.copy()
            for variable, edges in enumerate(variable_edges):
                posterior[variable] += sum(
                    check_messages[check][edge] for check, edge in edges
                )
            hard_bits = (posterior < 0.0).astype(np.uint8)
            if not np.any(self.parity_check(hard_bits)):
                return hard_bits[:self.information_length]

            for check_index, row in enumerate(checks):
                for edge_index, variable in enumerate(row):
                    variable_messages[check_index][edge_index] = (
                        posterior[int(variable)]
                        - check_messages[check_index][edge_index]
                    )

        return (posterior[:self.information_length] < 0.0).astype(np.uint8)

    def _check_adjacency(self) -> list[np.ndarray]:
        """Expand the QC description into sparse check-node adjacency."""

        rows: list[list[int]] = [
            [] for _ in range(self.parity_blocks * self.lifting_size)
        ]
        parity_offset = self.information_length
        for check_block, connections in enumerate(self.shifts):
            for local_row in range(self.lifting_size):
                row = rows[check_block * self.lifting_size + local_row]
                for information_block, shift in connections:
                    local_column = (local_row - shift) % self.lifting_size
                    row.append(
                        information_block * self.lifting_size + local_column
                    )
                row.append(
                    parity_offset
                    + check_block * self.lifting_size
                    + local_row
                )
                if check_block:
                    row.append(
                        parity_offset
                        + (check_block - 1) * self.lifting_size
                        + local_row
                    )
        return [np.asarray(row, dtype=int) for row in rows]


def create_qc_ldpc_code(rate: str, maximum_codeword_length: int) -> QCLDPCCode:
    """Create the largest supported QC-LDPC code fitting one OTFS frame."""

    if rate not in SUPPORTED_RATES:
        raise ValueError(f"rate must be one of {SUPPORTED_RATES}.")
    lifting_size = maximum_codeword_length // BASE_CODEWORD_BLOCKS
    if lifting_size < 2:
        raise ValueError("The OTFS data region is too small for QC-LDPC.")

    fraction = Fraction(rate)
    information_blocks = BASE_CODEWORD_BLOCKS * fraction.numerator // fraction.denominator
    parity_blocks = BASE_CODEWORD_BLOCKS - information_blocks
    connections: list[list[tuple[int, int]]] = [
        [] for _ in range(parity_blocks)
    ]
    for information_block in range(information_blocks):
        for edge in range(3):
            check_block = (
                information_block * 5
                + edge * 7
                + information_block // max(1, parity_blocks)
            ) % parity_blocks
            shift = (
                information_block * 7
                + edge * 11
                + check_block * 3
            ) % lifting_size
            connections[check_block].append((information_block, shift))

    return QCLDPCCode(
        rate=rate,
        lifting_size=lifting_size,
        information_blocks=information_blocks,
        parity_blocks=parity_blocks,
        shifts=tuple(tuple(row) for row in connections),
    )
