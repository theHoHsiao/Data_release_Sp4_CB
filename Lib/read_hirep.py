#!/usr/bin/env python3

from functools import lru_cache
import re
import logging

import numpy as np

from correlator import CorrelatorEnsemble, Correlator
from CB_correlator import CB_CorrelatorEnsemble, CB_Correlator


def add_metadata(metadata, line_contents):
    if (
        line_contents[0] == "[GEOMETRY][0]Global"
        or line_contents[0] == "[GEOMETRY_INIT][0]Global"
        or line_contents[0] == "[MAIN][0]global"
    ):
        NT, NX, NY, NZ = map(
            int,
            re.match("([0-9]+)x([0-9]+)x([0-9]+)x([0-9]+)", line_contents[3]).groups(),
        )
        metadata["NT"] = NT
        metadata["NX"] = NX
        metadata["NY"] = NY
        metadata["NZ"] = NZ


def parse_cfg_filename(filename):
    """
    Parse out the run name and trajectory index from a configuration filename.

    Arguments:

        filename: The configuration filename

    Returns:

        run_name: The name of the run/stream
        cfg_index: The index of the trajectory in the stream
    """

    matched_filename = re.match(
        r".*/([^/]*)_[0-9]+x[0-9]+x[0-9]+x[0-9]+nc[0-9]+(?:r[A-Z]+)?(?:nf[0-9]+)?b[0-9]+\.[0-9]+n([0-9]+)",
        filename,
    )
    run_name, cfg_index = matched_filename.groups()
    return run_name, cfg_index


def add_row(ensemble, split_line, stream_name, cfg_index, epsilon):
    """
    Add a single correlation function measurement to the ensemble.

    Arguments:

        ensemble: The CorrelatorEnsemble to append to.
        split_line: A list of strings, the split line of the input data.
        stream_name: The identifier of the Monte Carlo stream from
                     which the data are taken.
        cfg_index: The index of the configuration being measured on
                   within its Monte Carlo stream.
    """
    valence_mass = float(split_line[2][5:])

    try:
        _ = float(split_line[5])
    except ValueError:
        # Column 5 is a channel name, so source type is explicit
        source_type = split_line.pop(3)

    else:
        # Column 5 is a number, so source type is not stated
        source_type = "DEFAULT_SEMWALL"

    # print(split_line)
    connection_type = split_line[3]
    channel = split_line[4][:-1]
    correlator = np.asarray(split_line[5:], dtype=float)

    ensemble.append(
        Correlator(
            stream_name,
            cfg_index,
            source_type,
            epsilon,
            connection_type,
            channel,
            valence_mass,
            correlator,
        )
    )


def CB_add_row(ensemble, split_line, stream_name, cfg_index, epsilon_f, epsilon_as):
    """
    Add a single correlation function measurement to the ensemble.

    Arguments:

        ensemble: The CorrelatorEnsemble to append to.
        split_line: A list of strings, the split line of the input data.
        stream_name: The identifier of the Monte Carlo stream from
                     which the data are taken.
        cfg_index: The index of the configuration being measured on
                   within its Monte Carlo stream.
    """
    mf = float(split_line[2][3:])
    mas = float(split_line[3][4:])

    source_type = split_line.pop(4)

    channel = split_line[4][:-1]
    correlator = np.asarray(split_line[5:], dtype=float)

    ensemble.append(
        CB_Correlator(
            stream_name,
            cfg_index,
            source_type,
            epsilon_f,
            epsilon_as,
            channel,
            mf,
            mas,
            correlator,
        )
    )


@lru_cache(maxsize=8)
def read_correlators_hirep(filename):
    correlators_f = CorrelatorEnsemble(filename)
    correlators_as = CorrelatorEnsemble(filename)
    # correlators = CorrelatorEnsemble(filename)

    CB_correlators = CB_CorrelatorEnsemble(filename)

    # Track which configurations have provided results;
    # don't allow duplicates
    read_cfgs = set()

    with open(filename) as f:
        for line in f.readlines():
            line_contents = line.split()
            if (
                line_contents[0] == "[IO][0]Configuration"
                and line_contents[2] == "read"
            ):
                print(line_contents[1][1:-1])
                run_name, cfg_index = parse_cfg_filename(line_contents[1][1:-1])

                if (run_name, cfg_index) in read_cfgs:
                    logging.warn(
                        f"Possible duplicate data in {run_name} trajectory {cfg_index} of file {filename}"
                    )
                continue

            add_metadata(correlators_f.metadata, line_contents)
            add_metadata(correlators_as.metadata, line_contents)
            add_metadata(CB_correlators.metadata, line_contents)

            if line_contents[0] == "[SMEAR][0]Fundamental":
                epsilon_f = float(line_contents[5])
                continue

            if line_contents[0] == "[SMEAR][0]source":
                epsilon_as = float(line_contents[4])
                continue

            if line_contents[0] == "[MAIN][0]conf":
                read_cfgs.add((run_name, cfg_index))

                if line_contents[3][-4:] == "fund":
                    add_row(
                        correlators_f,
                        line_contents,
                        run_name,
                        int(cfg_index),
                        epsilon_f,
                    )

                elif line_contents[3][-4:] == "anti":
                    add_row(
                        correlators_as,
                        line_contents,
                        run_name,
                        int(cfg_index),
                        epsilon_as,
                    )

            if line_contents[0] == "[CHIMERA][0]conf":
                # CB_read_cfgs.add((run_name, cfg_index))
                CB_add_row(
                    CB_correlators,
                    line_contents,
                    run_name,
                    int(cfg_index),
                    epsilon_f,
                    epsilon_as,
                )

    correlators_f.freeze()
    correlators_as.freeze()
    CB_correlators.freeze()

    # if not correlators.is_consistent:
    #    logging.warning("Correlator is not self-consistent.")

    if not correlators_f.is_consistent:
        logging.warning("FUND. Correlator is not self-consistent.")
    if not correlators_f.is_consistent:
        logging.warning("ANTI. Correlator is not self-consistent.")
    if not correlators_f.is_consistent:
        logging.warning("CHIM. Correlator is not self-consistent.")

    return correlators_f, correlators_as, CB_correlators
