import glob
import h5py
import sys


sys.path.insert(1, "./Lib")
import read_hirep


def save_to_hdf5_meson(raw_data, lb, group_had):
    if len(raw_data.correlators) == 0:
        return

    if lb in list(group_had.keys()):
        group = group_had[lb]

    else:
        group = group_had.create_group(lb)

    for bare_mass in sorted(set(raw_data.correlators.valence_mass)):
        for source_type in sorted(set(raw_data.correlators.source_type)):
            Nconfg = raw_data.get_array(
                source_type=source_type, channel="g5", valence_mass=bare_mass
            ).shape[0]
            if Nconfg != 200:
                print(
                    "        imcomplete measurement: m_0 =",
                    bare_mass,
                    "N=",
                    Nconfg,
                    " skip...",
                    source_type,
                )
                continue

            if str(bare_mass) in list(group.keys()):
                mass_subgroup = group[f"{bare_mass}"]
            else:
                mass_subgroup = group.create_group(f"{bare_mass}")
                print("\ncreate mass subgroup m_0=", bare_mass)

            if source_type in list(mass_subgroup.keys()):
                continue
            else:
                print("        create smear subgroup: ", source_type)
                smear_subgroup = mass_subgroup.create_group(f"{source_type}")
                smear_subgroup.attrs["epsilon"] = str(raw_data.correlators.epsilon[0])

            for channel in set(raw_data.correlators.channel):
                subset_correlators = raw_data.correlators[
                    (raw_data.correlators.valence_mass == bare_mass)
                    & (raw_data.correlators.source_type == source_type)
                    & (raw_data.correlators.channel == channel)
                ].drop(columns=["channel", "valence_mass", "source_type"])

                if subset_correlators.empty:
                    print("empty", channel)
                    continue

                channel_subgroup = smear_subgroup.create_group(f"{channel}")

                channel_subgroup.create_dataset(
                    "config_indices", data=subset_correlators.cfg_index.to_numpy()
                )

                channel_subgroup.create_dataset(
                    "correlators",
                    data=raw_data.get_array(
                        source_type=source_type, channel=channel, valence_mass=bare_mass
                    ),
                )

    return


def save_to_hdf5_baryon(raw_data, lb, group_had):
    if len(raw_data.correlators) == 0:
        return

    if lb in list(group_had.keys()):
        group = group_had[lb]

    else:
        group = group_had.create_group(lb)

    for bare_mass_f in set(raw_data.correlators.mf):
        for bare_mass_as in set(raw_data.correlators.mas):
            for source_type in sorted(set(raw_data.correlators.source_type)):
                if (
                    raw_data.get_array(
                        source_type=source_type,
                        channel="Chimera_OC_even_re",
                        mf=bare_mass_f,
                        mas=bare_mass_as,
                    ).shape[0]
                    != 200
                ):
                    print(
                        "        imcomplete measurement: chimera",
                        bare_mass_f,
                        bare_mass_as,
                    )
                    continue

                if "mf" + str(bare_mass_f) in list(group.keys()):
                    mass_subgroup_f = group["mf" + f"{bare_mass_f}"]
                else:
                    mass_subgroup_f = group.create_group("mf" + f"{bare_mass_f}")
                    print("\ncreate mass subgroup mf_0=", bare_mass_f)

                if "mas" + str(bare_mass_as) in list(mass_subgroup_f.keys()):
                    mass_subgroup_as = mass_subgroup_f["mas" + f"{bare_mass_as}"]
                else:
                    mass_subgroup_as = mass_subgroup_f.create_group(
                        "mas" + f"{bare_mass_as}"
                    )
                    print("\n    create mass subgroup mas_0=", bare_mass_as)

                if source_type in list(mass_subgroup_as.keys()):
                    continue

                else:
                    print("        create smear subgroup: ", source_type)
                    smear_subgroup = mass_subgroup_as.create_group(f"{source_type}")
                    smear_subgroup.attrs["epsilon_f"] = str(
                        raw_data.correlators.epsilon_f[0]
                    )
                    smear_subgroup.attrs["epsilon_as"] = str(
                        raw_data.correlators.epsilon_as[0]
                    )

                for channel in set(raw_data.correlators.channel):
                    subset_correlators = raw_data.correlators[
                        (raw_data.correlators.mf == bare_mass_f)
                        & (raw_data.correlators.mas == bare_mass_as)
                        & (raw_data.correlators.source_type == source_type)
                        & (raw_data.correlators.channel == channel)
                    ].drop(columns=["channel", "mf", "mas", "source_type"])

                    if subset_correlators.empty:
                        print("empty", channel)
                        continue

                    channel_subgroup = smear_subgroup.create_group(f"{channel}")

                    channel_subgroup.create_dataset(
                        "config_indices", data=subset_correlators.cfg_index.to_numpy()
                    )

                    channel_subgroup.create_dataset(
                        "correlators",
                        data=raw_data.get_array(
                            source_type=source_type,
                            channel=channel,
                            mf=bare_mass_f,
                            mas=bare_mass_as,
                        ),
                    )
    return


def main():
    ENS = [
        "48x24x24x24b7.62",
        "60x48x48x48b7.7",
        "60x48x48x48b7.85",
        "60x48x48x48b8.0",
        "60x48x48x48b8.2",
    ]  #

    with h5py.File("data.h5", "w") as f:
        group_f = f.create_group("meson_fund")
        group_as = f.create_group("meson_anti")
        group_cb = f.create_group("chimera")

        for ens in ENS:
            path = "raw_data/" + ens + "/chimera_out_*"

            filenames = glob.glob(path)

            for name in filenames:
                print("\n reading...", name)

                a_f, a_as, a_CB = read_hirep.read_correlators_hirep(name)

                save_to_hdf5_meson(a_f, ens, group_f)
                save_to_hdf5_meson(a_as, ens, group_as)
                save_to_hdf5_baryon(a_CB, ens, group_cb)

if __name__ == "__main__":
    main()

