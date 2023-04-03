import pathlib
import typing
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import nzshm_model as nm
import pandas as pd
import solvis


def get_ruptures(fault_system_solution, polygon):
    # get the ruptures of interest
    rupt_ids = fault_system_solution.get_ruptures_intersecting(polygon)
    print(f"area {whv_polygon} has {len(rupt_ids)} intersecting ruptures")

    # filter the rupture_with_rates dataframe
    rr = fault_system_solution.ruptures_with_rates
    rupts_df = rr[rr["Rupture Index"].isin(list(rupt_ids))]
    return rupts_df


def build_mfd(rupts_df, rate_column: str = "rate_weighted_mean"):
    # build the MFD
    bins = [round(x / 100, 2) for x in range(500, 1000, 10)]
    return rupts_df.groupby(pd.cut(rupts_df.Magnitude, bins=bins)).sum()[rate_column]


def plot_mfd(mfd: pd.DataFrame, title: str = "Title"):
    mag = [a.mid for a in mfd.index]
    rate = np.asarray(mfd)
    rate[rate == 0] = 1e-20  # set minimum rate for log plots

    fig = plt.figure()
    # ax = plt.subplot()

    fig.set_facecolor("white")

    plt.title(title)
    plt.ylabel("Incremental Rate ($yr^-1$)")
    plt.xlabel("Magnitude")

    # n, bins, patches = plt.hist(rate, 49, density=True, facecolor='r', alpha=0.75)
    plt.semilogy(mag, rate, color="red")  # ,linewidth=1) #, nonpositive='clip')
    # plt.bar(mag, rate, color='red')
    # ax.set_yscale('log')
    plt.axis([6.0, 9.0, 0.000001, 1.0])
    plt.grid(True)
    return plt


class FaultSystemBranchRuptureRates(typing.NamedTuple):
    solution_id: str
    weight: float
    values: typing.List[typing.Any]
    rates_df: pd.DataFrame


def fault_system_branches(
    sol: solvis.CompositeSolution, rupt_ids: typing.List[int], short_name: str = "CRU"
):
    for fslt in slt.fault_system_lts:
        print(fslt.short_name)
        cr = sol.composite_rates
        if fslt.short_name == short_name:
            cr = cr[cr.fault_system == short_name]
            for branch in fslt.branches:
                sol_id = branch.inversion_solution_id
                weight = branch.weight
                values = branch.values
                crs = cr[cr.solution_id == sol_id]
                crs = crs[crs["Rupture Index"].isin(rupt_ids)]
                yield FaultSystemBranchRuptureRates(sol_id, weight, values, crs)


if __name__ == "__main__":
    # setup some output folders
    output_folder = pathlib.Path("./OUTPUTS")
    geojson_folder = pathlib.Path(output_folder, "geojson")
    geojson_folder.mkdir(parents=True, exist_ok=True)
    branches_folder = pathlib.Path(output_folder, "branches")
    branches_folder.mkdir(parents=True, exist_ok=True)

    # Get the polygon supplied by Jeff
    f = pathlib.Path("Area of interest.kmz")
    kmz = zipfile.ZipFile(f, "r")
    kml = kmz.open("doc.kml")

    try:
        input_gdf = gpd.read_file(kml, driver="KML")
    except Exception:
        print('WARNING: we have to try/catch this once, otherwise the fiona KML setup fails (HUH ???)')
        pass
    finally:
        # Enable fionas KML driver
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
    kml = kmz.open("doc.kml")
    input_gdf = gpd.read_file(kml, driver="KML")

    whv_polygon = input_gdf.geometry[0]
    whv_polygon_buffered = whv_polygon.buffer(0.05)

    model = nm.get_model_version("NSHM_1.0.4")
    slt = model.source_logic_tree()
    comp = solvis.CompositeSolution.from_archive(
        pathlib.Path("NSHM_1.0.4_CompositeSolution.zip"), slt
    )

    fault_system_solution = comp._solutions["CRU"]

    # Composite report
    filtered_rates = get_ruptures(fault_system_solution, whv_polygon_buffered)
    filtered_rates.to_csv(
        pathlib.Path(output_folder, "NSHM_1.0.4_WHV_study_ruptures.csv")
    )

    mfd = build_mfd(filtered_rates)
    mfd.to_csv(pathlib.Path(output_folder, "NSHM_1.0.4_WHV_study_mfd.csv"))

    p = plot_mfd(mfd, title="NSHM_1.0.4 WHV_study MFD")
    p.savefig(pathlib.Path(output_folder, "NSHM_1.0.4_WHV_study_mfd_plot.png"))
    p.close()

    rupt_ids = list(
        fault_system_solution.get_ruptures_intersecting(whv_polygon_buffered)
    )

    # Rupture details as geojson
    for rupt_id in rupt_ids:
        solvis.export_geojson(
            fault_system_solution.rupture_surface(rupt_id),
            filename=pathlib.Path(geojson_folder, f"rupture_{rupt_id}.geojson"),
            indent=2,
        )

    # Dump the polygon area
    solvis.export_geojson(
        gpd.GeoDataFrame(geometry=[whv_polygon]),
        filename=pathlib.Path(geojson_folder, "area_of_interest.geojson"),
        indent=2,
    )

    # Dump the polygon area
    solvis.export_geojson(
        gpd.GeoDataFrame(geometry=[whv_polygon_buffered]),
        filename=pathlib.Path(geojson_folder, "area_of_interest_buffered.geojson"),
        indent=2,
    )

    # Branch reports
    for branch in fault_system_branches(comp, rupt_ids, "CRU"):
        # Join rupture_info
        fname = "NSHM_1.0.4_" + "_".join([str(v) for v in branch.values]).replace(
            ", ", "~"
        )

        ruptures_with_rates_df = branch.rates_df.join(
            fault_system_solution.ruptures.drop(columns="Rupture Index"),
            on=branch.rates_df["Rupture Index"],
        )
        ruptures_with_rates_df.to_csv(
            pathlib.Path(branches_folder, f"{fname}_ruptures.csv"), index=False
        )

        # build the mfd
        mfd = build_mfd(ruptures_with_rates_df, "Annual Rate")
        mfd.to_csv(pathlib.Path(branches_folder, f"{fname}_mfd.csv"))

        # plot the mfd
        p = plot_mfd(
            mfd, title=f"Branch: {branch.values} \n weight: {round(branch.weight, 6)}"
        )
        p.savefig(pathlib.Path(branches_folder, f"{fname}_mfd_plot.png"))
        p.close()
