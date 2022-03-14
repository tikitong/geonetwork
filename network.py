import momepy
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from pointpats import PointPattern
from shapely.geometry import LineString


def point_pattern(file: str) -> object:
    """ Generate PointPattern from geodesic coordinates.
    """

    gdf = gpd.read_file(file)
    mls = gdf.geometry
    coordinates = [list(x.centroid.coords) for x in mls]
    matrix = np.array(coordinates)
    data = np.reshape(matrix, (-1, 2))
    pp = PointPattern(data)

    return pp


def infoNetwork(frame: gpd.GeoDataFrame
                ) -> tuple:
    """ Generate nodes and edges with information.
    """

    Graph = momepy.gdf_to_nx(frame, approach="primal")
    centrality = momepy.closeness_centrality(Graph)
    degree = momepy.node_degree(Graph, name='degree')
    Full = nx.compose_all([centrality, degree])
    nodes, edges, sw = momepy.nx_to_gdf(Full, points=True, lines=True,
                                        spatial_weights=True)
    nodes['neighbors'] = pd.Series(sw.neighbors)

    return nodes, edges


def deduplicate(geo_data: np.ndarray
                ) -> np.ndarray:
    """Remove identical LineString data.
    """

    data = geo_data.reshape(-1, 2, 2)
    dt = f'f{data.itemsize}'  # f4 or f8
    data = data.view([('x', dt), ('y', dt)])
    # eliminate differences
    ixs = np.argsort(data, -2, order=('x', 'y'))
    data_no_df = np.take_along_axis(
        data, ixs, axis=-2)  # sorted by 'x' then by 'y'
    # get unique
    _, uni_ixs = np.unique(data_no_df, True, axis=0)
    uni_ixs.sort()  # inplace sort 1d-array
    # unique, originally ordered and shaped
    data_deduplicated = geo_data[uni_ixs]

    return data_deduplicated


def solution(coordinates: list) -> list:
    """Generate new edges (LineString) for all nodes. 
    """

    matrix = np.array(coordinates)
    result = deduplicate(matrix)
    final_result = [list(map(tuple, pair)) for pair in result.tolist()]
    lines = [{'geometry': LineString(pair)} for pair in final_result]

    return lines


def networker(pp: PointPattern, k: int) -> list:
    """Generate edges for the k nearest neighbours. 
    pp {PointPattern]: pp of points 
    k [int, optional]: number of nearest neighbours. Defaults to 2.

    Returns:
        list: list of LineString
    """

    coordinates = pp.points
    nearest_nodes = (pp.knn(k)[0]).tolist()
    total_lines = []
    for i in range(len(nearest_nodes)):
        nodes = [nodeID for nodeID in nearest_nodes[i]]
        first_node = (coordinates.at[i, 'x'], coordinates.at[i, 'y'])
        other_node = [(
            coordinates.at[value, 'x'], coordinates.at[value, 'y']) for value in nodes]
        line = [(first_node,) + (paire,) for paire in other_node]
        #lines = [{'geometry': LineString(pair)} for pair in line]
        total_lines += line

    return total_lines


def make_network(file: str, k: int = 2) -> None:
    """Generate geojson file with nodes, edges and data for each geometry data in the file 
    file [str]: geojson file to analyse. 
    k [int]: number of nearest neighbours. 
    """

    gdf = gpd.read_file(file)
    pp = point_pattern(file=file)
    rows = networker(pp=pp, k=k)
    output = solution(coordinates=rows)
    output = gpd.GeoDataFrame(output)
    nodes, edges = infoNetwork(frame=output)
    edges.to_file("edges.geojson", driver='GeoJSON')

    for row in gdf.itertuples():
        gdf.at[row.Index, 'geometry'] = row.geometry.centroid
    foundLabel = pd.merge(nodes, gdf[["label", "geometry"]], on=[
        "geometry"], how="left")

    for row in foundLabel.itertuples():
        foundLabel.at[row.Index, "neighbors_label"] = " ".join(
            f'{foundLabel.at[indice, "label"]}' for indice in row.neighbors)

    for row in foundLabel.itertuples():
        foundLabel.at[row.Index, 'neighbors'] = " ".join(
            str(x) for x in row.neighbors)

    foundLabel.to_file('nodes.geojson', driver='GeoJSON')


if __name__ == '__main__':
    make_network(file='dataset.geojson')
