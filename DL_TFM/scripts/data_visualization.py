import geopandas as gpd
import numpy as np

from shapely.geometry import Point, Polygon


def plotCell(brdx, brdy, width, height):
    zipped = np.array(list(zip(brdx, brdy)))  # array with (x,y) pairs of cell border coordinates
    polygon = Polygon(zipped)  # create polygon

    interior = np.zeros(shape=(width, height), dtype=int)  # create all zero matrix
    for i in range(len(interior)):  # set all elements in interior matrix to 1 that actually lie within the cell
        for j in range(len(interior[i])):
            point = Point(i, j)
            if polygon.contains(point):
                interior[i][j] = 1

    p = gpd.GeoSeries(polygon)
    p.plot()


