from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import split

out_grid_count = 0

def divide_map_into_grids(x_, nx, ny):

    # 10km offset from center
    del_x = 0.12
    del_y = 0.09

    # compute minx, miny, maxx, maxy for 20 x 20 km from center
    x = [(x_[0] + x_[2]) / 2 - del_x, (x_[1] + x_[3]) / 2 - del_y,
         (x_[0] + x_[2]) / 2 + del_x, (x_[1] + x_[3]) / 2 + del_y]

    rec = [(x[0], x[1]), (x[0], x[3]), (x[2], x[3]), (x[2], x[1])]
    print(rec)
    polygon = Polygon(rec)

    # compute splitter
    minx, miny, maxx, maxy = polygon.bounds
    dx = (maxx - minx) / nx  # width of a small part
    dy = (maxy - miny) / ny  # height of a small part
    horizontal_splitters = [LineString([(minx, miny + i * dy), (maxx, miny + i * dy)]) for i in range(ny)]
    vertical_splitters = [LineString([(minx + i * dx, miny), (minx + i * dx, maxy)]) for i in range(nx)]
    splitters = horizontal_splitters + vertical_splitters

    # split
    result = polygon
    for splitter in splitters:
        result = MultiPolygon(split(result, splitter))
    grids = list(result.geoms)
    return grids

def ret_index(i, nx, ny):
    return (ny - int(i/nx))-1, int(i%nx)

def grid_index(grids, point):
    global out_grid_count
    for i in range(len(grids)):
        if point.within(grids[i]):
            return i
    out_grid_count += 1
    return -1