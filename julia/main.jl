
using Gadfly
using RDatasets

function spiral(radius, theta, sequence)
    rs = map(radius, sequence)
    ts = map(theta, sequence)
    xs = map((r, t) -> r * sin(t), rs, ts)
    ys = map((r, t) -> r * cos(t), rs, ts)
    return (xs, ys)
end

# data points
class0 = spiral(i -> 6.5 * (104 - i) / 104,
    i -> pi * i / 8,
    0:10)

class1 = spiral(i -> 6.5 * (104 - i) / -104,
    i -> pi * i / 8,
    0:10)
# typeof(dataset("datasets", "iris"))

# plotting spirals
l0 = layer(x = class0[1], y = class0[2], Geom.point, Theme(default_color = color("coral")))
l1 = layer(x = class1[1], y = class1[2], Geom.point, Theme(default_color = color("cornflowerblue")))
#
p0 = plot(l0, l1, Coord.cartesian(fixed = true))
