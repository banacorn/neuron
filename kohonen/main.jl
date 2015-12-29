module Kohonen

using Distributions
using Gadfly
using RDatasets

typealias Value Float64
typealias Class Matrix{Value}
# # these are the data points we are going to cluster
# mvr0 = rand(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), 250)'
# mvr1 = rand(MvNormal([5.0, 0.0], [1.0 0.0; 0.0 4.0]), 250)'
# mvr2 = rand(MvNormal([5.0, 5.0], [4.0 0.0; 0.0 1.0]), 250)'
# mvr3 = rand(MvNormal([0.0, 5.0], [1.0 0.0; 0.0 1.0]), 250)'
# inputs = [mvr0; mvr1; mvr2; mvr3]





################################################################################
##  Datatypes
################################################################################

abstract Metric
abstract Manhattan <: Metric
abstract Chebyshev <: Metric
abstract Linear <: Metric

type Map{Metric}
    nodes :: Matrix{Float64}
    dimension :: Tuple
end

################################################################################
##  Plotting Map
################################################################################

function drawOrthoganal(nodes)
    layers = []

    # vertical lines
    for i in eachindex(nodes[1,:,1])
        line = layer(x = nodes[1,i,:], y = nodes[2,i,:], Geom.path,
            Theme(default_color = colorant"lightgrey", line_width = 0.5px))
        push!(layers, line)
    end

    # horizontal lines
    for i in eachindex(nodes[1,1,:])
        line = layer(x = nodes[1,:,i], y = nodes[2,:,i], Geom.path,
            Theme(default_color = colorant"lightgrey", line_width = 0.5px))
        push!(layers, line)
    end

    return layers
end
function drawDiagonal(nodes)
    layers = []

    # diagnal
    height = size(nodes)[2]
    width = size(nodes)[3]

    for h in 2:(height + width - 2)
        stripe = [ (h - w + 1, w) for w in 1:width ]
        stripe = filter((p) -> p[1] > 0 && p[1] <= height, stripe)
        xs = map((p) -> nodes[1,p[1],p[2]], stripe)
        ys = map((p) -> nodes[2,p[1],p[2]], stripe)
        line = layer(x = xs, y = ys, Geom.path,
            Theme(default_color = colorant"lightgrey", line_width = 0.5px))
        push!(layers, line)
    end

    for h in (1 - width):(height - width + 1)
        stripe = [ (h + w + 1, w) for w in 1:width ]
        stripe = filter((p) -> p[1] > 0 && p[1] <= height, stripe)
        xs = map((p) -> nodes[1,p[1],p[2]], stripe)
        ys = map((p) -> nodes[2,p[1],p[2]], stripe)
        line = layer(x = xs, y = ys, Geom.path,
            Theme(default_color = colorant"lightgrey", line_width = 0.5px))
        push!(layers, line)
    end
    return layers
end

function plotMap(m :: Map{Linear})
    points = layer(x = m.nodes[1,:], y = m.nodes[2,:], Geom.point)
    lines = layer(x = m.nodes[1,:], y = m.nodes[2,:], Geom.path,
        Theme(default_color = colorant"lightgrey", line_width = 0.5px))
    return plot([points; lines]..., Theme(background_color = colorant"white"))
end

function plotMap(m :: Map{Manhattan})
    nodes = reshape(m.nodes, (2, m.dimension[1], m.dimension[2]))
    points = layer(x = nodes[1,:,:], y = nodes[2,:,:], Geom.point)
    orthoganal = drawOrthoganal(nodes)
    return plot([points; orthoganal]..., Theme(background_color = colorant"white"))
end

function plotMap(m :: Map{Chebyshev})
    nodes = reshape(m.nodes, (2, m.dimension[1], m.dimension[2]))
    points = layer(x=nodes[1,:,:], y=nodes[2,:,:], Geom.point)
    orthoganal = drawOrthoganal(nodes)
    diagonal = drawDiagonal(nodes)
    return plot([points; orthoganal; diagonal]..., Theme(background_color = colorant"white"))
end

################################################################################
##  Metrics
################################################################################

function metric(m :: Map{Linear}, i, j)
    return j - i
end

function metric(m :: Map{Manhattan}, i, j)
    width = m.dimension[2]
    return abs(div(i,width) - div(j,width)) + abs(i%width - j%width)
end

function metric(m :: Map{Chebyshev}, i, j)
    width = m.dimension[2]
    return max(abs(div(i,width) - div(j,width)), abs(i%width - j%width))
end


################################################################################
##  Best Matching Unit
################################################################################

distanceSq = (w, v) -> ((w - v)' * (w - v))[1]

function findBMU(m, v)
    leastDist = distanceSq(m.nodes[:,1,1], v)
    leastNode = 1
    for i in eachindex(m.nodes[1,:])
        dist = distanceSq(m.nodes[:,i], v)
        if dist < leastDist
            leastDist = dist
            leastNode = i
        end
    end
    return leastNode
end

################################################################################
##  Update the Map
################################################################################

# exponential decay
sigma = (t) -> exp(-t)

radius(m::Map{Linear}) = div(m.dimension[1], 2)
radius(m::Map{Manhattan}) = div(max(m.dimension[1], m.dimension[2]), 2)
radius(m::Map{Chebyshev}) = div(max(m.dimension[1], m.dimension[2]), 2)

# Manhattan
R = 3000
G = 1000

# # Chebyshev
# R = 3000
# G = 1000

# # Linear
# R = 1000
# G = 10000

function update(m, vector, time)
    nodes = m.nodes
    bmu = findBMU(m, vector)
    # decreasing influence radius
    influenceRadius = radius(m) * sigma(time/R)

    if influenceRadius >= 1
        for i in eachindex(nodes[1,:])
            dist = metric(m, bmu, i)
            if dist <= influenceRadius
                gain = sigma(time/G) * sigma(dist/G)
                nodes[:,i] = nodes[:,i] + gain * (vector - nodes[:,i])
                # println("$time\traduis: $influenceRadius\tgain: $gain")
            end
        end
    else    # adjust BMU only
        # println("$time\tsigma: $(sigma(time/G))")
        nodes[:,bmu] = nodes[:,bmu] + sigma(time/G) * (vector - nodes[:,bmu])
    end
end

l = Map{Linear}(rand(2, 10 * 10), (100,))
m = Map{Manhattan}(rand(2, 10 * 10), (10, 10))
c = Map{Chebyshev}(rand(2, 10 * 10), (10, 10))

for time in 0:10000
    input = rand(2)
    update(m, input, time)
end
draw(PNG("som.png", 600px, 600px), plotMap(m))

end
