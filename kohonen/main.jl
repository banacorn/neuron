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
        # @show stripe
        xs = map((p) -> nodes[1,p[1],p[2]], stripe)
        ys = map((p) -> nodes[2,p[1],p[2]], stripe)
        line = layer(x = xs, y = ys, Geom.path,
            Theme(default_color = colorant"lightgrey", line_width = 0.5px))
        push!(layers, line)
    end

    for h in (1 - width):(height - 3)
        stripe = [ (h + w + 1, w) for w in 1:width ]
        stripe = filter((p) -> p[1] > 0 && p[1] <= height, stripe)
        # @show stripe
        xs = map((p) -> nodes[1,p[1],p[2]], stripe)
        ys = map((p) -> nodes[2,p[1],p[2]], stripe)
        line = layer(x = xs, y = ys, Geom.path,
            Theme(default_color = colorant"lightgrey", line_width = 0.5px))
        push!(layers, line)
    end
    return layers
end

function plotMap(m :: Map{Linear}, n = 10000, input = [])
    stack = []
    if length(input) > 0
        push!(stack, layer(x = input[1,:], y = input[2,:], Geom.point,
            Theme(default_color = colorant"coral", default_point_size = 1px)))
    end
    push!(stack, layer(x = m.nodes[1,:], y = m.nodes[2,:], Geom.point))
    push!(stack, layer(x = m.nodes[1,:], y = m.nodes[2,:], Geom.path,
        Theme(default_color = colorant"lightgrey", line_width = 0.5px)))
    return plot(stack...,
        Theme(background_color = colorant"white"),
        Coord.cartesian(fixed = true),
        Scale.x_continuous(minvalue = 0, maxvalue = 1),
        Scale.y_continuous(minvalue = 0, maxvalue = 1),
        Guide.title("n = $n"))
        # Guide.title("Self-organizing map, $(div(length(m.nodes),2)) clusters\nEuclidean distance (2 neighbors)\n$n data points, uniformly distributed"))
end

function plotMap(m :: Map{Manhattan}, n = 10000, input = [])
    nodes = reshape(m.nodes, (2, m.dimension[1], m.dimension[2]))

    stack = []
    if length(input) > 0
        push!(stack, layer(x = input[1,:], y = input[2,:], Geom.point,
            Theme(default_color = colorant"coral", default_point_size = 1px)))
    end
    push!(stack, layer(x = nodes[1,:,:], y = nodes[2,:,:], Geom.point))
    stack = [stack; drawOrthoganal(nodes)]
    return plot(stack...,
        Theme(background_color = colorant"white"),
        Coord.cartesian(fixed = true),
        Guide.title("n = $n"))
        # Guide.title("Self-organizing map, $(div(length(m.nodes),2)) clusters\nManhattan distance (4 neighbors)\n$n data points, uniformly distributed"))
end

function plotMap(m :: Map{Chebyshev}, n = 10000, input = [])
    nodes = reshape(m.nodes, (2, m.dimension[1], m.dimension[2]))
    stack = []
    if length(input) > 0
        push!(stack, layer(x = input[1,:], y = input[2,:], Geom.point,
            Theme(default_color = colorant"coral", default_point_size = 1px)))
    end
    push!(stack, layer(x = nodes[1,:,:], y = nodes[2,:,:], Geom.point))
    stack = [stack; drawOrthoganal(nodes)]
    stack = [stack; drawDiagonal(nodes)]
    return plot(stack...,
        Theme(background_color = colorant"white"),
        Coord.cartesian(fixed = true),
        Guide.title("n = $n"))
        # Guide.title("Self-organizing map, $(div(length(m.nodes),2)) clusters\nChebyshev distance (8 neighbors)\n$n data points, uniformly distributed"))
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

function update(m, vector, time, R = 3000, G = 1000)
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

function trainAndPlot(timestamp, input, m, R, G)
    stack = []

    # at 0
    push!(stack, plotMap(m, 0, input))

    current = 0
    for t in timestamp
        iterations = t - current
        for i in 1:iterations
            update(m, input[:,i + current], i, R, G)
        end
        push!(stack, plotMap(m, t))
        current = t
    end
    return stack
end


l = Map{Linear}(rand(2, 10 * 10), (100,))
lStack = trainAndPlot([25, 100, 500, 1000, 5000], rand(2, 10000), l, 100, 10000)
draw(PNG("linear.png", 1800px, 1200px), vstack(hstack(lStack[1:3]...), hstack(lStack[4:6]...)))

m = Map{Manhattan}(rand(2, 10 * 10), (10, 10))
mStack = trainAndPlot([25, 100, 500, 1000, 5000], rand(2, 10000), m, 3000, 1000)
draw(PNG("manhattan.png", 1800px, 1200px), vstack(hstack(mStack[1:3]...), hstack(mStack[4:6]...)))

c = Map{Chebyshev}(rand(2, 10 * 10), (10, 10))
cStack = trainAndPlot([25, 100, 500, 1000, 5000], rand(2, 10000), c, 3000, 1000)
draw(PNG("chebyshev.png", 1800px, 1200px), vstack(hstack(cStack[1:3]...), hstack(cStack[4:6]...)))

end
