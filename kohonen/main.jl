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




distanceSq = (w, v) -> ((w - v)' * (w - v))[1]

function findBMU(net, v)
    leastDist = distanceSq(net[:,1,1], v)
    leastNode = (1, 1)
    for i in eachindex(net[1,:,1])
        for j in eachindex(net[1,1,:])
            d = distanceSq(net[:,i,j], v)
            if d < leastDist
                leastDist = d
                leastNode = (i, j)
            end
        end
    end
    return leastNode
end

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
    width = size(nodes)[2]
    height = size(nodes)[1]

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
    # @show m.nodes[1,:]
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

function plotMap{Chebyshev}(m :: Map{Chebyshev})
    nodes = reshape(m.nodes, (2, m.dimension[1], m.dimension[2]))
    points = layer(x=nodes[1,:,:], y=nodes[2,:,:], Geom.point)
    orthoganal = drawOrthoganal(nodes)
    diagonal = drawDiagonal(nodes)
    return plot([points; orthoganal; diagonal]..., Theme(background_color = colorant"white"))
end

################################################################################
##  Functions
################################################################################

# exponential decay
sigma = (t) -> exp(-t)

distSqDecay = (distSq, t) -> exp(- distSq / (2 * sigma(t) * sigma(t)))

function adjustNet(net, v, time)
    BMU = findBMU(net, v)
    # timeConstant = 10

    # netRadius = size(net)[3] / 2
    # influRadius = netRadius * sigma(time)
    # # influRadius = netRadius * distSqDecay(dist, time)
    # # @printf "influence radius: %f\n" influRadius
    # for i in eachindex(net[1,:,1])
    #     for j in eachindex(net[1,1,:])
    #         node = net[:,i,j]
    #         nodeBMUDistSq = distanceSq(collect(BMU), [i, j])
    #         # @printf "distance from BMU (%d, %d) : %d\n" BMU[1] BMU[2] distanceSq(collect(BMU), [i, j])
    #         if nodeBMUDistSq <= influRadius * influRadius
    #             # @printf "(%d, %d)\t| [%f, %f] => " i j node[1] node[2]
    #             net[:,i,j] = node + sigma(time) * distSqDecay(nodeBMUDistSq, time) * (v - node)
    #             # @printf "(%d, %d)\t| [%f, %f] + %f * %f * [%f, %f] = [%f, %f]\n" i j  node[1] node[2] sigma(time) distSqDecay(nodeBMUDistSq, time) (v - node)[1] (v - node)[2] net[:,i,j][1] net[:,i,j][2]
    #             # node[1] node[2] sigma(time) diff[1] diff[2]
    #             # @printf "[%f, %f] \n" net[:,i,j][1] net[:,i,j][2]
    #         end
    #     end
    # end
    # @show BMU
    return net
end
# the weights of the net
# net = rand(2, 10, 10) * 10
#
# for t in 0:0.01:10
#     # input = rand(MvNormal([5.0, 5.0], [1.0 0.0; 0.0 1.0]))
#     input = [rand() * 10, rand() * 10]
#     net = adjustNet(net, input, t)
# end

# netPlot = plotNet(net)
# draw(PNG("som.png", 600px, 600px), netPlot)

l = Map{Linear}(rand(2, 10 * 10), (100,))
m = Map{Manhattan}(rand(2, 10 * 10), (10, 10))
c = Map{Chebyshev}(rand(2, 10 * 10), (10, 10))

draw(PNG("som.png", 600px, 600px), plotMap(l))

end
