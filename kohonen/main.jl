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

function plotNet(net)
    layers = []
    # draw the nodes first
    push!(layers, layer(x=net[1,:,:], y=net[2,:,:], Geom.point))

    # connect the dots, vertical lines
    for i in eachindex(net[1,:,1])
        line = layer(x = net[1,i,:], y = net[2,i,:],
            Geom.line,
            Theme(default_color = colorant"lightgrey",
                line_width = 0.5px
            ))
        push!(layers, line)
    end

    # horizontal
    for i in eachindex(net[1,1,:])
        line = layer(x = net[1,:,i], y = net[2,:,i],
            Geom.line,
            Theme(default_color = colorant"lightgrey",
                line_width = 0.5px
            ))
        push!(layers, line)
    end

    # diagnal
    width = size(net)[3]
    height = size(net)[2]

    for h in 2:(height + width - 2)
        stripe = [ (h - w + 1, w) for w in 1:width ]
        stripe = filter((p) -> p[1] > 0 && p[1] <= height, stripe)
        xs = map((p) -> net[1,p[1],p[2]], stripe)
        ys = map((p) -> net[2,p[1],p[2]], stripe)
        line = layer(x = xs, y = ys,
            Geom.line,
            Theme(default_color = colorant"lightgrey",
                line_width = 0.5px
            ))
        push!(layers, line)
    end

    for h in (1 - width):(height - width + 1)
        stripe = [ (h + w + 1, w) for w in 1:width ]
        stripe = filter((p) -> p[1] > 0 && p[1] <= height, stripe)
        xs = map((p) -> net[1,p[1],p[2]], stripe)
        ys = map((p) -> net[2,p[1],p[2]], stripe)
        line = layer(x = xs, y = ys,
            Geom.line,
            Theme(default_color = colorant"lightgrey",
                line_width = 0.5px
            ))
        push!(layers, line)
    end

    return plot(layers..., Theme(background_color = colorant"white"))
end

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

# exponential decay
lambda = 1
sigma = (t) -> exp(-t / lambda)

distSqDecay = (distSq, t) -> exp(- distSq / (2 * sigma(t) * sigma(t)))

function adjustNet(net, v, time)
    BMU = findBMU(net, v)
    timeConstant = 10

    netRadius = size(net)[3] / 2
    influRadius = netRadius * sigma(time)
    # influRadius = netRadius * distSqDecay(dist, time)
    # @printf "influence radius: %f\n" influRadius
    for i in eachindex(net[1,:,1])
        for j in eachindex(net[1,1,:])
            node = net[:,i,j]
            nodeBMUDistSq = distanceSq(collect(BMU), [i, j])
            # @printf "distance from BMU (%d, %d) : %d\n" BMU[1] BMU[2] distanceSq(collect(BMU), [i, j])
            if nodeBMUDistSq <= influRadius * influRadius
                # @printf "(%d, %d)\t| [%f, %f] => " i j node[1] node[2]
                net[:,i,j] = node + sigma(time/10) * distSqDecay(nodeBMUDistSq, time/10) * (v - node)
                # @printf "(%d, %d)\t| [%f, %f] + %f * %f * [%f, %f] = [%f, %f]\n" i j  node[1] node[2] sigma(time) distSqDecay(nodeBMUDistSq, time) (v - node)[1] (v - node)[2] net[:,i,j][1] net[:,i,j][2]
                # node[1] node[2] sigma(time) diff[1] diff[2]
                # @printf "[%f, %f] \n" net[:,i,j][1] net[:,i,j][2]
            end
        end
    end
    # @show BMU
end
# the weights of the net
net = rand(2, 10, 10) * 10

for t in 0:0.01:10
    # input = rand(MvNormal([5.0, 5.0], [1.0 0.0; 0.0 1.0]))
    input = [rand() * 10, rand() * 10]
    adjustNet(net, input, t)
end


netPlot = plotNet(net)
draw(PNG("som.png", 600px, 600px), netPlot)
