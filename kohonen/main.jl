module Kohonen

using Distributions
using Gadfly
using RDatasets

typealias Value Float64
typealias Class Matrix{Value}

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
        Scale.x_continuous(minvalue = 0, maxvalue = 1),
        Scale.y_continuous(minvalue = 0, maxvalue = 1),
        Guide.title("n = $n"))
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
        Scale.x_continuous(minvalue = 0, maxvalue = 1),
        Scale.y_continuous(minvalue = 0, maxvalue = 1),
        Guide.title("n = $n"))
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

radius(m::Map{Linear}) = div(m.dimension[1], 2)
radius(m::Map{Manhattan}) = div(max(m.dimension[1], m.dimension[2]), 2)
radius(m::Map{Chebyshev}) = div(max(m.dimension[1], m.dimension[2]), 2)

function update(m, vector, time, totalTime, R, G)
    nodes = m.nodes
    bmu = findBMU(m, vector)
    # decreasing influence radius
    influenceRadius = radius(m) * exp(-time/R)

    if influenceRadius >= 1
        for i in eachindex(nodes[1,:])
            dist = metric(m, bmu, i)
            if dist <= influenceRadius
                gain = exp(-time/G) * exp(-dist/G)
                nodes[:,i] = nodes[:,i] + gain * (vector - nodes[:,i])
            end
        end
    else    # adjust BMU only
        # rate = L * exp(-time/totalTime)
        nodes[:,bmu] = nodes[:,bmu] + exp(-time/G) * (vector - nodes[:,bmu])
    end
end

function trainAndPlot(timestamp, input, m, R, G)
    stack = []
    total = div(length(input), 2)
    # at 0
    push!(stack, plotMap(m, 0, input))

    current = 0
    for t in timestamp
        iterations = t - current
        for i in 1:iterations
            update(m, input[:,i + current], i + current, total, R, G)
        end
        push!(stack, plotMap(m, t))
        current = t
    end
    return stack
end


l = Map{Linear}(rand(2, 100), (100,))
lStack = trainAndPlot([25, 100, 500, 1000, 5000], rand(2, 5000), l, 100, 5000)
draw(PNG("linear.png", 1200px, 1800px), vstack(hstack(lStack[1:2]...), hstack(lStack[3:4]...), hstack(lStack[5:6]...)))

m = Map{Manhattan}(rand(2, 10 * 10), (10, 10))
mStack = trainAndPlot([25, 100, 500, 1000, 5000], rand(2, 5000), m, 3000, 1000)
draw(PNG("manhattan.png", 1200px, 1800px), vstack(hstack(mStack[1:2]...), hstack(mStack[3:4]...), hstack(mStack[5:6]...)))

c = Map{Chebyshev}(rand(2, 10 * 10), (10, 10))
cStack = trainAndPlot([25, 100, 500, 1000, 5000], rand(2, 5000), c, 3000, 1000)
draw(PNG("chebyshev.png", 1200px, 1800px), vstack(hstack(cStack[1:2]...), hstack(cStack[3:4]...), hstack(cStack[5:6]...)))


################################################################################
##  Plotting Cities
################################################################################

function ringMetric(tour, i, j)
    tourLength = div(length(tour), 3)
    return min((i - j + tourLength)%tourLength, (j - i + tourLength)%tourLength)
end

function plotTour(cities, tour)
    path = layer(x = tour[1,:], y = tour[2,:], Geom.path,
        Theme(default_color = colorant"lightgrey", line_width = 2px))
    bridge = layer(x = [tour[1,1], tour[1, end]],
        y = [tour[2,1], tour[2, end]], Geom.path,
        Theme(default_color = colorant"lightgrey", line_width = 2px))
    vertices = layer(x = tour[1,:], y = tour[2,:], Geom.point,
            Theme(default_point_size = 3px))
    cities = layer(x = cities[1,:], y = cities[2,:], Geom.point,
            Theme(default_color = colorant"coral", default_point_size = 5px))
    return plot([vertices; cities; path; bridge],
        Theme(background_color = colorant"white"),
        Coord.cartesian(fixed = true),
        Scale.x_continuous(minvalue = 0, maxvalue = 1),
        Scale.y_continuous(minvalue = 0, maxvalue = 1),
        Guide.title("TSP"))
end

function findBMV(tour, city)
    leastDist = distanceSq(tour[1:2,1], city)
    leastNode = 1
    for i in eachindex(tour[1,:])
        dist = distanceSq(tour[1:2,i], city)
        if dist < leastDist
            leastDist = dist
            leastNode = i
        end
    end
    return leastNode
end

function insertVertex(tour, i, newVertex)
    tourLength = div(length(tour), 3)
    firstHalf = tour[:,1:i]
    secondHalf = tour[:,i+1:tourLength]
    return [firstHalf newVertex secondHalf]
end

# 1 iteration
function updateTour(cities, tour, iteration, G, R)
    for c in eachindex(cities[1,:])
        city = cities[:,c]
        bmv = findBMV(tour, city)

        if tour[3,bmv] != iteration     # not visited
            # update the vertex and it's neighbors
            for i in eachindex(tour[1,:])
                dist = ringMetric(tour, bmv, i)
                gain = 0.707106781 * exp(-(dist^2)/G^2)
                tour[1:2,i] = tour[1:2,i] + gain * (city - tour[1:2,i])
            end

            # mark the best matching vertex visited
            tour[3,bmv] = iteration
        else
            # copy the best matching vertex, in case we need to insert it into the
            # tour as a new vertex
            newVertex = [tour[1:2,bmv]; iteration - 1]

            # update the vertex and it's neighbors
            for i in eachindex(tour[1,:])
                dist = ringMetric(tour, bmv, i)
                gain = 0.707106781 * exp(-(dist^2)/G^2)
                tour[1:2,i] = tour[1:2,i] + gain * (city - tour[1:2,i])
            end

            # mark the best matching vertex visited
            tour[3,bmv] = iteration

            tour = insertVertex(tour, bmv, newVertex)
        end
    end

    # delete old nodes
    suvivorIndices = []
    for v in eachindex(tour[1,:])
        if iteration - tour[3,v] < 3
            push!(suvivorIndices, v)
        end
    end
    tour = tour[:,suvivorIndices]
    return tour
end

function tourLength(tour)
    l = sqrt(distanceSq(tour[1:2,end], tour[1:2,1]))
    for v in eachindex(tour[1,1:end-1])
        l = l + sqrt(distanceSq(tour[1:2,v], tour[1:2,v+1]))
    end
    return l
end

function trainAndPlotHistogram(rate, iteration)
    cities = rand(2, 30)
    tour = [0.0, 0.0, 0]
    lengths = []
    # begin with 1 node at (0, 0), visited 0 times
    for i in 1:iteration
        if i % 100 == 0
            @show i
        end
        tour = [0.0, 0.0, 0]
        permCities = cities[:,randcycle(30)]
        G = 1
        R = 1/rate
        for j in 1:(rate + 5)
            tour = updateTour(permCities, tour, j, G, R)
            G = (1 - R) * G
        end
        push!(lengths,tourLength(tour))
    end

    minL = round(minimum(lengths), 3)
    maxL = round(maximum(lengths), 3)
    meanL = round(mean(lengths), 3)
    histo = plot(x=map((x) -> floor(x * 5)/5 , lengths), Geom.histogram,
        Theme(background_color = colorant"white"),
        Guide.title("$iteration tries, α = $(1/rate)\naverage: $meanL, lower bould: $minL, upper bould: $maxL"))
    draw(PNG("tour-$rate.png", 600px, 600px), plotTour(cities, tour))
    draw(PNG("histo-$rate.png", 600px, 300px), histo)
end

trainAndPlotHistogram(1000, 1000)

end
