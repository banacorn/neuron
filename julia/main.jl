using Gadfly
using RDatasets

function spiral(radius, theta, sequence)
    rs = map(radius, sequence)
    ts = map(theta, sequence)
    xs = map((r, t) -> r * sin(t), rs, ts)
    ys = map((r, t) -> r * cos(t), rs, ts)
    return (xs, ys)
end

typealias Weight Float64
typealias Value Float64
typealias Point Vector{Value}
typealias Class Matrix{Value}
typealias DataSet Vector{Class}


sigmoid = n -> 1 / (1 + exp(-n))

unvec = (x) -> reshape(x, 1, length(x))

function computeOutput(weights, input)
    output = Matrix{Value}[input]
    for w in weights
        push!(output, map(sigmoid, last(output) * w))
    end
    return output
end

function classify(weights, input)
    output = input
    for w in weights
        output = map(sigmoid, output * w)
    end
    return indmax(output)
end

function learn(weights, input, target, learningRate = 0.5)
    n = size(weights, 1)
    outputs = computeOutput(weights, input)
    deltas = fill(Matrix{Value}(1, 1), n)

    # tabulate the output layer first
    o = last(outputs)
    deltas[n] = (target - o) .* o .* (1 - o)

    # tabulate from output to input
    for i in (n-1):-1:1
        o = outputs[i+1]
        w = weights[i+1]
        d = deltas[i+1]
        deltas[i] = unvec(w * vec(d))
    end

    for i in 1:n
        delta = [d * o for o = outputs[i], d = deltas[i]]
        weights[i] = weights[i] + delta
    end

    return Dict("weights" => weights)
end

function train(weights, dataset, iteration = 1000)
    n = size(dataset, 1)
    for i in 1:iteration
        for (j, class) in enumerate(dataset)    # for each class
            target = zeros((1, n))
            target[j] = 1
            m = size(class, 1)
            for k in 1:m
                result = learn(w, [class[k,:] 1], target)    # augmented
                weights = result["weights"]
            end
        end
    end
    return weights
end

# testing weights
w0 = Weight[0.3 0.4; 0.5 0.6; 0.4 0.5]
# w0 = Weight[0 1 2 3; 1 2 3 4; 3 4 5 6]
# w1 = Weight[1 2; 0 1; 1 2; 0 1]
w = Array[w0]

# data points
d0 = Class([1 1])
d1 = Class([1 -1; -1 -1; -1 1])
# d0 = Class([-2 1; 3 -1])
# d1 = Class([1 1; -1 -1])
d = Class[d0, d1]

w = train(w, d, 10000)

# the grid
function generateDataSet(dataset)
    result = DataFrame(x = [], y = [], class = [])
    for (i, class) in enumerate(dataset)
        result = [result; DataFrame(x = class[:,1], y = class[:,2], class = i)]
    end
    return result
end

function generateGrid(seq, weights)
    n = length(seq)
    xs = repeat(seq, inner=[n])
    ys = repeat(seq, outer=[n])
    rs = map((x, y) -> classify(weights, [x y 1]), xs, ys)
    return DataFrame(x = xs, y = ys, class = rs)
end

function drawPoint(data)
    return layer(data, x = "x", y = "y", color = "class", Geom.point)
end

function drawGrid(data)
    return layer(data, x = "x", y = "y", color = "class", Geom.rectbin)
end

ds = generateDataSet(d)
layerDS = drawPoint(ds)
grid = generateGrid(collect(-8:0.2:8), w)
layerGrid = drawGrid(grid)
plot(layerDS, layerGrid, Coord.cartesian(fixed = true), Guide.title("The Two Spirals Problem"))



# classify([1, 1, 0], w)


# # data points
# class0 = spiral(i -> 6.5 * (104 - i) / 104,
#     i -> pi * i / 8,
#     0:10)
#
# class1 = spiral(i -> 6.5 * (104 - i) / -104,
#     i -> pi * i / 8,
#     0:10)
# # typeof(dataset("datasets", "iris"))
#
# # the grid
#
#
# # # plotting spirals
# # s0 = layer(x = class0[1], y = class0[2], Geom.point, Theme(default_color = color("coral")))
# # s1 = layer(x = class1[1], y = class1[2], Geom.point, Theme(default_color = color("cornflowerblue")))
#
# # p0 = plot(s0, s1, Coord.cartesian(fixed = true), Guide.title("The Two Spirals Problem"))
