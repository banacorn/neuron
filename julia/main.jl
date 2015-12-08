using Distributions
using Gadfly
using RDatasets

function spiral(radius, theta, sequence)
    rs = map(radius, sequence)
    ts = map(theta, sequence)
    xs = map((r, t) -> r * sin(t), rs, ts)
    ys = map((r, t) -> r * cos(t), rs, ts)

    return [xs ys]
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

function mse(target, output)
    diff = target - output
    return sum(diff * diff) / 2
end

function learn(weights, input, target, learningRate = 0.8)
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
        sums = (w * d')'
        deltas[i] = sums .* o .* (1 - o)
    end

    for i in 1:n
        delta = [d * o for o = outputs[i], d = deltas[i]]
        if (all(abs(delta) .< 0.0001))
            weights[i] = weights[i] + delta * learningRate * 1000
        else
            weights[i] = weights[i] + delta * learningRate
        end
    end

    return Dict(
        "weights" => weights,
        "error" => sum(map(mse, target, last(outputs)))
    )
end

function train(weights, dataset, iteration = 1000)
    n = size(dataset, 1)
    totalSize = sum([size(class, 1) for class = dataset])
    errors = []
    for i in 1:iteration
        accumulatedError = 0
        for (j, class) in enumerate(dataset)    # for each class
            target = zeros((1, n))
            target[j] = 1
            m = size(class, 1)
            for k in 1:m
                result = learn(weights, [class[k,:] 1], target)    # augmented
                weights = result["weights"]
                accumulatedError = accumulatedError + result["error"]
            end
        end
        push!(errors, accumulatedError)
        if (accumulatedError < 0.005 * totalSize)
            break
        end
        if (i % 100 == 0)
            @show (i, last(errors))
        end
    end
    return Dict(
        "weights" => weights,
        "error" => errors
    )
end

function generateRandomWeights(layers)
    n = size(layers, 1)
    w = Array{Weight}[]
    for i in 1:(n-1)
        s = layers[i] * layers[i+1]
        push!(w, reshape(rand(s), layers[i], layers[i+1]))
    end
    return w
end

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
    return layer(data, x = "x", y = "y", color = "class", Geom.point,
        Theme(default_point_size = 3px, highlight_width = 1.5px))
end

function drawGrid(data)
    return layer(data, x = "x", y = "y", color = "class", Geom.rectbin)
end


function drawError(data)
    x = collect(1:length(data))
    return layer(x = x, y = data, Geom.line)
end



################################################################################
#   2
################################################################################


# PDF
spiralPDFS = Cairo.CairoPDFSurface("spiral.pdf", 600, 600)
spiralPDFC = Cairo.CairoContext(spiralPDFS)
spiralPDF = Compose.CAIROSURFACE(spiralPDFS)

# data points
spiral0 = spiral(i -> 6.5 * (104 - i) / 104,
    i -> pi * i / 16,
    40:96)

spiral1 = spiral(i -> 6.5 * (104 - i) / -104,
    i -> pi * i / 16,
    40:96)
spiralDataPoints = Class[spiral0, spiral1]

# weights
spiralWeights = generateRandomWeights([3, 80, 20, 10, 2])
spiralResult = train(spiralWeights, spiralDataPoints, 10000)
spiralWeights = spiralResult["weights"]
spiralErrors = spiralResult["error"]


# plot spirals
spiralDataSet = generateDataSet(spiralDataPoints)
spiralDataSetLayer = drawPoint(spiralDataSet)

# plot the grid
spiralGrid = generateGrid(collect(-8:0.1:8), spiralWeights)
spiralGridLayer = drawGrid(spiralGrid)

spiralPlot = plot(spiralDataSetLayer, spiralGridLayer,
    Coord.cartesian(fixed = true),
    Guide.title("The Two Spirals Problem"),
    Scale.color_discrete_manual(colorant"coral", colorant"cornflowerblue"))

spiralErrorsLayer = drawError(spiralErrors)
spiralErrorsPlot = plot(spiralErrorsLayer,
    Coord.Cartesian(ymin = 0),
    Guide.ylabel("Error"),
    Guide.xlabel("Iteration"))

draw(spiralPDF, spiralPlot)
Cairo.show_page(spiralPDFC)
draw(spiralPDF, spiralErrorsPlot)
Cairo.finish(spiralPDFS)


################################################################################
#   2
################################################################################

moonsPDFS = Cairo.CairoPDFSurface("moons.pdf", 600, 600)
moonsPDFC = Cairo.CairoContext(moonsPDFS)
moonsPDF = Compose.CAIROSURFACE(moonsPDFS)

# the double moon problem
n = 200
thetas = linspace(-180, 180, n)*pi/360;
r = 8
x0 = -5 + r * sin(thetas)' + randn(1, n)
y0 =      r * cos(thetas)' + randn(1, n)
x1 =  5 + r * sin(thetas)' + randn(1, n)
y1 =     -r * cos(thetas)' + randn(1, n)
moons = Class[[vec(x0) vec(y0)], [vec(x1) vec(y1)]]

#
moonsWeights = generateRandomWeights([3, 8, 8, 2])
moonsResult = train(moonsWeights, moons, 10000)
moonsWeights = moonsResult["weights"]
moonsErrors = moonsResult["error"]

# plot moons
moonsDS = generateDataSet(moons)
moonsDSLayer = drawPoint(moonsDS)

# plot the grid
moonsGrid = generateGrid(collect(-18:0.5:18), moonsWeights)
moonsGridLayer = drawGrid(moonsGrid)

moonsPlot = plot(moonsDSLayer, moonsGridLayer,
    Coord.cartesian(fixed = true),
    Guide.title("The Double Moons Problem"),
    Scale.color_discrete_manual(colorant"coral", colorant"cornflowerblue"))

moonsErrorsLayer = drawError(moonsResult["error"])
moonsErrorsPlot = plot(moonsErrorsLayer,
    Guide.ylabel("Error"),
    Guide.xlabel("Iteration"))

draw(moonsPDF, moonsPlot)
Cairo.show_page(moonsPDFC)
draw(moonsPDF, moonsErrorsPlot)
Cairo.finish(moonsPDFS)

################################################################################
#   3
################################################################################

# PDF
mvrPDFS = Cairo.CairoPDFSurface("mvrs.pdf", 600, 600)
mvrPDFC = Cairo.CairoContext(mvrPDFS)
mvrPDF = Compose.CAIROSURFACE(mvrPDFS)


mvr0 = rand(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), 100)'
mvr1 = rand(MvNormal([10.0, 0.0], [1.0 0.0; 0.0 4.0]), 100)'
mvr2 = rand(MvNormal([5.0, 10.0], [4.0 0.0; 0.0 1.0]), 100)'
mvr3 = rand(MvNormal([5.0, 5.0], [1.0 0.0; 0.0 1.0]), 100)'
mvrDataPoints = Class[mvr0, mvr1, mvr2, mvr3]

# weights
mvrWeights = generateRandomWeights([3, 10, 10, 4])
mvrResult = train(mvrWeights, mvrDataPoints, 10000)
mvrWeights = mvrResult["weights"]
mvrErrors = mvrResult["error"]


# plot distributions
mvrDataSet = generateDataSet(mvrDataPoints)
mvrDataSetLayer = drawPoint(mvrDataSet)

# plot the grid
mvrGrid = generateGrid(collect(-8:0.5:18), mvrWeights)
mvrGridLayer = drawGrid(mvrGrid)

mvrPlot = plot(mvrDataSetLayer, mvrGridLayer,
    Coord.cartesian(fixed = true),
    Guide.title("4 Multivariate Distributions"))
    # Scale.color_discrete_manual(colorant"coral", colorant"cornflowerblue"))

mvrErrorsLayer = drawError(mvrErrors)
mvrErrorsPlot = plot(mvrErrorsLayer,
    Coord.Cartesian(ymin = 0),
    Guide.ylabel("Error"),
    Guide.xlabel("Iteration"))

draw(mvrPDF, mvrPlot)
Cairo.show_page(mvrPDFC)
draw(mvrPDF, mvrErrorsPlot)
Cairo.finish(mvrPDFS)
