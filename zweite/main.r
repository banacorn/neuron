library("plyr")
library("MASS")

################################################################################
#   a
################################################################################
colours <- terrain.colors(5)

drawDataPoints <- function (inputs) {
    classA <- inputs[inputs$class == 1, ]
    classB <- inputs[inputs$class == 0, ]

    points(classA$x, classA$y, col = colours[1], lwd = 3)
    points(classB$x, classB$y, col = colours[2], lwd = 3)
}


drawDecisionBoundaryL <- function (weights, i = 1) {
    a <- weights[1]
    b <- weights[2]
    c <- weights[3]
    plot(NA, NA, xlim=c(-2,2), ylim=c(-2,2), xlab="x", ylab="y", col = colours[i])
    abline(-(c/b), -(a/b))
}

drawDecisionBoundaryQ <- function (weights, xlim = c(-2,2), ylim = c(-2,2), length = 10) {
    f <- Vectorize(function(x,y) sum(featureQ(c(x, y)) * weights))
    x <- seq(xlim[1], xlim[2], length = length)
    y <- seq(ylim[1], ylim[2], length = length)
    z <- outer(x,y,f)
    contour(
        x = x, y = x, z = z,
        levels = 0, las = 1, drawlabels = FALSE, lwd = 3,
        xlim=xlim, ylim=ylim, xlab="x", ylab="y", add = TRUE
    )
}
drawErrors <- function (errors) {
    plot(errors, xlab="iteration", ylab="error", type="s")
}


featureL <- function (input) {
    x <- unlist(input[1])
    y <- unlist(input[2])
    return(unlist(c(x, y, 1)))
}

featureQ <- function (input) {
    x <- unlist(input[1])
    y <- unlist(input[2])
    return(unlist(c(x * x, x * y, y * y, x, y, 1)))
}

# activation function
hardlimiter <- function (n) {
    if (n > 0) {
        return(1)
    } else {
        return(0)
    }
}

sigmoid <- function (n) 1/(1 + exp(-n))

id <- function (n) n

perceptron <- function (input, weights, feature, activate) {
    return(activate(sum(feature(input) * weights)))
}

classifyA <- function (feature, activation, learningRate, iteration, inputs, weights) {
    errors <- c()
    for (j in 1:iteration) {
        numberOfErrors <- 0
        for (i in 1:nrow(inputs)) {
            input <- inputs[i, ]
            output <- perceptron(input, weights, feature, activation)
            error <- input$class - output
            correction <- error * learningRate
            weights <- weights + feature(input) * correction
            if (error != 0)
                numberOfErrors <- numberOfErrors + 1
        }
        errors <- c(errors, numberOfErrors)
    }
    return(list(
        weights = weights,
        errors = errors
        ))
}


inputs <- data.frame(
    x = c(-1, -1, 1, 1),
    y = c(-1, 1, -1, 1),
    class = c(1, 0, 0, 1))


result <- classifyA(featureQ, hardlimiter, 0.1, 100, inputs, rep(0, 6))
plot(NA, NA, xlim=c(-2,2), ylim=c(-2,2), xlab="x", ylab="y")
drawDecisionBoundaryQ(result$weights)
drawDataPoints(inputs)
title("(a) dividing XOR with quadratic perceptrons")

drawErrors(result$errors)
title("error / iteration")

################################################################################
#   b
################################################################################

plotBoundary3 <- function(weights) {
    # vectors
    l01 <- weights[, 1] - weights[, 2]
    l12 <- weights[, 2] - weights[, 3]
    l20 <- weights[, 3] - weights[, 1]

    # cross point
    x <- ((l01[2]*l12[3])/l12[2] - l01[3]) / (l01[1] - (l12[1]*l01[2])/l12[2])
    y <- ((l01[1]*l12[3])/l12[1] - l01[3]) / (l01[2] - (l12[2]*l01[1])/l12[1])

    slope01 = l01[1]/(-l01[2])
    slope12 = l12[1]/(-l12[2])
    slope20 = l20[1]/(-l20[2])
    segments(x, y, x - 100, y - slope01 * 100)
    segments(x, y, x + 100, y + slope12 * 100)
    segments(x, y, x - 100, y - slope20 * 100)
}


tagClass <- function (inputs, class) {
    return(data.frame(x = inputs[, 1], y = inputs[, 2], class = class))
}

class0 = list(
    mean = c(0, 0),
    sigma = matrix(c(1, 0, 0, 1), 2, 2))

class1 = list(
    mean = c(10, 0),
    sigma = matrix(c(1, 0, 0, 4), 2, 2))

class2 = list(
    mean = c(5, 10),
    sigma = matrix(c(4, 0, 0, 1), 2, 2))

class0raw = mvrnorm(n = 100, class0$mean, class0$sigma)
class1raw = mvrnorm(n = 100, class1$mean, class1$sigma)
class2raw = mvrnorm(n = 100, class2$mean, class2$sigma)


class0 = tagClass(class0raw, 0)
class1 = tagClass(class1raw, 1)
class2 = tagClass(class2raw, 2)


weights = matrix(0, 3, 3)
learningRate <- 0.5
iteration <- 100
errors <- c()
for (j in 1:iteration) {    # interation
    size <- nrow(class0)
    numberOfErrors <- 0
    for (i in 1:size) { # number of data
        inputs <- matrix(c(class0[i, ], class1[i, ], class2[i, ]), 3, 3)
        numberOfClasses <- 3
        for (k in 1:numberOfClasses) { # each classes
            input <- inputs[, k]
            outputs <- c(perceptron(input, weights[, 1], featureL, hardlimiter),
                perceptron(input, weights[, 2], featureL, hardlimiter),
                perceptron(input, weights[, 3], featureL, hardlimiter))

            correction <- featureL(input) * learningRate
            for (m in 1:numberOfClasses) {
                if (m == k && outputs[m] == 0) { # reinforce
                    weights[, m] <- weights[, m] + correction
                    numberOfErrors <- numberOfErrors + 1
                }

                if (m != k && outputs[m] == 1) { # punish
                    weights[, m] <- weights[, m] - correction
                    numberOfErrors <- numberOfErrors + 1
                }
            }
        }
    }
    errors <- c(errors, numberOfErrors)
}
plot(NA, NA, xlim=c(-20,20), ylim=c(-20,20), xlab="x", ylab="y")
points(class0, col = colours[1])
points(class1, col = colours[2])
points(class2, col = colours[3])

plotBoundary3(weights)
title("(b) linear multiclass perceptron with hardlimiter activation function")
drawErrors(errors)
title("error / iteration")

weights = matrix(0, 3, 3)
learningRate <- 0.5
iteration <- 100
errors <- c()
for (j in 1:iteration) {    # interation
    size <- nrow(class0)
    numberOfErrors <- 0
    for (i in 1:size) { # number of data
        inputs <- matrix(c(class0[i, ], class1[i, ], class2[i, ]), 3, 3)
        numberOfClasses <- 3
        for (k in 1:numberOfClasses) { # each classes
            input <- inputs[, k]
            outputs <- c(perceptron(input, weights[, 1], featureL, sigmoid),
                perceptron(input, weights[, 2], featureL, sigmoid),
                perceptron(input, weights[, 3], featureL, sigmoid))

            # wrong prediction, adjust all weights
            if (max(outputs) != outputs[k] || min(outputs) == outputs[k]) {
                correction <- featureL(input) * learningRate
                for (m in 1:numberOfClasses) {
                    if (m == k) {   # reinforce
                        weights[, m] <- weights[, m] + correction
                    } else {
                        weights[, m] <- weights[, m] - correction
                    }
                }
                numberOfErrors <- numberOfErrors + 1
            }
        }
    }
    errors <- c(errors, numberOfErrors)
}


plot(NA, NA, xlim=c(-20,20), ylim=c(-20,20), xlab="x", ylab="y")
points(class0, col = colours[1])
points(class1, col = colours[2])
points(class2, col = colours[3])

plotBoundary3(weights)
title("(b) linear multiclass perceptron with sigmoid activation function")

drawErrors(errors)
title("error / iteration")

################################################################################
#   c
################################################################################

class3 = list(
    mean = c(5, 5),
    sigma = matrix(c(1, 0, 0, 1), 2, 2))
class3 = tagClass(mvrnorm(n = 100, class3$mean, class3$sigma), 3)

weights = matrix(0, 6, 4)
learningRate <- 1
iteration <- 500
errors <- c()
for (j in 1:iteration) {    # interation
    size <- nrow(class0)
    numberOfErrors <- 0
    for (i in 1:size) { # number of data
        inputs <- matrix(c(class0[i, ], class1[i, ], class2[i, ], class3[i, ]), 3, 4)
        numberOfClasses <- 4
        for (k in 1:numberOfClasses) { # each classes
            input <- inputs[, k]
            outputs <- c(perceptron(input, weights[, 1], featureQ, hardlimiter),
                perceptron(input, weights[, 2], featureQ, hardlimiter),
                perceptron(input, weights[, 3], featureQ, hardlimiter),
                perceptron(input, weights[, 4], featureQ, hardlimiter))
            correction <- featureQ(input) * learningRate
            for (m in 1:numberOfClasses) {
                if (m == k && outputs[m] == 0) { # reinforce
                    weights[, m] <- weights[, m] + correction
                    numberOfErrors <- numberOfErrors + 1
                }

                if (m != k && outputs[m] == 1) { # punish
                    weights[, m] <- weights[, m] - correction
                    numberOfErrors <- numberOfErrors + 1
                }
            }
        }
    }
    errors <- c(errors, numberOfErrors)
}

plot(NA, NA, xlim=c(-20,20), ylim=c(-20,20), xlab="x", ylab="y")
points(class0, col = colours[1])
points(class1, col = colours[2])
points(class2, col = colours[3])
points(class3, col = colours[4])
drawDecisionBoundaryQ(weights[, 1], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
drawDecisionBoundaryQ(weights[, 2], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
drawDecisionBoundaryQ(weights[, 3], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
drawDecisionBoundaryQ(weights[, 4], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
title("(b) quadratic multiclass perceptron with hardlimiter activation function")

drawErrors(errors)
title("error / iteration")


weights = matrix(0.5, 6, 4)
learningRate <- 0.1
iteration <- 500
errors <- c()
for (j in 1:iteration) {    # interation
    size <- nrow(class0)
    numberOfErrors <- 0
    for (i in 1:size) { # number of data
        inputs <- matrix(c(class0[i, ], class1[i, ], class2[i, ], class3[i, ]), 3, 4)
        numberOfClasses <- 4
        for (k in 1:numberOfClasses) { # each classes
            input <- inputs[, k]
            outputs <- c(
                perceptron(input, weights[, 1], featureQ, sigmoid),
                perceptron(input, weights[, 2], featureQ, sigmoid),
                perceptron(input, weights[, 3], featureQ, sigmoid),
                perceptron(input, weights[, 4], featureQ, sigmoid))

            for (m in 1:numberOfClasses) {
                if (m == k && outputs[m] < 0.9) { # reinforce
                    correction <- featureQ(input) * learningRate * (1 - outputs[m])
                    weights[, m] <- weights[, m] + correction
                    numberOfErrors <- numberOfErrors + 1
                }

                if (m != k && outputs[m] > 0.1) { # punish
                    correction <- featureQ(input) * learningRate * (0 - outputs[m])
                    weights[, m] <- weights[, m] + correction
                    numberOfErrors <- numberOfErrors + 1
                }
            }
        }
    }
    errors <- c(errors, numberOfErrors)
}

plot(NA, NA, xlim=c(-20,20), ylim=c(-20,20), xlab="x", ylab="y")
points(class0, col = colours[1])
points(class1, col = colours[2])
points(class2, col = colours[3])
points(class3, col = colours[4])
drawDecisionBoundaryQ(weights[, 1], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
drawDecisionBoundaryQ(weights[, 2], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
drawDecisionBoundaryQ(weights[, 3], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
drawDecisionBoundaryQ(weights[, 4], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
title("(b) quadratic multiclass perceptron with sigmoid activation function")

drawErrors(errors)
title("error / iteration")
