library("plyr")
library("MASS")
#
# ################################################################################
# #   0
# ################################################################################
#
# library("ElemStatLearn")
# require("class")
# mixture.example$xnew
# x <- mixture.example$x
# g <- mixture.example$y
# xnew <- mixture.example$xnew
# mod15 <- knn(x, xnew, g, k=15, prob=TRUE)
# prob <- attr(mod15, "prob")
# prob <- ifelse(mod15=="1", prob, 1-prob)
# px1 <- mixture.example$px1
# px2 <- mixture.example$px2
# prob15 <- matrix(prob, length(px1), length(px2))
# par(mar=rep(2,4))
# contour(px1, px2, prob15, levels=0.5, labels="", xlab="", ylab="", main=
#         "15-nearest neighbour", axes=FALSE)
# points(x, col=ifelse(g==1, "coral", "cornflowerblue"))
# gd <- expand.grid(x=px1, y=px2)
# points(gd, pch=".", cex=1.2, col=ifelse(prob15>0.5, "coral", "cornflowerblue"))
# box()


# spirals
spiral <- function (radius, theta, sequence, col) {
    r <- radius(sequence)
    t <- theta(sequence)
    x <- r * sin(t)
    y <- r * cos(t)
    return(data.frame(x=x, y=y))
    # points(x, y, col=col)
}
sigmoid <- function (n) 1/(1 + exp(-n))

computeLayer <- function (input, weights) {
    sums <- colSums(input * weights)
    return(sigmoid(sums))
}

computeAll <- function (input, weights) {
    numberOfLayers <- length(weights)
    output <- list()
    output[[1]] <- input
    for (i in 1:numberOfLayers) {
        output[[i + 1]] <- computeLayer(output[[i]], weights[[i]])
    }
    return(output)
}

classify <- function (input, weights) {
    numberOfLayers <- length(weights)
    output <- input
    for (i in 1:numberOfLayers) {
        output <- computeLayer(output, weights[[i]])
    }
    return(output)
}

mse <- function (target, output) {
    diff <- target - output
    return(sum(diff * diff)/2)
}

train <- function (weights, input, target, learningRate = 0.5) {

    outputs <- computeAll(input, weights)
    numberOfLayers <- length(weights)

    # linear programming, tabulate deltas
    deltas <- list()

    # tabulate the output layer first
    output <- outputs[[numberOfLayers + 1]]
    deltas[[numberOfLayers]] <- (target - output) * output * (1 - output)
    # tabulate the rest
    if (numberOfLayers > 1) {
        for (layer in (numberOfLayers - 1):1) {
            output <- outputs[[layer + 1]]
            weight <- weights[[layer + 1]]

            sums <- rowSums(deltas[[layer + 1]] * weights[[layer + 1]])
            deltas[[layer]] <- sums * output * (1 - output)
        }
    }

    for (layer in 1:numberOfLayers) {
        weights[[layer]] <- weights[[layer]] + outputs[[layer]] %o% deltas[[layer]] * learningRate
    }

    return(list(
        weights = weights,
        error = mse(target, outputs[[numberOfLayers + 1]])
        ))
    # return(weights)
}

drawErrors <- function (errors) {
    plot(errors, xlab="iteration", ylab="error", type="s")
    title("error / iteration")
}


################################################################################
#   1
################################################################################


plot(NA, NA, xlim=c(-8,8), ylim=c(-8,8),
    xlab="x", ylab="y",
    axes=FALSE, asp=1)
axis(side=1, at=seq(-8, 8, by=2), cex.axis=0.8)
axis(side=2, at=seq(-8, 8, by=2), cex.axis=0.8)
title("Two Spiral Problem")

class0 <- spiral(function (i) (6.5 * (104 - i) / 104),
    function (i) (pi * i / 16),
    0:96)

class1 <- spiral(function (i) (6.5 * (104 - i) / -104),
    function (i) (pi * i / 16),
    0:96)

# class0 <- data.frame(x=c(-4, -4, 4, 0), y=c(-4, 4, 4, 0))
# class1 <- data.frame(x=c(4), y=c(-4))

# class0 <- data.frame(x=c(-4, -4, 4, 0), y=c(-4, 4, -4, 0))
# class1 <- data.frame(x=c(4, -1), y=c(4, -2))
# class0 <- data.frame(x=c(-4, 4), y=c(4, -4))
# class1 <- data.frame(x=c(4, -4), y=c(4, -4))


points(class0, col="cornflowerblue", cex=2)
points(class1, col="coral", cex=2)


initializeWeights <- function (nodes) {
    weights <- list()
    for (i in 1:(length(nodes) - 1)) {
        rows <- nodes[i]
        cols <- nodes[i + 1]
        size <- rows * cols
        weights[[i]] <- matrix(sigmoid(rexp(size)), rows, cols)
    }
    return(weights)
}


weights <- initializeWeights(c(2, 5, 5, 2))

errors <- c()
# start training
iteration <- 100
# weights[[2]] <- matrix(0, 2, 2)
for (i in 1:iteration) {
    # class 0
    size <- length(class0$x)
    for (j in 1:size) {
        data <- class0[j, ]
        result <- train(weights, c(data$x, data$y), c(1, 0), 0.8)
        weights <- result$weights
        errors <- c(errors, result$error)
    }
    # class 1
    size <- length(class1$x)
    for (j in 1:size) {
        data <- class1[j, ]
        result <- train(weights, c(data$x, data$y), c(0, 1), 0.8)
        weights <- result$weights
        errors <- c(errors, result$error)
    }
    # print(weights)
}


# grid <- expand.grid(x=seq(-8, 8, by=2), y=seq(-8, 8, by=2))
grid <- expand.grid(x=seq(-8, 8, by=0.2), y=seq(-8, 8, by=0.2))
gridResult <- aaply(grid, 1, function (data) {
        result <- classify(c(data$x, data$y), weights)
        return(ifelse(result[1] > result[2], "cornflowerblue", "coral"))
    })
points(grid, pch=".", cex=2, col=gridResult)

# classify(c(-4, -4, 1), weights)

drawErrors(errors)

# ################################################################################
# #   a
# ################################################################################
# colours <- terrain.colors(5)
#
# drawDataPoints <- function (inputs) {
#     classA <- inputs[inputs$class == 1, ]
#     classB <- inputs[inputs$class == 0, ]
#
#     points(classA$x, classA$y, col = colours[1], lwd = 3)
#     points(classB$x, classB$y, col = colours[2], lwd = 3)
# }
#
#
# drawDecisionBoundaryL <- function (weights, i = 1) {
#     a <- weights[1]
#     b <- weights[2]
#     c <- weights[3]
#     plot(NA, NA, xlim=c(-2,2), ylim=c(-2,2), xlab="x", ylab="y", col = colours[i])
#     abline(-(c/b), -(a/b))
# }
#
# drawDecisionBoundaryQ <- function (weights, xlim = c(-2,2), ylim = c(-2,2), length = 10) {
#     f <- Vectorize(function(x,y) sum(featureQ(c(x, y)) * weights))
#     x <- seq(xlim[1], xlim[2], length = length)
#     y <- seq(ylim[1], ylim[2], length = length)
#     z <- outer(x,y,f)
#     contour(
#         x = x, y = x, z = z,
#         levels = 0, las = 1, drawlabels = FALSE, lwd = 3,
#         xlim=xlim, ylim=ylim, xlab="x", ylab="y", add = TRUE
#     )
# }
# drawErrors <- function (errors) {
#     plot(errors, xlab="iteration", ylab="error", type="s")
# }
#
#
# featureL <- function (input) {
#     x <- unlist(input[1])
#     y <- unlist(input[2])
#     return(unlist(c(x, y, 1)))
# }
#
# featureQ <- function (input) {
#     x <- unlist(input[1])
#     y <- unlist(input[2])
#     return(unlist(c(x * x, x * y, y * y, x, y, 1)))
# }
#
# # activation function
# hardlimiter <- function (n) {
#     if (n > 0) {
#         return(1)
#     } else {
#         return(0)
#     }
# }
#
# sigmoid <- function (n) 1/(1 + exp(-n))
#
# id <- function (n) n
#
# perceptron <- function (input, weights, feature, activate) {
#     return(activate(sum(feature(input) * weights)))
# }
#
# classifyA <- function (feature, activation, learningRate, iteration, inputs, weights) {
#     errors <- c()
#     for (j in 1:iteration) {
#         numberOfErrors <- 0
#         for (i in 1:nrow(inputs)) {
#             input <- inputs[i, ]
#             output <- perceptron(input, weights, feature, activation)
#             error <- input$class - output
#             correction <- error * learningRate
#             weights <- weights + feature(input) * correction
#             if (error != 0)
#                 numberOfErrors <- numberOfErrors + 1
#         }
#         errors <- c(errors, numberOfErrors)
#     }
#     return(list(
#         weights = weights,
#         errors = errors
#         ))
# }
#
#
# inputs <- data.frame(
#     x = c(-1, -1, 1, 1),
#     y = c(-1, 1, -1, 1),
#     class = c(1, 0, 0, 1))
#
#
# result <- classifyA(featureQ, hardlimiter, 0.1, 100, inputs, rep(0, 6))
# plot(NA, NA, xlim=c(-2,2), ylim=c(-2,2), xlab="x", ylab="y")
# drawDecisionBoundaryQ(result$weights)
# drawDataPoints(inputs)
# title("(a) dividing XOR with quadratic perceptrons")
#
# drawErrors(result$errors)
# title("error / iteration")
#
# ################################################################################
# #   b
# ################################################################################
#
# plotBoundary3 <- function(weights) {
#     # vectors
#     l01 <- weights[, 1] - weights[, 2]
#     l12 <- weights[, 2] - weights[, 3]
#     l20 <- weights[, 3] - weights[, 1]
#
#     # cross point
#     x <- ((l01[2]*l12[3])/l12[2] - l01[3]) / (l01[1] - (l12[1]*l01[2])/l12[2])
#     y <- ((l01[1]*l12[3])/l12[1] - l01[3]) / (l01[2] - (l12[2]*l01[1])/l12[1])
#
#     slope01 = l01[1]/(-l01[2])
#     slope12 = l12[1]/(-l12[2])
#     slope20 = l20[1]/(-l20[2])
#     segments(x, y, x - 100, y - slope01 * 100)
#     segments(x, y, x + 100, y + slope12 * 100)
#     segments(x, y, x - 100, y - slope20 * 100)
# }
#
#
# tagClass <- function (inputs, class) {
#     return(data.frame(x = inputs[, 1], y = inputs[, 2], class = class))
# }
#
# class0 = list(
#     mean = c(0, 0),
#     sigma = matrix(c(1, 0, 0, 1), 2, 2))
#
# class1 = list(
#     mean = c(10, 0),
#     sigma = matrix(c(1, 0, 0, 4), 2, 2))
#
# class2 = list(
#     mean = c(5, 10),
#     sigma = matrix(c(4, 0, 0, 1), 2, 2))
#
# class0raw = mvrnorm(n = 100, class0$mean, class0$sigma)
# class1raw = mvrnorm(n = 100, class1$mean, class1$sigma)
# class2raw = mvrnorm(n = 100, class2$mean, class2$sigma)
#
#
# class0 = tagClass(class0raw, 0)
# class1 = tagClass(class1raw, 1)
# class2 = tagClass(class2raw, 2)
#
#
# weights = matrix(0, 3, 3)
# learningRate <- 0.5
# iteration <- 100
# errors <- c()
# for (j in 1:iteration) {    # interation
#     size <- nrow(class0)
#     numberOfErrors <- 0
#     for (i in 1:size) { # number of data
#         inputs <- matrix(c(class0[i, ], class1[i, ], class2[i, ]), 3, 3)
#         numberOfClasses <- 3
#         for (k in 1:numberOfClasses) { # each classes
#             input <- inputs[, k]
#             outputs <- c(perceptron(input, weights[, 1], featureL, hardlimiter),
#                 perceptron(input, weights[, 2], featureL, hardlimiter),
#                 perceptron(input, weights[, 3], featureL, hardlimiter))
#
#             correction <- featureL(input) * learningRate
#             for (m in 1:numberOfClasses) {
#                 if (m == k && outputs[m] == 0) { # reinforce
#                     weights[, m] <- weights[, m] + correction
#                     numberOfErrors <- numberOfErrors + 1
#                 }
#
#                 if (m != k && outputs[m] == 1) { # punish
#                     weights[, m] <- weights[, m] - correction
#                     numberOfErrors <- numberOfErrors + 1
#                 }
#             }
#         }
#     }
#     errors <- c(errors, numberOfErrors)
# }
# plot(NA, NA, xlim=c(-20,20), ylim=c(-20,20), xlab="x", ylab="y")
# points(class0, col = colours[1])
# points(class1, col = colours[2])
# points(class2, col = colours[3])
#
# plotBoundary3(weights)
# title("(b) linear multiclass perceptron with hardlimiter activation function")
# drawErrors(errors)
# title("error / iteration")
#
# weights = matrix(0, 3, 3)
# learningRate <- 0.5
# iteration <- 100
# errors <- c()
# for (j in 1:iteration) {    # interation
#     size <- nrow(class0)
#     numberOfErrors <- 0
#     for (i in 1:size) { # number of data
#         inputs <- matrix(c(class0[i, ], class1[i, ], class2[i, ]), 3, 3)
#         numberOfClasses <- 3
#         for (k in 1:numberOfClasses) { # each classes
#             input <- inputs[, k]
#             outputs <- c(perceptron(input, weights[, 1], featureL, sigmoid),
#                 perceptron(input, weights[, 2], featureL, sigmoid),
#                 perceptron(input, weights[, 3], featureL, sigmoid))
#
#             # wrong prediction, adjust all weights
#             if (max(outputs) != outputs[k] || min(outputs) == outputs[k]) {
#                 correction <- featureL(input) * learningRate
#                 for (m in 1:numberOfClasses) {
#                     if (m == k) {   # reinforce
#                         weights[, m] <- weights[, m] + correction
#                     } else {
#                         weights[, m] <- weights[, m] - correction
#                     }
#                 }
#                 numberOfErrors <- numberOfErrors + 1
#             }
#         }
#     }
#     errors <- c(errors, numberOfErrors)
# }
#
#
# plot(NA, NA, xlim=c(-20,20), ylim=c(-20,20), xlab="x", ylab="y")
# points(class0, col = colours[1])
# points(class1, col = colours[2])
# points(class2, col = colours[3])
#
# plotBoundary3(weights)
# title("(b) linear multiclass perceptron with sigmoid activation function")
#
# drawErrors(errors)
# title("error / iteration")
#
# ################################################################################
# #   c
# ################################################################################
#
# class3 = list(
#     mean = c(5, 5),
#     sigma = matrix(c(1, 0, 0, 1), 2, 2))
# class3 = tagClass(mvrnorm(n = 100, class3$mean, class3$sigma), 3)
#
# weights = matrix(0, 6, 4)
# learningRate <- 1
# iteration <- 500
# errors <- c()
# for (j in 1:iteration) {    # interation
#     size <- nrow(class0)
#     numberOfErrors <- 0
#     for (i in 1:size) { # number of data
#         inputs <- matrix(c(class0[i, ], class1[i, ], class2[i, ], class3[i, ]), 3, 4)
#         numberOfClasses <- 4
#         for (k in 1:numberOfClasses) { # each classes
#             input <- inputs[, k]
#             outputs <- c(perceptron(input, weights[, 1], featureQ, hardlimiter),
#                 perceptron(input, weights[, 2], featureQ, hardlimiter),
#                 perceptron(input, weights[, 3], featureQ, hardlimiter),
#                 perceptron(input, weights[, 4], featureQ, hardlimiter))
#             correction <- featureQ(input) * learningRate
#             for (m in 1:numberOfClasses) {
#                 if (m == k && outputs[m] == 0) { # reinforce
#                     weights[, m] <- weights[, m] + correction
#                     numberOfErrors <- numberOfErrors + 1
#                 }
#
#                 if (m != k && outputs[m] == 1) { # punish
#                     weights[, m] <- weights[, m] - correction
#                     numberOfErrors <- numberOfErrors + 1
#                 }
#             }
#         }
#     }
#     errors <- c(errors, numberOfErrors)
# }
#
# plot(NA, NA, xlim=c(-20,20), ylim=c(-20,20), xlab="x", ylab="y")
# points(class0, col = colours[1])
# points(class1, col = colours[2])
# points(class2, col = colours[3])
# points(class3, col = colours[4])
# drawDecisionBoundaryQ(weights[, 1], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
# drawDecisionBoundaryQ(weights[, 2], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
# drawDecisionBoundaryQ(weights[, 3], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
# drawDecisionBoundaryQ(weights[, 4], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
# title("(b) quadratic multiclass perceptron with hardlimiter activation function")
#
# drawErrors(errors)
# title("error / iteration")
#
#
# weights = matrix(0.5, 6, 4)
# learningRate <- 0.1
# iteration <- 500
# errors <- c()
# for (j in 1:iteration) {    # interation
#     size <- nrow(class0)
#     numberOfErrors <- 0
#     for (i in 1:size) { # number of data
#         inputs <- matrix(c(class0[i, ], class1[i, ], class2[i, ], class3[i, ]), 3, 4)
#         numberOfClasses <- 4
#         for (k in 1:numberOfClasses) { # each classes
#             input <- inputs[, k]
#             outputs <- c(
#                 perceptron(input, weights[, 1], featureQ, sigmoid),
#                 perceptron(input, weights[, 2], featureQ, sigmoid),
#                 perceptron(input, weights[, 3], featureQ, sigmoid),
#                 perceptron(input, weights[, 4], featureQ, sigmoid))
#
#             for (m in 1:numberOfClasses) {
#                 if (m == k && outputs[m] < 0.9) { # reinforce
#                     correction <- featureQ(input) * learningRate * (1 - outputs[m])
#                     weights[, m] <- weights[, m] + correction
#                     numberOfErrors <- numberOfErrors + 1
#                 }
#
#                 if (m != k && outputs[m] > 0.1) { # punish
#                     correction <- featureQ(input) * learningRate * (0 - outputs[m])
#                     weights[, m] <- weights[, m] + correction
#                     numberOfErrors <- numberOfErrors + 1
#                 }
#             }
#         }
#     }
#     errors <- c(errors, numberOfErrors)
# }
#
# plot(NA, NA, xlim=c(-20,20), ylim=c(-20,20), xlab="x", ylab="y")
# points(class0, col = colours[1])
# points(class1, col = colours[2])
# points(class2, col = colours[3])
# points(class3, col = colours[4])
# drawDecisionBoundaryQ(weights[, 1], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
# drawDecisionBoundaryQ(weights[, 2], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
# drawDecisionBoundaryQ(weights[, 3], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
# drawDecisionBoundaryQ(weights[, 4], xlim = c(-20, 20), ylim = c(-20, 20), length = 100)
# title("(b) quadratic multiclass perceptron with sigmoid activation function")
#
# drawErrors(errors)
# title("error / iteration")
