library("animation")
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

learn <- function (weights, input, target, learningRate = 0.5) {

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

plotErrors <- function (errors) {
    if (length(errors) > 0) {
        plot(errors, xlab="iteration", ylab="error", type="h")
        title("error / iteration")
    }
}


reconstruct <- function (weights) {
    structure <- c()
    for (i in 1:length(weights)) {
        structure <- c(structure, nrow(weights[[i]]))
    }
    structure <- c(structure, ncol(weights[[i]]))
    return(structure)
}

identityWeights <- function (nodes) {
    weights <- list()
    for (i in 1:(length(nodes) - 1)) {
        rows <- nodes[i]
        cols <- nodes[i + 1]
        weights[[i]] <- matrix(0, rows, cols)
    }
    return(weights)
}

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

accumulateWeights <- function (acc, x) {
    for (i in 1:length(acc)) {
        acc[[i]] <- acc[[i]] + x[[i]]
    }
    return(acc)
}

train <- function (weights, callback, iteration = 100000, until = 0.1, every = 1000) {

    errors <- c()
    learningRate <- 1

    for (i in 1:iteration) {
        accumulatedError <- 0

        # class 0
        size <- length(class0$x)
        for (j in 1:size) {
            data <- class0[j, ]
            result <- learn(weights, c(data$x, data$y, 1), c(1, 0), learningRate)
            weights <- result$weights
            accumulatedError <- accumulatedError + result$error
        }
        # class 1
        size <- length(class1$x)
        for (j in 1:size) {
            data <- class1[j, ]
            result <- learn(weights, c(data$x, data$y, 1), c(0, 1), learningRate)
            weights <- result$weights
            accumulatedError <- accumulatedError + result$error
        }
        errors <- c(errors, accumulatedError)

        if (accumulatedError < until) {
            print(i)
            break
        }

        if (i %% every == 0) {
            callback(i, weights, errors)
        }
    }
    return(list(
        weights = weights,
        errors = errors
        ))
}

drawGrid <- function (weights) {
    grid <- expand.grid(x=seq(-8, 8, by=0.2), y=seq(-8, 8, by=0.2))
    gridResult <- aaply(grid, 1, function (data) {
            result <- classify(c(data$x, data$y, 1), weights)
            return(ifelse(result[1] > result[2], "cornflowerblue", "coral"))
        })
    points(grid, pch=".", cex=3, col=gridResult)
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

# class0 <- spiral(function (i) (6.5 * (104 - i) / 104),
#     function (i) (pi * i / 8),
#     40:52)
#
# class1 <- spiral(function (i) (6.5 * (104 - i) / -104),
#     function (i) (pi * i / 8),
#     40:52)
# class0 <- spiral(function (i) (6.5 * (104 - i) / 104),
#     function (i) (pi * i / 8),
#     0:10)
#
# class1 <- spiral(function (i) (6.5 * (104 - i) / -104),
#     function (i) (pi * i / 8),
#     0:10)
class0 <- data.frame(x=c(-5), y=c(-2))
class1 <- data.frame(x=c(1), y=c(-1))

# class0 <- data.frame(x=c(-4, -4, 4, 0), y=c(-4, 4, 4, 0))
# class1 <- data.frame(x=c(4), y=c(-4))

# class0 <- data.frame(x=c(-4, -4, 4, 0), y=c(-4, 4, -4, 0))
# class1 <- data.frame(x=c(4, -2, 0, 2, 4), y=c(4, -2, -2, -2, -2))
# class0 <- data.frame(x=c(-4, 4), y=c(4, -4))
# class1 <- data.frame(x=c(4, -4), y=c(4, -4))


points(class0, col="cornflowerblue", cex=2)
points(class1, col="coral", cex=2)



layers <- c(3, 5, 2)
# weights <- initializeWeights(layers)

weights <- list()
weights[[1]] <- matrix(
    c(1.716595448658756, 1.1764761912424921, 1.557118196208473, 1.7041037817911824, 1.6406029506187856,
     0.6146408255673677, 1.2674559523877986, 0.8036832609525906, 0.4969173598004548, 0.7056984961660893,
     0.978526932134667, 0.3158700142016504, 0.7265454504592824, 0.6660428783020633, 0.6623111268153622),
     3, 5, byrow = TRUE)
weights[[2]] <- matrix(
    c(-1.3136394181316513, 1.0616230793282784,
     -0.1289445413211639, 1.0017360756800076,
     -0.9759845197492095, 1.0231283450037518,
     -1.276113700799778, 1.1112948075375635,
     -1.325443401614583, 1.1986825244676422), 5, 2, byrow = TRUE)

# print(computeLayer(c(1, -1, 1), weights[[1]]))

result <- train(weights, function (i, weights, errors) {
    print(errors[i])
    }, 10000)
print(result$weights)
drawGrid(result$weights)
plotErrors(result$errors)
