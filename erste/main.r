library("plot3D")

################################################################################
#   1
################################################################################

x <- seq(0, 10, by = 0.001)
plot(x, dnorm(x, 5, 1), type = 'l')
title("1-d Gaussian function")

################################################################################
#   2
################################################################################

x <- seq(0, 10, 0.1)
y <- x
f <- function(x, y) {
    mean <- 5
    distanceToMean <- sqrt( (x - mean)^2 + (y - mean)^2 )
    return(dnorm(distanceToMean, 0, 1))
}
z <- outer(x, y, f)
persp(x, y, z, theta = 30, phi = 30, col = "lightblue", ticktype = "detailed")
title("2-d Gaussian function")

################################################################################
#   3
################################################################################

x <- seq(1, 10, 0.001)
hist(rnorm(x, 5, 1), breaks = 100)
# title("1-d Gaussian random data histogram")

################################################################################
#   4
################################################################################

acc <- matrix(0, 40, 40)
x <- rnorm(10000, 5, 1)
y <- rnorm(10000, 5, 1)

plot(x, y)
title("2-d Gaussian random data")

for (i in 1:10000) {
    xi <- ceiling(x[i] * 4)
    yi <- ceiling(y[i] * 4)
    acc[xi, yi] <- acc[xi, yi] + 1
}
hist3D(z = acc, ticktype = "detailed",)
title("2-d Gaussian random data histogram")

################################################################################
#   5
################################################################################

tcol <- terrain.colors(12)
contour(acc, col = tcol, lty = "solid")
title("2-d Gaussian random data contour")
