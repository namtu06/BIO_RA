# Simple R test

# Print a message
print("Hello, R is working!")

# Basic arithmetic
a <- 5
b <- 3
sum <- a + b
print(paste("The sum of", a, "and", b, "is", sum))

# Simple plot
x <- 1:10
y <- x^2

png("test_plot.png")
plot(x, y, type = "b", col = "blue", main = "Test Plot", xlab = "x", ylab = "y")
dev.off()