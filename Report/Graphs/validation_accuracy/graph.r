word_count = read.csv("history_word_count.csv", header=TRUE)
ngrams = read.csv("history_ngrams.csv", header=TRUE)
word2vec = read.csv("history_word2vec.csv", header=TRUE)
lstm = read.csv("history_lstm.csv", header=TRUE)
combined = read.csv("history_combined.csv", header=TRUE)

jpeg("plot.jpg", width = 500, height = 500)

plot(c(1:10), word_count[, 2], type="l", col="red", ylim=c(0.49, 0.73), xlab="epoch", ylab="validation_accuracy")
lines(c(1:10), ngrams[, 2], col="green")
lines(c(1:10), word2vec[, 2], col="blue")
lines(c(1:10), lstm[, 2], col="yellow")
lines(c(1:10), combined[, 2], col="purple")

legend(6, 0.58, legend=c("combined", "word_count", "ngrams", "ngrams_lstm", "word2vec"), col=c("purple", "red", "blue", "yellow", "green"), lty=1, cex=1.5)

dev.off()
