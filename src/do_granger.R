# Perform granger casaulity test, up to 2 lags, for two series
lex <- read.csv("./granger_text_input.txt", header=F, as.is=T)
econ <- read.csv("./granger_econ_input.txt", header=F, as.is=T)

#  1 - lag
y_t1 <- lex$V1[1:(length(lex$V1)-1)]
y_t <- lex$V1[2:length(lex$V1)]
x_t1 <- econ$V1[1:(length(lex$V1)-1)]

lr <- lm(y_t ~ y_t1 + x_t1)
summary(lr)


y_t1 <- lex$V1[2:(length(lex$V1)-1)]
y_t <- lex$V1[3:length(lex$V1)]
x_t1 <- econ$V1[2:(length(lex$V1)-1)]
y_t2 <- lex$V1[1:(length(lex$V1)-2)]
x_t2 <- econ$V1[1:(length(econ$V1)-2)]


lr_lag_2_full <- lm(y_t ~ y_t1 + y_t2 + x_t1 + x_t2)
summary(lr_lag_2_full)