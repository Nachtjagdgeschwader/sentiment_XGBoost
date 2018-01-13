library(data.table)
library(dplyr)
library(caret)
library(RTextTools)
library(xgboost)
library(ROCR)
library(tm)
# Loading manually coded dataset for supervised learning: 10-15.
# 1 stays for "positive sentiment", 2 is for "negative".
data <-
  read.csv2(
    "C:/path/Manually_coded_data.csv",
    header = T,
    sep = ";"
  )

# Text cleaning: 18-35.
data$Text <- as.character(data$Text)
data$Text <- gsub('\n',' ',data$Text)
data$Text <- gsub(" ?(f|ht)(tp)(s?)(://)(.*)[.|/](.*)", " ",data$Text)
data$Text <- gsub("[^[:alnum:]]", " ",data$Text)
data$Text <- tolower(data$Text)
data$Text <- gsub('\\b\\w{30,}\\b','',data$Text)
data$Text <- gsub("\\s+|^\\s+|\\s+$", " ",data$Text)
# Loading stopwords for Ukrainian, Russian and English language: 26-27.
stopwords <- read.csv2("C:/path/Stopwords.csv", header = F, sep = ";")$V1
stopwords1 <- as.character(stopwords)
# Loading keywords for Ukrainian, Russian and English language: 29-30.
keywords <- read.csv2("C:/path/Ключові слова.csv", header = F, sep = ";")$V1
keywords1 <- as.character(keywords)
key <- paste0("\\b(", paste0(keywords1, collapse="|"), ")\\b")
stop <- paste0("\\b(", paste0(stopwords1, collapse="|"), ")\\b")
data$Text <- gsub(stop, "",data$Text)
data$Text <- gsub(key, "",data$Text)
data$Text <- gsub("[[:digit:]]+", " ",data$Text)
# Defining coded and virgin dataframes: 37-55.
data$doc.id <- 1:length(data$Text)
data$bad.sent <- 0
data$bad.sent[data$Sentiment == 2] <- 1
trainIdx <- createDataPartition(data$bad.sent,
                                p = .5,
                                list = FALSE,
                                times = 1)
train <- data[trainIdx, ]
test <- data[-trainIdx, ]
sparsity <- .998
corp_g <- Corpus(DataframeSource(data.frame(c(train$Text[train$bad.sent == 0]))))
corp_g <- tm_map(corp_g, stripWhitespace)
corp_g <- tm_map(corp_g, stemDocument, language = "russian")
good.dtm <- DocumentTermMatrix(corp_g)
good.dtm <- removeSparseTerms(good.dtm,sparsity)
good.dtm.df <- as.data.frame(as.matrix(good.dtm), 
                             row.names = train$doc.id[train$bad.sent == 0])
corp_b <- Corpus(DataframeSource(data.frame(c(train$Text[train$bad.sent == 1]))))
corp_b <- tm_map(corp_b, stripWhitespace)
# corp_b <- tm_map(corp_b, stemDocument, language = "russian")
# Stemming coded texts. Optional. Sometimes unstemmed texts perform better.
bad.dtm <- DocumentTermMatrix(corp_b)
bad.dtm <- removeSparseTerms(bad.dtm,sparsity)
bad.dtm.df <- as.data.frame(as.matrix(bad.dtm), 
                            row.names = train$doc.id[train$bad.sent == 1])
train.dtm.df <- bind_rows(bad.dtm.df, good.dtm.df)
train.dtm.df$doc.id <- c(train$doc.id[train$bad.sent == 0], train$doc.id[train$bad.sent == 1])
train.dtm.df <- arrange(train.dtm.df, doc.id)
train.dtm.df$bad.sent <- train$bad.sent
train.dtm.df$doc.id=NULL
corp_t <- Corpus(DataframeSource(data.frame(c(test$Text))))
corp_t <- tm_map(corp_t, stripWhitespace)
# corp_t <- tm_map(corp_t, stemDocument, language = "russian")
test.dtm <- DocumentTermMatrix(corp_t)
test.dtm <- removeSparseTerms(test.dtm,sparsity)
test.dtm.df <- data.table(as.matrix(test.dtm))
test.dtm.df$doc.id <- test$doc.id
test.dtm.df$bad.sent <- test$bad.sent
test.dtm.df <- head(bind_rows(test.dtm.df, train.dtm.df[1, ]), -1)
test.dtm.df <- test.dtm.df %>% 
  select(one_of(colnames(train.dtm.df)))
test.dtm.df[is.na(test.dtm.df)] <- 0
test.dtm.df$doc.id=NULL
# "Teaching" the model.
baseline.acc <- sum(test$bad.sent == "1") / nrow(test)
XGB.train <- as.matrix(select(train.dtm.df, -bad.sent),
                       dimnames = dimnames(train.dtm.df))
XGB.test <- as.matrix(select(test.dtm.df, -bad.sent),
                      dimnames=dimnames(test.dtm.df))
XGB.model <- xgboost(data = XGB.train, 
                     label = train.dtm.df$bad.sent,
                     nrounds = 400, 
                     objective = "binary:logistic")
XGB.predict <- predict(XGB.model, XGB.test)
XGB.results <- data.frame(bad.sent = test$bad.sent,
                          pred = XGB.predict)
# Visually assess prediction threshold.
ROCR.pred <- prediction(XGB.results$pred, XGB.results$bad.sent)
ROCR.perf <- performance(ROCR.pred, 'tnr','fnr') 
plot(ROCR.perf, colorize = TRUE)
# Applying the threshold and assessing overall model accuracy.
XGB.table <- table(true = XGB.results$bad.sent, 
                   pred = as.integer(XGB.results$pred >= 0.97))
XGB.table
XGB.acc <- sum(diag(XGB.table)) / nrow(test)
# Assessing individual words importance for a text being classifyied
# as positive or negative.
names <- as.character(colnames(test.dtm.df))
names <- colnames(test.dtm.df)
importance.matrix <- xgb.importance(names, model = XGB.model)
xgb.plot.importance(importance.matrix[1:20, ])