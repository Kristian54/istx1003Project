# This scode was written to scrape data for the activity published in JSDSE.
# ARTICLE TITLE:  Building your Multiple Linear Regression Model with LEGO Bricks
# You can run the application by clicking the 'Run' button above.
# Code last ran on May 21, 2021

# Install the following packages
library(rvest)
library(stringr)
library(dplyr)
library(reshape2)

# Extracts the set number and name of all LEGO sets sold in years 2018, 2019, and 2020
# If you want different years, change the numbers to different years
year1 <- 2018
year2 <- 2019
year3 <- 2020

# Finds the total number of web pages to extract the Item Number from year1
urlcount <- paste("https://brickset.com/sets/year-", year1, "/category-Normal/page-1", sep = "")
webpage <- read_html(urlcount)
countpages1 <- ceiling(as.numeric(strsplit(html_text(html_nodes(webpage, ".results")), " ")[[1]][5]) / 25)
           
# Goes through all web pages found in countpages1 and saves the Item Number and Name of all Lego Sets in year1
setnum1 <- NULL
for (i in 1:countpages1) {
  url <- paste("https://brickset.com/sets/year-", year1, "/category-Normal/page-", i, sep = "")
  webpage <- read_html(url)
  setnum1 <- c(setnum1, html_text(html_nodes(webpage, "h1 a")))
}

# Finds the total number of web pages to extract the Item Number from year2
urlcount <- paste("https://brickset.com/sets/year-", year2, "/category-Normal/page-1", sep = "")
webpage <- read_html(urlcount)
countpages2 <- ceiling(as.numeric(strsplit(html_text(html_nodes(webpage, ".results")), " ")[[1]][5]) / 25)

# Goes through all web pages found in countpages2 and saves the Item Number and Name of all Lego Sets in year2
setnum2 <- NULL
for (i in 1:countpages2) {
  url <- paste("https://brickset.com/sets/year-", year2, "/category-Normal/page-", i, sep = "")
  webpage <- read_html(url)
  setnum2 <- c(setnum2, html_text(html_nodes(webpage, "h1 a")))
}

# Finds the total number of web pages to extract the Item Number from year3
urlcount <- paste("https://brickset.com/sets/year-", year3, "/category-Normal/page-1", sep = "")
webpage <- read_html(urlcount)
countpages3 <- ceiling(as.numeric(strsplit(html_text(html_nodes(webpage, ".results")), " ")[[1]][5]) / 25)

# Goes through all web pages found in countpages3 and saves the Item Number and Name of all Lego Sets in year3
setnum3 <- NULL
for (i in 1:countpages3) {
  url <- paste("https://brickset.com/sets/year-", year3, "/category-Normal/page-", i, sep = "")
  webpage <- read_html(url)
  setnum3 <- c(setnum3, html_text(html_nodes(webpage, "h1 a")))
}

# Coverts and combines the scraped data to a matrix and adds the variable Year
setnumOut3 <- data.frame(matrix(unlist(strsplit(setnum1, "\\:\\ \\ ")), ncol = 2, byrow = T), "year" = year1)
setnumOut2 <- data.frame(matrix(unlist(strsplit(setnum2, "\\:\\ \\ ")), ncol = 2, byrow = T), "year" = year2)
setnumOut1 <- data.frame(matrix(unlist(strsplit(setnum3, "\\:\\ \\ ")), ncol = 2, byrow = T), "year" = year3)
setnumOut <- rbind(setnumOut1, setnumOut2, setnumOut3)

# Removes Collectable Minifigure Mystery sets from the data
dup_out <- (1 - duplicated(setnumOut[, 1])) * seq(1, length(setnumOut[, 1]), by = 1)
setnumOut <- setnumOut[dup_out, ]
n_l <- dim(setnumOut)[1]


# Extracts Amazon prices as posted on Brickset.com/buy/amazon only available for the most recent years
# These Amazon prices update periodically
webamazon <- read_html("https://brickset.com/buy/amazon")
amazonout <- matrix(html_text(html_nodes(webamazon, ".f td , .c td")), ncol = 12, byrow = TRUE)
amazonfinal <- data.frame(colsplit(amazonout[, 1], " ", c("SetNum", "setName2")), amazonout[, c(2, 3)])
names(amazonfinal) <- c("SetNum", "setName2", "price2", "Amazon_Price")
names(setnumOut) <- c("SetNum", "Set_Name", "Year")
amazondataout <- merge(amazonfinal, setnumOut, by = "SetNum", all = TRUE)


# Scrapes number of pages in the LEGO insturctions from lego.brickinstructions.com,
# scrapes number of Unique Pieces from brickset.com/inventories,
# and organizes the data into a matrix.
countfunct <- function(SetNum) {
  countpages <- NA
  unique_pieces <- NA
  tryCatch(
    {
      countpages <- length(html_nodes(read_html(paste("http://lego.brickinstructions.com/en/lego_instructions/set/", SetNum, "/rb", sep = "")), ".image"))
    },
    error = function(e) {
      cat("ERROR :", conditionMessage(e), SetNum, "\n")
    }
  )
  tryCatch(
    {
      unique_pieces <- html_text(html_nodes(read_html(paste("https://brickset.com/inventories/", SetNum, "-1", sep = "")), ".settools+ .settools tfoot td:nth-child(2)"))
    },
    error = function(e) {
      cat("ERROR :", conditionMessage(e), SetNum, "\n")
    }
  )

  ifelse(countpages == 0, countpages <- NA, countpages <- countpages)
  ifelse(unique_pieces == 0, unique_pieces <- NA, unique_pieces <- unique_pieces)
  return(data.frame(SetNum, countpages, unique_pieces))
}

mini_pages <- apply(matrix(setnumOut[, 1], ncol = 1), 1, countfunct) #This line will take a long time to run.
mini_out <- Reduce(function(d1, d2) merge(d1, d2, all = TRUE), mini_pages)


# Scrapes multiple variables from brickset.com/sets
brickset_scrape_function <- function(Item_Number) {
  url3 <- paste("https://brickset.com/sets/", Item_Number, sep = "")
  webpage3 <- read_html(url3)
  out4 <- html_text(html_nodes(webpage3, "dd, dt"))
  Theme <- ifelse(sum(out4 == "Theme") > 0, out4[c(FALSE, out4 == "Theme")], "NA")
  Pieces <- ifelse(sum(out4 == "Pieces") > 0, out4[c(FALSE, out4 == "Pieces")], "NA")
  Price <- ifelse(sum(out4 == "RRP") > 0, out4[c(FALSE, out4 == "RRP")], "NA")
  Ages <- ifelse(sum(out4 == "Age range") > 0, out4[c(FALSE, out4 == "Age range")], "NA")
  Minifigures <- ifelse(sum(out4 == "Minifigs") > 0, out4[c(FALSE, out4 == "Minifigs")], "NA")
  Packaging <- ifelse(sum(out4 == "Packaging") > 0, out4[c(FALSE, out4 == "Packaging")], "NA")
  Weight <- ifelse(sum(out4 == "Weight") > 0, out4[c(FALSE, out4 == "Weight")], "NA")
  Availability <- ifelse(sum(out4 == "Availability") > 0, out4[c(FALSE, out4 == "Availability")], "NA")

  return(data.frame(Item_Number, Theme, Pieces, Price, Ages, Minifigures, Packaging, Weight, Availability))
}

brickset_data <- apply(matrix(setnumOut[, 1], ncol = 1), 1, brickset_scrape_function) #This line will take a long time to run.


# Formats the Brickset data
Data_Brickset <- Reduce(function(d1, d2) merge(d1, d2, all = TRUE), brickset_data)
Data_Brickset$Price <- colsplit(Data_Brickset[, 7], "/ ", c("Euro", "Price", "Other"))[, 2]
Data_Brickset$Minifigures <- colsplit(Data_Brickset[, 6], " ", c("Minifigures", "Unique Minifigures"))[, 1]


# Organizes, merges, formats, and saves all data in a .csv file
Merge1 <- merge(amazondataout[, c(1, 5, 4, 6)], mini_out, by = "SetNum", all = TRUE)
names(Merge1) <- c("Item_Number", "Set_Name", "Amazon_Price", "Year", "Pages", "unique_pieces")
FinalData <- merge(Merge1, Data_Brickset, by = "Item_Number", all = TRUE)
FinalData$Size <- c(FinalData$Theme == "Duplo") * 1
FinalData$Size <- recode(FinalData$Size, "0" = "Small", "1" = "Large")
FinalData$Ages <- paste("Ages_", FinalData$Ages, sep = "")
write.csv(FinalData, file = "lego.population.csv")