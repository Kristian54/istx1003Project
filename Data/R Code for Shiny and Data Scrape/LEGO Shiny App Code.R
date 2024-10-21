# This Shiny web application was created for the activity published in JSDSE.
# ARTICLE TITLE:  Building your Multiple Linear Regression Model with LEGO Bricks
# You can run the application by clicking the 'Run App' button above.
# Code last ran on May 25, 2021

# Install the following packages
library(shiny)
library(ggplot2)
library(dplyr)

# Import sample dataset for LEGO City, Friends, and DUPLO (CFD)
lego.CFD <- read.csv("lego.sample.csv")

# Replace blanks with NAs
lego.CFD$blank <- NA

# Clean the price variables
lego.CFD$Price <- as.numeric(gsub("\\$", "", lego.CFD$Price))
lego.CFD$Amazon_Price <- as.numeric(gsub("\\$", "", lego.CFD$Amazon_Price))

# Create indicator variables for Size and Theme
lego.CFD$Size_2 <- recode(lego.CFD$Size, Small = "0", Large = "1")
lego.CFD$Theme_2 <- recode(lego.CFD$Theme, Friends = "0", City = "1", .default = "")

# Attach the dataset
attach(lego.CFD)


# UI
ui <- fluidPage(
  sidebarLayout(

    # Input(s)
    sidebarPanel(

      # Select data frame
      selectInput(
        inputId = "dataFrame",
        label = "Select a Dataset:",
        choices = c("City and Friends" = "cf", "City, Friends, and Duplo" = "cfd")
      ),

      # Select variable for y-axis
      selectInput(
        inputId = "y",
        label = "Y Variable:",
        choices = c(
          " " = "blank", "Amazon Price" = "Amazon_Price", "Number of Pieces" = "Pieces",
          "Number of Minifigures" = "Minifigures", "Number of Pages" = "Pages",
          "Number of Unique Pieces" = "Unique_Pieces"
        ),
        selected = "blank"
      ),

      # Select variable for x-axis
      selectInput(
        inputId = "x",
        label = "X Variable:",
        choices = c(
          " " = "blank", "Amazon Price" = "Amazon_Price", "Number of Pieces" = "Pieces",
          "Number of Minifigures" = "Minifigures", "Number of Pages" = "Pages",
          "Number of Unique Pieces" = "Unique_Pieces"
        ),
        selected = "blank"
      ),

      # Select categorical variable
      selectInput(
        inputId = "categorical",
        label = "Categorical Variable:",
        choices = c("No categorical" = "blank", "Theme" = "Theme", "Size" = "Size"),
        selected = "blank"
      ),

      # Add checkbox for least squares regression line(s)
      checkboxInput(
        inputId = "lines",
        label = "Add least squares regression line",
        value = FALSE
      ),
    ),


    # Output(s)
    mainPanel(
      tabsetPanel(
        type = "tabs",
        tabPanel(
          title = "Plots",
          h4("Visualing LEGO Set Characteristics"),
          h5("LEGO City and Friends"),
          plotOutput(outputId = "scatterplot"),
          verbatimTextOutput(outputId = "eq") # Simple linear regression equation(s)
        ),
        tabPanel(
          title = "Multiple Regression",
          h4("Multiple Linear Regression Equation"),
          verbatimTextOutput(outputId = "eq2") # Multiple linear regression equation
        ),
        tabPanel(
          title = "Data Table",
          DT::dataTableOutput("legotable")
        )
      )
    )
  )
)


# Server
server <- function(input, output) {

  # Create scatterplot
  options <- c(
    "Amazon Price" = "Amazon_Price", "Number of Pieces" = "Pieces", "Number of Minifigures" = "Minifigures",
    "Number of Pages" = "Pages", "Number of Unique Pieces" = "Unique_Pieces"
  )

  output$scatterplot <- renderPlot({
    g <- ggplot(data = lego.CFD, aes_string(x = input$x, y = input$y)) +
      labs(x = names(options[which(options == input$x)]), y = names(options[which(options == input$y)])) +
      theme_bw()

    if (identical(input$dataFrame, "cf")) {
      g <- g %+% subset(lego.CFD, Theme %in% c("City", "Friends"))
    }

    if (identical(input$categorical, "Theme")) {
      g <- g + geom_point(aes(color = Theme))
    }

    else if (identical(input$categorical, "Size")) {
      g <- g + geom_point(aes(color = Size))
    }

    else {
      g <- g + geom_point()
    }

    if (input$lines & identical(input$categorical, "blank")) {
      g <- g + geom_smooth(method = "lm", se = FALSE)
    }

    if (input$lines & identical(input$categorical, "Theme")) {
      g <- g + geom_smooth(aes(color = Theme), method = "lm", se = FALSE)
    }

    if (input$lines & identical(input$categorical, "Size")) {
      g <- g + geom_smooth(aes(color = Size), method = "lm", se = FALSE)
    }

    g
  })


  # Run simple linear regression analysis for LEGO City, Friends, and Duplo combined
  runSimpleRegCFD <- reactive({
    regCFD <- lm(as.formula(paste(input$y, " ~ ", paste(input$x, collapse = "+"))), data = lego.CFD)
  })

  # Run simple linear regression analysis for LEGO City and Friends combined
  runSimpleRegCF <- reactive({
    regCF <- lm(as.formula(paste(input$y, " ~ ", paste(input$x, collapse = "+"))), data = subset(lego.CFD, Theme == "City" | Theme == "Friends"))
  })

  # Run simple linear regression analysis for LEGO City
  runSimpleRegC <- reactive({
    regC <- lm(as.formula(paste(input$y, " ~ ", paste(input$x, collapse = "+"))), data = subset(lego.CFD, Theme == "City"))
  })

  # Run simple linear regression analysis for LEGO Friends
  runSimpleRegF <- reactive({
    regF <- lm(as.formula(paste(input$y, " ~ ", paste(input$x, collapse = "+"))), data = subset(lego.CFD, Theme == "Friends"))
  })

  # Run simple linear regression analysis for LEGO Duplo
  runSimpleRegD <- reactive({
    regD <- lm(as.formula(paste(input$y, " ~ ", paste(input$x, collapse = "+"))), data = subset(lego.CFD, Theme_2 == ""))
  })

  # Run simple linear regression analysis for small bricks
  runSimpleRegS <- reactive({
    regS <- lm(as.formula(paste(input$y, " ~ ", paste(input$x, collapse = "+"))), data = subset(lego.CFD, Size == "Small"))
  })

  # Run simple linear regression analysis for large bricks
  runSimpleRegL <- reactive({
    regL <- lm(as.formula(paste(input$y, " ~ ", paste(input$x, collapse = "+"))), data = subset(lego.CFD, Size == "Large"))
  })

  # Run multiple linear regression analysis for LEGO City and Friends
  runMultipleRegCF <- reactive({
    reg2CF <- lm(as.formula(paste(input$y, " ~ ", paste(input$x, collapse = "+"), "+ Theme_2")), data = subset(lego.CFD, Theme == "City" | Theme == "Friends"))
  })

  # Run multiple linear regression analysis for size small and large bricks
  runMultipleRegSL <- reactive({
    reg2SL <- lm(as.formula(paste(input$y, " ~ ", paste(input$x, collapse = "+"), "+ Size_2")), data = lego.CFD)
  })


  # Output simple linear regression equation(s)
  output$eq <- renderText({
    if (input$lines & identical(input$categorical, "blank") & identical(input$dataFrame, "cf")) {
      interceptCF <- round(runSimpleRegCF()$coefficients[1], digits = 2)
      slopeCF <- round(runSimpleRegCF()$coefficients[2], digits = 2)
      regression <- paste("Predicted", input$y, "=", interceptCF, "+", slopeCF, "(", input$x, ")")
    }

    else if (input$lines & identical(input$categorical, "blank") & identical(input$dataFrame, "cfd")) {
      interceptCFD <- round(runSimpleRegCFD()$coefficients[1], digits = 2)
      slopeCFD <- round(runSimpleRegCFD()$coefficients[2], digits = 2)
      regression <- paste("Predicted", input$y, "=", interceptCFD, "+", slopeCFD, "(", input$x, ")")
    }

    else if (input$lines & identical(input$categorical, "Theme") & identical(input$dataFrame, "cf")) {
      interceptF <- round(runSimpleRegF()$coefficients[1], digits = 2)
      slopeF <- round(runSimpleRegF()$coefficients[2], digits = 2)
      interceptC <- round(runSimpleRegC()$coefficients[1], digits = 2)
      slopeC <- round(runSimpleRegC()$coefficients[2], digits = 2)
      regression <- paste(
        "City:  Predicted", input$y, "=", interceptC, "+", slopeC, "(", input$x, ")", "\n",
        "Friends:  Predicted", input$y, "=", interceptF, "+", slopeF, "(", input$x, ")"
      )
    }

    else if (input$lines & identical(input$categorical, "Theme") & identical(input$dataFrame, "cfd")) {
      interceptF <- round(runSimpleRegF()$coefficients[1], digits = 2)
      slopeF <- round(runSimpleRegF()$coefficients[2], digits = 2)
      interceptC <- round(runSimpleRegC()$coefficients[1], digits = 2)
      slopeC <- round(runSimpleRegC()$coefficients[2], digits = 2)
      interceptD <- round(runSimpleRegD()$coefficients[1], digits = 2)
      slopeD <- round(runSimpleRegD()$coefficients[2], digits = 2)
      regression <- paste(
        "City:  Predicted", input$y, "=", interceptC, "+", slopeC, "(", input$x, ")", "\n",
        "Friends:  Predicted", input$y, "=", interceptF, "+", slopeF, "(", input$x, ")", "\n",
        "Duplo:  Predicted", input$y, "=", interceptD, "+", slopeD, "(", input$x, ")"
      )
    }

    else if (input$lines & identical(input$categorical, "Size") & identical(input$dataFrame, "cf")) {
      interceptS <- round(runSimpleRegS()$coefficients[1], digits = 2)
      slopeS <- round(runSimpleRegS()$coefficients[2], digits = 2)
      regression <- paste("Small:  Predicted", input$y, "=", interceptS, "+", slopeS, "(", input$x, ")")
    }

    else if (input$lines & identical(input$categorical, "Size") & identical(input$dataFrame, "cfd")) {
      interceptS <- round(runSimpleRegS()$coefficients[1], digits = 2)
      slopeS <- round(runSimpleRegS()$coefficients[2], digits = 2)
      interceptL <- round(runSimpleRegL()$coefficients[1], digits = 2)
      slopeL <- round(runSimpleRegL()$coefficients[2], digits = 2)
      regression <- paste(
        "Small:  Predicted", input$y, "=", interceptS, "+", slopeS, "(", input$x, ")", "\n",
        "Large:  Predicted", input$y, "=", interceptL, "+", slopeL, "(", input$x, ")"
      )
    }

    else {
      regression <- paste("")
    }
    regression
  })

  # Output multiple linear regression equation
  output$eq2 <- renderText({
    if (identical(input$categorical, "Theme") & identical(input$dataFrame, "cf")) {
      intercept2 <- round(runMultipleRegCF()$coefficients[1], digits = 2)
      slope2 <- round(runMultipleRegCF()$coefficients[2], digits = 2)
      coefTheme <- round(runMultipleRegCF()$coefficients[3], digits = 2)
      regression <- paste("Predicted", input$y, "=", intercept2, "+", slope2, "(", input$x, ") +", coefTheme, "( Theme 2 )")
    }

    else if (identical(input$categorical, "Size") & identical(input$dataFrame, "cfd")) {
      intercept2 <- round(runMultipleRegSL()$coefficients[1], digits = 2)
      slope2 <- round(runMultipleRegSL()$coefficients[2], digits = 2)
      coefTheme <- round(runMultipleRegSL()$coefficients[3], digits = 2)
      regression <- paste("Predicted", input$y, "=", intercept2, "+", slope2, "(", input$x, ") +", coefTheme, "( Size 2 )")
    }

    else if (identical(input$categorical, "blank") | identical(input$x, "blank") | identical(input$y, "blank")) {
      regression <- paste(
        "Select your Y, X, and categorical variables, and appropriate data set", "\n",
        "from the dropdown menu's to see the multiple linear regression equation."
      )
    }

    else if (identical(input$categorical, "Size") & identical(input$dataFrame, "cf")) {
      regression <- paste(
        " Select the City, Friends, and DUPLO data set from the dropdown menu", "\n",
        "  to see the multiple linear regression equation with Size 2.", "\n", "\n",
        "OR", "\n", "\n",
        "Select the Theme categorical variable from the drop down menu", "\n",
        "  to see the mulitple linear regression equation with Theme 2."
      )
    }

    else if (identical(input$categorical, "Theme") & identical(input$dataFrame, "cfd")) {
      regression <- paste(
        " Select the City and Friends data set from the dropdown menu", "\n",
        "  to see the multiple linear regression equation with Theme 2.", "\n", "\n",
        "OR", "\n", "\n",
        "Select the Size categorical variable from the drop down menu", "\n",
        "  to see the mulitple linear regression equation with Size 2."
      )
    }
    regression
  })


  # Output Data Table
  output$legotable <- DT::renderDataTable({
    if (identical(input$dataFrame, "cf")) {
      DT::datatable(
        data = subset(lego.CFD, Theme == "City" | Theme == "Friends") %>% select(1:3, 19, 6, 4, 10, 9, 13, 15, 18),
        options = list(pageLength = 10),
        rownames = FALSE
      )
    }
    else {
      DT::datatable(
        data = lego.CFD %>% select(1:3, 19, 6, 4, 10, 9, 13, 15, 18),
        options = list(pageLength = 10),
        rownames = FALSE
      )
    }
  })
}


# Create a Shiny app object
shinyApp(ui = ui, server = server)
