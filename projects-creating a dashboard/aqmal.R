# Load required libraries
library(shiny)          # Shiny library for creating web applications
library(shinydashboard) # Shiny Dashboard library for creating dashboards
library(ggplot2)        # ggplot2 library for data visualization
library(tidyverse)      # Tidyverse library for data manipulation
library(dplyr)          # dplyr library for data manipulation
library(corrplot)       # corrplot library for correlation plots
library(readr)

# Set working directory to the desired location
setwd("C:/MyProject(DA)")

# Read the "corona_virus.csv" file into the corona variable
corona <- read.csv("corona_virus.csv")

# Data enrichment and manipulation
corona[is.na(corona) | corona == "" | corona == 'NA'] <- 0    # Replace NA, empty strings, and 'NA' with 0
corona[is.na(corona) | corona == 'N/A' | corona == " "] <- 0  # Replace NA, 'N/A', and empty spaces with 0

# Define the layout of the app

# Fluid rows for different sections of the dashboard
frowTab6 <- fluidRow(
  box(
    title = "Density Chart",
    status = "primary",
    solidHeader = TRUE,
    collapsible = TRUE,
    width = NULL,
    background = "maroon",
    plotOutput("DensPlot", height = "300px") # Render a density plot in this box
  )
)

frowTab5 <- fluidRow(
  box(
    title = "Pie Chart",
    status = "primary",
    solidHeader = TRUE,
    collapsible = TRUE,
    width = NULL,
    background = "black",
    plotOutput("piChart", height = "300px") # Render a pie chart in this box
  )
)

frowTab4 <- fluidRow(
  box(
    title = "Tufte Boxplot",
    status = "primary",
    solidHeader = TRUE,
    collapsible = TRUE,
    width = NULL,
    background = "lime",
    plotOutput("tufteBoxplot", height = "300px") # Render a Tufte boxplot in this box
  )
)

frowTab3 <- fluidRow(
  box(
    title = "Scatter Plot",
    status = "primary",
    solidHeader = TRUE,
    collapsible = TRUE,
    width = NULL,
    background = "green",
    plotOutput("scatterPlot", height = "300px") # Render a scatter plot in this box
  )
)

frowTab2 <- fluidRow(
  box(
    title = "Bar Chart",
    status = "primary",
    solidHeader = TRUE,
    collapsible = TRUE,
    width = NULL,
    background = "blue",
    plotOutput("barChart", height = "300px") # Render a bar chart in this box
  )
)

frowTab1 <- fluidRow(
  box(
    title="BoxChart",
    status = "primary",
    solidHeader = TRUE,
    collapsible = TRUE,
    background = "red",
    width = NULL,
    plotOutput("boxchartID",height = "300px") # Render a boxplot in this box
  )
)

# Define the layout of the app continued...

# Fluid rows for the main section of the dashboard
frow1 <- fluidRow(
  box(
    title = "Dataset Header of Corona Virus Data 2023",
    status = "info",
    solidHeader = TRUE,
    collapsible = TRUE,
    width = NULL,
    verbatimTextOutput("header_output") # Render the dataset header in this box
  ),
  box(
    title = "Display First 5 rows",
    status = "info",
    solidHeader = TRUE,
    collapsible = TRUE,
    width = NULL,
    background = "red",
    tableOutput("first5rows_output") # Render the first 5 rows of the dataset in a table in this box
  )
)

frow2 <- fluidRow(
  tags$div(
    id = "dataset-summary", 
    class = "container",
    style = "margin-justify",
    tags$h3("Dataset Summary"),
    tags$p("The WHO coronavirus (COVID-19) dashboard presents official daily counts of COVID-19 cases, deaths and vaccine utilisation reported by countries, territories and areas. Data is subject to continuous verification and change, and all counts are subject to variations in case detection, definitions, laboratory testing, vaccination strategy, and reporting strategies. Data for Bonaire, Sint Eustatius and Saba have been disaggregated and displayed at the subnational level.")
  ),
  box(
    title = "Testing Image",
    status = "primary",
    solidHeader = TRUE,
    collapsible = TRUE,
    width = NULL,
    column(
      width = 12,
      align = "center",
      h4("Details of Covid-19"),
      imageOutput("Image_2") # Render an image in this box
    ),
    column(
      width = 12,
      h4("Corona Virus World Data 2023"),
      align = "center",
      imageOutput("Image_3",height = "410px") # Render an image in this box with a specified height
    ),
    column(
      width = 12,
      h4("A Virus Attacks a Cell"),
      align = "center",
      HTML('<iframe width="800" height="400" src="https://www.youtube.com/embed/QGF-UNk3UdE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>') # Render an image in this box with a specified height
    )
  )
)

frow3 <- fluidRow(    
  box(
    title = "Dataset Properties",
    status = "info",
    solidHeader = TRUE,
    collapsible = TRUE,
    width = NULL,
    background = "maroon",
    verbatimTextOutput("strDset") # Render the dataset properties in this box
  )
)

frow4 <- fluidRow(
  box(
    title = "Dataset Summary",
    status = "primary",
    solidHeader = TRUE,
    collapsible = TRUE,
    width = NULL,
    background = "yellow",
    verbatimTextOutput("sumDset") # Render the dataset summary in this box
  )
)

# Define the sidebar menu
sidebar <- dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
      menuItem("Visit - us", icon = icon("send", lib = 'glyphicon'), href = "https://www.ums.edu.my/fssa")
    ),
    h2("Properties & Summary Data"),
    selectInput(
      inputId = "dvInput",
      label = "Column Category",
      choices = c("All_Columns", "Country_Other", "Total_Cases", "New_Cases", "Total_Deaths", "New_Deaths",
                  "Total_Recovered", "New_Recovered", "Active_Cases", "Serious_Critical", "Tot_Cases_1M_pop",
                  "Deaths_1Mpop", "Total_Tests", "Tests_1Mpop", "Population")
    ),
    h2("Download Any Two Dataset"),
    selectInput(
      inputId = "dvInput1",
      label = "Column 1",
      choices = c("All_Columns", "Country_Other", "Total_Cases", "New_Cases", "Total_Deaths", "New_Deaths",
                  "Total_Recovered", "New_Recovered", "Active_Cases", "Serious_Critical", "Tot_Cases_1M_pop",
                  "Deaths_1Mpop", "Total_Tests", "Tests_1Mpop", "Population")
    ),
    selectInput(
      inputId = "dvInput2",
      label = "Column 2",
      choices = c("All_Columns", "Country_Other", "Total_Cases", "New_Cases", "Total_Deaths", "New_Deaths",
                  "Total_Recovered", "New_Recovered", "Active_Cases", "Serious_Critical", "Tot_Cases_1M_pop",
                  "Deaths_1Mpop", "Total_Tests", "Tests_1Mpop", "Population")
    ),
    downloadButton("download", "Download Columns")
)

# Define the dashboard header
dbheader <- dashboardHeader(title = "Amirul Aqmal", titleWidth = "calc(100%-44px)") # Dashboard header with title

# Define the dashboard body
dbody <- dashboardBody(
  tabItems(
    tabItem(tabName = "dashboard",
            h2("Corona Virus Data 2023"), # Heading for the dashboard tab
            tabsetPanel(
              tabPanel("Main",frow1,frow2,frow3,frow4), # Main tab panel with rows 1, 2, 3, and 4
              tabPanel("Graph",frowTab1,frowTab2,frowTab3,frowTab4,frowTab5,frowTab6) # Graph tab panel with rows Tab1 to Tab6
            )
    )
  )
)

# Complete dashboard page with header, sidebar, and body
myPage <- dashboardPage(dbheader, sidebar, dbody, skin = "blue")

# Define the UI with the dashboard page
ui <- fluidPage(myPage)

# Define the server logic
server <- function(input, output) {
  
  
  datasetInput1 <- reactive({
    
    column1 <- as.character(input$dvInput1)
    column2 <- as.character(input$dvInput2)
    
    if (column1 == "All_Columns" & column2 == "All_Columns") {
      corona
    } else if (column1 == "All_Columns") {
      corona[, column2, drop = FALSE]
    } else if (column2 == "All_Columns") {
      corona[, column1, drop = FALSE]
    } else {
      corona[, c(column1, column2), drop = FALSE]
    }
  })
  
  datasetInput <- reactive({
    
    column <- as.character(input$dvInput)
    
    if (column == "All_Columns") {
      corona # Return the entire dataset if "All_Columns" is selected
    } else {
      corona[, column, drop = FALSE] # Return the selected column of the dataset
    }
  })
  
  output$sumDset <- renderPrint({
    dvInput <- datasetInput()
    if (is.data.frame(dvInput)) {
      if (length(dvInput) == 1) {
        summary(dvInput) # Render the summary of the dataset if it is a data frame with length 1
      } else {
        lapply(dvInput, summary) # Render the summary of each column if the dataset is a data frame with length > 1
      }
    } else {
      summary(dvInput) # Render the summary of the dataset if it is not a data frame
    }
  })
  
  output$header_output <- renderPrint({
    colnames(datasetInput()) # Render the column names of the dataset
  })
  
  output$first5rows_output <- renderTable({
    head(datasetInput(), 5) # Render the first 5 rows of the dataset in a table
  })
  
  output$Image_3 <- renderImage({ # Renders Image 3
    list(
      src = "side1.jpg", # Image source file
      filetype = "image/jpeg", # File type of the image
      alt = "Image 1" # Alternative text for the image
    )
  }, deleteFile = FALSE)
  
  output$Image_2 <- renderImage({ # Renders Image 2
    list(
      src = "side3.jpg", # Image source file
      filetype = "image/jpeg", # File type of the image
      alt = "Image 2" # Alternative text for the image
    )
  }, deleteFile = FALSE)
  
  output$DensPlot <- renderPlot({ # Renders Density Plot
    df <- corona %>% 
      group_by(Serious_Critical) %>% # Grouping by 'Serious_Critical'
      filter(n() >= 2) %>% # Keeping groups with at least two data points
      ungroup() %>% # Ungrouping the data
      arrange(desc(Serious_Critical)) %>% 
      head(25) 
    
    if (nrow(df) >= 2) { # Check if there are at least two rows in the filtered data
      ggplot(df, aes(x = Serious_Critical, group = 1)) + # Add 'group = 1' to aes()
        geom_density(fill = "blue", alpha = 0.5) +
        labs(x = "Serious Critical level", y = "Density") +
        theme_minimal()
    } else {
      # Handle case when there are not enough data points to create the plot
      plot(NULL, xlim = c(0, 1), ylim = c(0, 1), xlab = "", ylab = "")
      text(0.5, 0.5, "Insufficient data to create plot", cex = 1.2, col = "red", font = 2)
    }
  })
  
  output$piChart <- renderPlot({ # Renders Pie Chart
    datasetInput() %>% # Retrieves the dataset input
      filter(!is.na(Total_Cases) & !is.na(Country_Other)) %>% # Filter out rows with missing values in Total_Cases or Country_Other
      mutate(Total_Cases = parse_number(Total_Cases)) %>% # Extracts numeric values from Total_Cases column
      group_by(Country_Other) %>% # Groups data by 'Country_Other'
      summarize(total_cases = sum(Total_Cases, na.rm = TRUE), .groups = "drop") %>% # Summarizes the total cases for each country, ignoring NAs, and drops the grouping
      filter(total_cases > 0) %>% # Filter out countries with zero total cases
      arrange(desc(total_cases)) %>% # Arranges the data in descending order of total cases
      slice(1:10) %>% # Selects the top 10 countries with the highest total cases
      ggplot(aes(x = "", y = total_cases, fill = Country_Other)) + # Creates a ggplot object with total_cases on y-axis and Country_Other for fill
      geom_bar(stat = "identity", width = 1, color = "white") + # Adds a bar plot with white color
      coord_polar(theta = "y") + # Makes the plot polar
      labs(fill = "Country", title = "Top 10 Countries by Total Cases") + # Labels the fill and title
      theme_void() + # Applies a theme with no background elements
      theme(legend.position = "right") # Positions the legend on the right
  })
  
  output$tufteBoxplot <- renderPlot({ # Renders Tufte Boxplot
    # Create a subset of the data
    df <- corona %>% # Selects 'Country_Other' and 'Total_Cases' columns from 'corona' dataset
      select(Country_Other, Total_Cases) %>%
      arrange(desc(Total_Cases)) %>% # Arranges the data in descending order of 'Total_Cases'
      head(10) # Selects the top 10 rows
    
    # Create the Tufte Boxplot
    ggplot(df, aes(x = Country_Other, y = Total_Cases)) +  # Creates a ggplot object with 'Country_Other' on x-axis and 'Total_Cases' on y-axis
      geom_boxplot(
        color = "black", # Sets the color of the boxplot
        fill = "red", # Sets the fill color of the boxplot
        outlier.shape = NA, # Removes outlier points
        coef = 0 # Adjusts the boxplot whiskers
      ) +
      geom_jitter(
        color = "black", # Sets the color of the jitter points
        size = 1, # Sets the size of the jitter points
        width = 0.1, # Sets the width of the jitter points
        alpha = 0.6 # Sets the transparency of the jitter points
      ) +
      labs(x = "Country", y = "Total Cases") + # Labels the axes
      theme_bw() + # Applies a black and white theme
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Adjusts the x-axis text angle
  })
  
  output$scatterPlot <- renderPlot({ # Renders Scatter Plot
    ggplot(data = corona, aes(x = Total_Cases, y = Total_Deaths)) + # Creates a ggplot object with 'Total_Cases' on x-axis and 'Total_Deaths' on y-axis
      geom_point(color = "red") + # Adds scatter points with red color
      xlab("Total Cases Data") + # Labels the x-axis
      ylab("Total Deaths Data") + # Labels the y-axis
      theme_bw() # Applies a black and white theme
  })
  
  output$barChart <- renderPlot({ # Renders Bar Chart
    ggplot(data = corona, aes(x = Country_Other, y = Total_Cases)) + # Creates a ggplot object with 'Country_Other' on x-axis and 'Total_Cases' on y-axis
      geom_bar(stat = "identity", fill = "blue") + # Adds a bar plot with blue color
      xlab("Country, Other Data") + # Labels the x-axis
      ylab("Total Cases Data") + # Labels the y-axis
      theme_bw() # Applies a black and white theme
  })
  
  output$boxchartID <- renderPlot({ # Renders Boxplot
    df <- ggplot(data = corona,aes(x= Country_Other, y = Total_Cases))+ xlab('Country,Other Data') + ylab('Total Cases Data') + geom_boxplot(
      color="black", # Sets the color of the boxplot
      fill="red", # Sets the fill color of the boxplot
      alpha=0.2, # Sets the transparency of the boxplot
      
      notch=TRUE, # Draws a notch around the median
      notchwidth = 0.8, # Sets the width of the notch
      
      # custom outliers
      outlier.colour="red", # Sets the color of outlier points
      outlier.fill="red",  # Sets the fill color of outlier points
      outlier.size=3)+ theme_bw() # Sets the size of outlier points and Applies a black and white theme
    
    df
  })
  
  output$strDset <- renderPrint({  # Renders Dataset Structure
    str(datasetInput()) # Prints the structure of the dataset input
  })

  output$download <- downloadHandler(  # Handles download of data
    filename = function() {
      paste("Dataset.csv", sep = "") # Sets the filename for download
    },
    content = function(file) {
      dataset <- datasetInput1() # Retrieves the dataset input
      write.csv(dataset, file, row.names = FALSE) # Writes the filtered dataset to a CSV file
    }
  )  
    
}

# Run the Shiny app
shinyApp(ui, server)
