#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
library(e1071)
library(shiny)
library(aplore3)
library(tidyverse)
library(summarytools)
library(randomForestSRC)
library(caret)
library(class)
library(tree)
library(rpart)
library(rpart.plot)
library(randomForest)



# Define UI for application that draws a histogram
ui <- navbarPage("Burns Study",
                 tabPanel("Introduction",
                          navlistPanel(
                        tabPanel("Variables",      HTML('The <b>goal</b> is to predict death in people who have been burned.  
                            The data can be found in the <i>burn1000</i> dataset from the <i>aplore3</i> package, which contains 1000 observations with following variables:
                                   
<pre class="tab"><b> id</b>
    Identification code (1 - 1000)
<br>
<b>facility</b>
    Burn facility (1 - 40)
<br>
<b>death</b>
    Hospital discharge status (0: Alive, 1: Dead)
<br>
<b>age</b>
    Age at admission (Years)
<br>
<b>gender</b>
    Gender (1: Female, 2: Male)
<br>
<b>race</b>
    Race (1: Non-White, 2: White)
<br>
<b>tbsa</b>
    Total burn surface area (0 - 100%)
<br>
<b>inh_inj</b>
    Burn involved inhalation injury (1: No, 2: Yes)
<br>
<b>flame</b>
    Flame involved in burn injury (1: No, 2: Yes)</pre> ')
 ),
 tabPanel("Methods",HTML('This Shiny App is designed to walk through basic summary statistics for each variables in the <i>burn1000</i> dataset and demonstrate the performance of prediction in different machine learning(ML) algorithms including KNN, logistic regression, CART and random forest.
                         <br>
                         <br>
                             Specifically, the dataset was divided as a train dataset(80%) and test dataset(20%) by stratified sampling based on death status, the table indicating alive/dead versus train/test is displayed below. Each ML method was first fit on the training dataset and then the performance was assessed based on test dataset'),
          h3('Table of Train/Test by death'),tableOutput('foo1')))), 
                 tabPanel("Descriptive Statistics/Graphs",
                          sidebarLayout(
                              sidebarPanel(
                                  selectInput("variable", "Variable:",
                                              list("facility" = "facility", 
                                                   "death" = "death", 
                                                   "age" = "age",
                                                   "gender"="gender",
                                                   "race"="race",
                                                   "tbsa"="tbsa",
                                                   "inh_inj"="inh_inj",
                                                   "flame"="flame")),
                                  #checkboxInput("outliers", "Show outliers", FALSE)
                                  # select plot type
                                  br(),
                                  selectInput("Plot", "Plot Type", 
                                              choices = c("Histogram/Barplot", "QQplot")),
                                  #add button to show table of summary
                                  actionButton("showSummary", "Display summary statistics!")
                              ),
                              mainPanel(h1('Interactive Variable Summary'),
                                plotOutput("Descriptive"),
                                br(),
                                verbatimTextOutput("Summary") 
                              )
                          )
                 ),
                 tabPanel("KNN",navlistPanel(
                     tabPanel('Tuning',p('KNN (K nearest neighbor) is a non-parametric machine learning algorithm for classification based on feature similarity: how closely out-of-sample features resemble our training set determines how we classify a given data point.'),
                     p('Dataset has been converted to numeric matrix in order to applied KNN algorithm. 
                     Specifically, factor levels of binary variables are converted as 0 and 1. Also numeric/integer variables have been normalized since KNN sensitive to different scales. 10-fold Cross Validation was applied to choose the optimal K (number of nearest neighbors) with results displayed below.
                                         K=6 was used for fitting final KNN model for prediction.'),
                              verbatimTextOutput("knn.cv")),
                     tabPanel('Performance',p('KNN prediction has 92% accuracy and a Kappa of 0.64.'),verbatimTextOutput("knn.result")  
                                       
                                  
                          ))),   

                          
                 tabPanel("Logistic Regression",navlistPanel(
                     tabPanel('Model Summary',p('Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable.'),

                          p('Below is the logistic regression results, there are three variables: age, tbsa and inh_in are statistically significant(p < 0.05), suggesting they are associated with the outcome.'),
                          verbatimTextOutput("Log.summary"),
                          p('Below shows the estimated Odds Ratios with 95% confidence intervals'),
                          verbatimTextOutput("Log.OR")
                 ),
                 tabPanel('Performance',p('Below shows the prediction performance of logistic regression results. Individuals in the test dataset with predicted probabilities
                                          <=0.63 were categorized as Alive while >0.63 were categorized as Dead. 
                                          The logistic regression model has 92.5% accuracy and a Kappa of 0.65. '),
                          verbatimTextOutput("Logistics")))),
                 tabPanel("TREE",navlistPanel(
                     tabPanel('TREE Figure',p('CART or "tree" is a binary decision tree that is constructed by splitting a node into two child nodes repeatedly, beginning with the root node that contains the whole learning sample.'),
                              p('The graph below shows how the "tree" splits.'),
                              p('As we can see variable tbsa is picked first and then age, the whole tree is splitting only based these two variable, suggesting total burn surface area and age are the most informative variables in the model.'),
                              plotOutput("tree.plot")),
                     tabPanel('Performance', p('Below shows the prediction performance of TREE results. 
                     It has 92.5% accuracy and a Kappa of 0.66.'),
                                                
                          verbatimTextOutput("tree.result"))
                 )),
                 tabPanel("Random Forest",navlistPanel(
                         tabPanel('Importance Plot',p('Random forest is an algorithm for classification and regression based on a forest of trees using random inputs.'),
                         p('The graph below on the left shows how the random forest error rate behaves as the number of trees grows. 
                         The graph on the right shows the variable importance whichs a measure of how big of a role each variable\'s played in training the model.
                        In the training dataset, we can see both tbsa and age have relatively high importance measure, and they are also much higher relative to the other variables 
                                                      for the "Dead" class, suggesting age and total burn surface area are the most important factors in separating out the "Dead" group'),
                                  plotOutput("vim.plot"))),
                         tabPanel('Performance',p('Below shows the prediction performance of RF results. 
                    Each time we run a RF model, the result changes a little since this method relies on random selection of observations and features as part of the algorithm, the accuracy is around 92% with Kappa around 0.64. '),
                          verbatimTextOutput("rf"))),
                 tabPanel("Code",HTML('My code can be found <a href="https://github.com/hxz305/Burn1000-Analysis">here</a>')     
                          
                 ))
                 
                



# Define server logic required to draw a histogram
server <- function(input, output, session) {
    
    ###########################Summary#################################
    
    Names <- reactive({paste(input$variable)})
    
    output$Descriptive <- renderPlot({
        
        plotdata <- burn1000[, input$variable,drop=F]
    
        var <- reactive({input$variable})
       
        # choose the type of plot
        if (input$Plot == "Histogram/Barplot"){
            
            # whether the variable is continuous or not
            if (is.numeric(plotdata[,1])|is.integer(plotdata[,1])){ 
                
                # histogram for continuous variable
                density<-ggplot(data=plotdata, aes_string(x=input$variable))
                density+geom_histogram(binwidth=(max(plotdata[,1])-min(plotdata[,1]))/20, color="black", fill="steelblue", aes(y=..density..)) +
                    labs(x=Names())+geom_density(stat="density", alpha=I(0.2), fill="blue") +
                     ylab("Density") + ggtitle("Histogram & Density Curve")
                #ggplot(burn1000, aes(names())) + geom_histogram(aes(y=..density.., fill=..density..))+geom_density()
                #ggplot(data.frame(plotdata),aes(x=plotdata)) + geom_histogram(aes(y=..density.., fill=..density..))+geom_density()
               #print(density)
                #hist(plotdata)
                #print(class(input$variable))
                #ggplot(data.frame(burn1000),aes(input$variable)) + geom_bar(aes(y=..density.., fill=..density..))#+geom_density()
                
            } else {
                
                # barplot for categorical variable
                #ggplot(data.frame(plotdata),aes_string(group=input$variable))+ geom_bar(fill=c('lightcoral','deepskyblue3'),aes(y = ..prop.., fill = factor(..x..)), stat="count")+
                 #   labs(x=Names())+geom_text(aes( label = scales::percent(..prop..),
                   #                               y= ..prop.. ), stat= "count", vjust = -.5)
                #ggplot(data = plotdata,aes_string(x = input$variable)) + scale_y_continuous(labels = scales::percent_format())+
                 #   geom_bar(mapping = aes(y = ..prop.., group = 2), stat = "count")+
                  #  geom_text(aes( label = scales::percent(..prop..),
                   #                                               y= ..prop.. ), stat= "count", vjust = -.5)
                ggplot(plotdata, aes_string(input$variable), fill=input$variable)+
                 geom_bar(fill=c('lightcoral','deepskyblue3')) + ggtitle("Barplot with percentages")+
                 geom_label(
                    aes(label=scales::percent(..count../sum(..count..))),
                    stat='count',
                    nudge_y=5 #,
                    #format_string='{:.1f}%'
                )
            }
            
            # if select the QQplot
        } else if (input$Plot == "QQplot") {
            #browser()
            # whether the variable is continuous or not
            if (is.numeric(plotdata[,1])|is.integer(plotdata[,1])){
                plotdata2=plotdata
                plotdata2[,1]=scale(plotdata2[,1],center = T,scale = T)
                # QQplot for continuous variables
                ggplot(plotdata2,aes_string(sample=input$variable)) + stat_qq(color='firebrick4')+
                    labs(x=Names())+geom_abline(aes(intercept=0,slope=1))+ ggtitle("QQplot with y=x reference line")
                
            }else{ggplot(plotdata,aes_string(sample=input$variable),fill='white')+ggtitle('QQplot not available for categorical variables')}
            }
    })
    
    #output$Summary <- renderTable({
    #   dfSummary(data.frame(burn1000[, input$variable]),plain.ascii = FALSE, style = "multiline", 
    #             graph.col=F,valid.col = FALSE, tmp.img.dir = "/tmp")
    # })
    
    
    theSummary <- eventReactive(input$showSummary,
                                {burn1000[, input$variable]
                                    })
    
    output$Summary <-renderPrint({
        Variable <- burn1000[, input$variable]
        label(Variable)=Names()
        if (input$showSummary){dfSummary(Variable,headings=F,varnumbers=F,graph.col=F,valid.col = F)
        } 
        
    })
    
    eventReactive(input$showSummary,
                  {burn1000[, input$variable]
                  })
    
    
    #80:20 sampling stratified by outcome
    dat=burn1000[,-c(1,2)]
    
    
    dat.Alive=dat[dat$death=='Alive',]
    dat.Dead=dat[dat$death=='Dead',]
    set.seed(11)
    Alive.idx=sample(1:dim(dat.Alive)[1],0.8*dim(dat.Alive)[1])
    set.seed(22)
    Dead.idx=sample(1:dim(dat.Dead)[1],0.8*dim(dat.Dead)[1])
    
    
    #train.idx=sample(1:dim(dat)[1],0.7*dim(dat)[1])
    train.data=rbind(dat.Alive[Alive.idx,],dat.Dead[Dead.idx,])
    test.data=rbind(dat.Alive[-Alive.idx,],dat.Dead[-Dead.idx,])
    
    output$train.table = renderPrint({
    total.data=rbind(train.data,test.data)
    total.data$group=c(rep('Train',800),rep('Test',200))
    table(total.data$group,total.data$death)
    })
    
    output$foo <- renderTable({
        as.data.frame.matrix(table(c(1, 1, 1, 2, 3), c(2, 2, 3, 4, 3)))
    })
    bold <- function(x){
        paste0('<b>', x, '</b>')
    }
    
    output$foo1 <- renderTable({
        total.data=rbind(train.data,test.data)
        total.data$group=c(rep('Train',800),rep('Test',200))
        my_tbl = addmargins(table(total.data$group,total.data$death))
        print_tbl = as.data.frame.matrix(my_tbl)
        print_tbl
    }, include.rownames=TRUE,include.colnames=TRUE,digits = 0,auto=T,sanitize.rownames.function = bold)
    
    ###########################logistic#################################
    mod.log <- glm(death ~ ., family='binomial',data=train.data)
    log.pred = predict(mod.log,test.data,type='response')
    table(log.pred>.63, test.data$death)
    log.prediction = as.factor(dplyr::if_else(log.pred<=.63,0,1))
    
    log.pre=log.prediction
    levels(log.pre)=c('Alive','Dead')
    
    confusionMatrix(log.pre, 
                    test.data$death,positive = "Dead")
    
    output$Logistics <-renderPrint({
        mod.log <- glm(death ~ ., family='binomial',data=train.data)
        log.pred = predict(mod.log,test.data,type='response')
        table(log.pred>.63, test.data$death)
        log.prediction = as.factor(dplyr::if_else(log.pred<=.63,0,1))
        log.pre=log.prediction
        levels(log.pre)=c('Alive','Dead')
        
        confusionMatrix(log.pre, 
                        test.data$death,positive = "Dead")
        
    })
    
    output$Log.summary <-renderPrint({
        summary(mod.log)
 
    })
    output$Log.OR <-renderPrint({
        OR=exp(mod.log$coefficients[-1])
        CI=exp(confint(mod.log))[-1,]
        df=data.frame(OR,CI)
        names(df)=c('OR','CI.Low','CI.High')
        df
    })
    ###########################KNN#################################
    set.seed(1212)
    #train.idx=sample(1:dim(dat)[1],0.8*dim(dat)[1])
    #train.data=dat[train.idx,]
    #test.data=dat[-train.idx,]
    
    
    #convert factor as numeric
    #dat=burn1000[,-c(1,2)]
    data=dat
    data$gender=as.numeric(data$gender)-1 #1 is male, 0 is female
    data$race=as.numeric(data$race)-1 #1 is white, 0 is Non-white
    data$inh_inj=as.numeric(data$inh_inj)-1 #1 is Yes, 0 is No
    data$flame=as.numeric(data$flame)-1 #1 is Yes, 0 is No
    
    #Normalizing
    normalize <- function(x){
        return((x-min(x))/(max(x)- min(x)))
    }
    data.knn=apply(data[,-1],2,normalize)
    data.knn=data.frame(death=data$death,data.knn)
    knn.Alive=data[data.knn$death=='Alive',]
    knn.Dead=data[data.knn$death=='Dead',]
    
    
    
    #assigning train and test dataset
    train.knn=rbind(knn.Alive[Alive.idx,],knn.Dead[Dead.idx,])
    test.knn=rbind(knn.Alive[-Alive.idx,],knn.Dead[-Dead.idx,])
    
    
    #10-fold cross-validation on train dataset to choose optimal K, on k=1:30
    trControl <- trainControl(method  = "cv",
                              number  = 10)
    
    output$knn.cv <-renderPrint({
        set.seed(8)
        knn.fit <- train(death ~ .,
                         method     = "knn",
                         tuneGrid   = expand.grid(k = 1:30),
                         trControl  = trControl,
                         metric     = "Accuracy",
                         data       = train.knn)
        knn.fit
    })
    output$knn.result <-renderPrint({
        set.seed(8)
        knn.pred=knn(train.knn[,-1],test.knn[,-1],train.knn$death,k=6)
        confusionMatrix(knn.pred, 
                        test.knn$death,positive = "Dead")
        
    })
    
    ###########################Tree#################################
    
    
    
    output$tree.plot <-renderPlot({
        set.seed(1213)
        tree.mod=rpart(death~.,train.data)
        rpart.plot(tree.mod)
    })
    output$tree.result <-renderPrint({
        set.seed(1213)
        tree.mod=rpart(death~.,train.data)
        tree.pred=predict(tree.mod,test.data,type="class")
        confusionMatrix(tree.pred, 
                        test.data$death,positive = "Dead")
    })
    
    ###########################Random Forest#################################

    output$rf <-renderPrint({
        set.seed(21)
        mod.rf <- rfsrc(death ~ .,data=train.data,importance = T)
        
        rf.pred=predict(mod.rf,newdata=test.data,type="class")
        rf.pred=predict(mod.rf,test.data)$class
        
        confusionMatrix(rf.pred, 
                        test.data$death,positive = "Dead")
        
        
        rf.mod=randomForest(death ~ .,data=train.data,mtry=6,importance=TRUE)
    yhat.rf = predict(rf.mod,newdata=test.data)
    confusionMatrix(yhat.rf, 
                    test.data$death,positive = "Dead")
    })
    output$vim.plot <-renderPlot({
    #importance(rf.mod)
        set.seed(21)
        mod.rf <- rfsrc(death ~ .,data=train.data,importance = T)
        plot(mod.rf,main='Random Forest Plot')
    })
    
}

# Run the application 
shinyApp(ui = ui, server = server)

