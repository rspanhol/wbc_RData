rm(list = ls())

library(tidyverse)
library(tidymodels)
library(janitor)
library(tictoc)

#install.packages("parallel")
#install.packages("doParallel")
#cargar datos

getwd()

load("wbc_data.RData")

glimpse(wbc)

#Division de los datos en train y test

set.seed(777)
wbc_split <- initial_split(wbc, strata = diagnosis)

wbc_train <- training(wbc_split)
wbc_test <- testing(wbc_split)



wbc_train %>% tabyl(diagnosis)
wbc_test %>% tabyl(diagnosis)


#Plantear un workflow

knn_recipe <- recipe(diagnosis ~., data= wbc_train) %>% 
                step_normalize(all_predictors())

knn_recipe


#Plantear el modelo

knn_mod <- nearest_neighbor(neighbors = tune(), weight_func = tune(),
                                  dist_power = tune()) %>% 
                              set_engine("kknn") %>% 
                              set_mode("classification")

knn_mod


#Construir un workflow

wk1 <- workflow() %>% 
                add_recipe(knn_recipe) %>% 
                add_model(knn_mod)

wk1


#Plantear en primer lugar los folds de la validación cruzada

set.seed(777)
wbc_folds <- vfold_cv(wbc_train, v = 10, strata = diagnosis)


wbc_folds$splits[[1]][1]


#El planteamiento del ajuste de K, dist_power, weigth_func
#
#
#Cuadriculas(grids) regulares
#

knn_params <- parameters(neighbors(range = c(1,50)),
                         dist_power(range = c(1,2)),
                         weight_func(values_weight_func[1:10]))


knn_params


knn_reg_grid <- grid_regular(knn_params,
                             levels=c(50,5,10))

knn_reg_grid

#Un gráfico de una regular grid


knn_reg_grid %>% 
  ggplot(aes(x=neighbors, y = dist_power, color=  weight_func))+
  geom_point() + facet_wrap(~weight_func)


#Parelización del modelo a evaluar
#

mis_metricas <- metric_set(accuracy,roc_auc)

parallel::detectCores()

clus <- parallel::makeCluster(8)
doParallel::registerDoParallel(clus)
tic()
knn_regular_res <- tune_grid(
  wk1,
  resamples = wbc_folds,
  control = control_resamples(save_pred = TRUE),
  grid = knn_reg_grid,
  metrics = mis_metricas)

parallel::stopCluster(clus)

toc()

#1123.78 segundos
#
#


#Representación de la cuadricula regular
knn_regular_res %>% autoplot()



knn_regular_res %>% 
  collect_metrics(summarize= TRUE) %>% arrange(desc(mean))


knn_regular_res %>% show_best(metric= "roc_auc", n= 1)



#Modelo 2 con cuadriculas irregulares
#
#Establecimiento de la  cuadricula aleatoria
knn_random_grid <- grid_random(knn_params,size = 50)


knn_random_grid %>% 
  ggplot(aes(x = neighbors, y = dist_power))+
  geom_point(aes(color = weight_func))

#Desarrollo del modelo en paralelo

clus <- parallel::makeCluster(8)
doParallel::registerDoParallel(clus)
tic()
knn_random_res <- tune_grid(
  wk1,
  resamples = wbc_folds,
  control = control_resamples(save_pred = TRUE),
  grid = knn_random_grid,
  metrics = mis_metricas)

parallel::stopCluster(clus)

toc()

#Tiempo de ejecución 117.49 segundos
#
knn_random_res %>% autoplot()

#Comparar los modelos

knn_regular_res %>% show_best(metric= "roc_auc", n= 1)
knn_random_res %>% show_best(metric= "roc_auc", n= 1)


#Modelo a escoger

knn_best <- knn_regular_res %>% tune::select_best(metric= "roc_auc")

knn_best
#Finalización del modelo knn que usa el mejor ajuste (tune)

knn_mod_final <- knn_mod %>% finalize_model(knn_best)

knn_mod_final


#Finalizar el recipe

knn_recipe_final <- knn_recipe %>% 
  finalize_recipe(knn_best)

knn_recipe


#Actualización del flujo de trabajo
wk1
wk_final <- wk1 %>% 
  update_recipe(knn_recipe_final) %>% 
  update_model(knn_mod_final)

wk_final


#Se corre el último ajuste del modelo
#

 #establecimiento del modelo final
knn_final_res <- last_fit(
  wk_final, split = wbc_split)




knn_final_res %>% collect_metrics()

#Representación gráfica de la curva roc
knn_final_res %>% 
  collect_predictions() %>% 
  roc_curve(truth= diagnosis, estimator= .pred_M) %>% 
  autoplot()



#Representaci'on de matriz de confusión con datos de prueba
knn_final_res %>% 
  collect_predictions() %>% 
  conf_mat(truth= diagnosis,estimate = .pred_class) %>% autoplot()

# 
# sessionInfo()
#R version 4.2.2 (2022-10-31 ucrt)
# Platform: x86_64-w64-mingw32/x64 (64-bit)
# Running under: Windows 10 x64 (build 22621)
# 
# Matrix products: default
# 
# locale:
#   [1] LC_COLLATE=English_United States.utf8  LC_CTYPE=English_United States.utf8   
# [3] LC_MONETARY=English_United States.utf8 LC_NUMERIC=C                          
# [5] LC_TIME=English_United States.utf8    
# 
# attached base packages:
#   [1] stats     graphics  grDevices utils     datasets  methods   base     
# 
# other attached packages:
#   [1] tictoc_1.0.1       janitor_2.1.0      yardstick_1.0.0    workflowsets_1.0.0
# [5] workflows_1.0.0    tune_1.0.0         rsample_1.0.0      recipes_1.0.1     
# [9] parsnip_1.0.0      modeldata_1.0.1    infer_1.0.2        dials_1.0.0       
# [13] scales_1.2.1       broom_1.0.0        tidymodels_1.0.0   forcats_0.5.1     
# [17] stringr_1.5.0      dplyr_1.0.9        purrr_0.3.4        readr_2.1.2       
# [21] tidyr_1.2.0        tibble_3.1.8       ggplot2_3.3.6      tidyverse_1.3.2   
# 
# loaded via a namespace (and not attached):
#   [1] fs_1.5.2            lubridate_1.8.0     doParallel_1.0.17   DiceDesign_1.9     
# [5] httr_1.4.4          tools_4.2.2         backports_1.4.1     utf8_1.2.2         
# [9] R6_2.5.1            rpart_4.1.16        DBI_1.1.3           colorspace_2.0-3   
# [13] nnet_7.3-18         withr_2.5.0         tidyselect_1.1.2    compiler_4.2.2     
# [17] cli_3.3.0           rvest_1.0.3         xml2_1.3.3          digest_0.6.29      
# [21] pkgconfig_2.0.3     parallelly_1.33.0   lhs_1.1.5           dbplyr_2.2.1       
# [25] rlang_1.0.6         readxl_1.4.1        rstudioapi_0.13     generics_0.1.3     
# [29] jsonlite_1.8.0      googlesheets4_1.0.0 magrittr_2.0.3      Matrix_1.5-1       
# [33] Rcpp_1.0.9          munsell_0.5.0       fansi_1.0.3         GPfit_1.0-8        
# [37] lifecycle_1.0.3     furrr_0.3.0         stringi_1.7.6       snakecase_0.11.0   
# [41] MASS_7.3-58.1       grid_4.2.2          parallel_4.2.2      listenv_0.8.0      
# [45] crayon_1.5.2        lattice_0.20-45     haven_2.5.0         splines_4.2.2      
# [49] hms_1.1.2           pillar_1.8.1        future.apply_1.9.0  codetools_0.2-18   
# [53] reprex_2.0.1        glue_1.6.2          modelr_0.1.8        vctrs_0.4.1        
# [57] tzdb_0.3.0          foreach_1.5.2       cellranger_1.1.0    gtable_0.3.0       
# # [61] future_1.27.0       assertthat_0.2.1    gower_1.0.0         prodlim_2019.11.13 
# # [65] class_7.3-20        survival_3.4-0      googledrive_2.0.0   gargle_1.2.0       
# # [69] timeDate_4021.104   iterators_1.0.14    hardhat_1.2.0       lava_1.6.10        
# [73] globals_0.15.1      ellipsis_0.3.2      ipred_0.9-13  
