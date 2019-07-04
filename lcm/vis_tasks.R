#' Visualize the Task Locations
#'
library("ggplot2")
library("dplyr")
library("reshape2")
library("readr")
theme_set(theme_bw())

train <- read_csv("../data/metadata_train.csv")
validate <- read_csv("../data/metadata_validate.csv")
metadata <- rbind(train, validate)

ggplot(metadata) +
    geom_point(
        aes(x = x_coord, y = y_coord, col = train),
        alpha = 0.1, size = 0.4
    ) +
    facet_wrap(~ region)


coords <- metadata %>%
    filter(region == "de", x_coord > 5840, y_coord > 4000) %>%
    select(patch_fn) %>%
    mutate(cmd = paste0("scp beluga:/scratch/mtrazzak/datasets/lcm/", patch_fn, " ../")) %>%
    .[["cmd"]]
sapply(coords, system)
