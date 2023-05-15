#!/usr/bin/env Rscript
library(tidyverse)

main <- function() {
  tbl <- read_csv("stats.csv")

  mean_n_steps <- mean(tbl$n_steps)
  g_n_steps <- ggplot(tbl, aes(x=n_steps)) +
    geom_histogram() +
    geom_vline(xintercept=mean_n_steps, color="red") +
    ggtitle("Number of Steps")
  ggsave("n_steps.pdf", g_n_steps)

  n_steps_p99 <- quantile(tbl$n_steps, 0.99)
  g_n_steps_trunc <- ggplot(filter(tbl, n_steps < n_steps_p99), aes(x=n_steps)) +
    geom_histogram() +
    geom_vline(xintercept=mean_n_steps, color="red") +
    ggtitle("Number of Steps, truncated at p99")
  ggsave("n_steps_p99_truncated.pdf", g_n_steps_trunc)

  nodes_max_p99 <- quantile(tbl$nodes_p100, probs=c(0.99999)) # 99.999% ~ 300 terms
  g_nodes <- ggplot(tbl, aes(x=nodes_p100)) +
    geom_histogram() +
    geom_vline(xintercept=mean_n_steps, color="red") +
    ggtitle("Max Nodes")
  ggsave("max_nodes.pdf", g_nodes)

  max_nodes_p99 <- quantile(tbl$nodes_p100, 0.99)
  max_nodes_mean <- mean(tbl$nodes_p100)
  g_max_nodes_trunc <- ggplot(filter(tbl, nodes_p100 < max_nodes_p99), aes(x=nodes_p100)) +
    geom_histogram() +
    geom_vline(xintercept=max_nodes_mean, color="red") +
    ggtitle("Max Nodes, truncated at p99")
  ggsave("max_nodes_p99_truncated.pdf", g_max_nodes_trunc)
}

if (!interactive()) {
  main()
}
