.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
library(tidyverse)
df <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/older_models_metrics/reconstructed_metrics_plots/correlation_values_boxplot.csv')
df$extracted_model <- sub(".* \\+ ", "", df$model_name)

df <- df %>%
  mutate(outliers = ifelse(outliers == "", NA, outliers),
         outliers = str_replace_all(outliers, "\\[|\\]", ""),  # Remove the square brackets
         outliers = str_split(outliers, ",\\s*")) %>%  # Split the string into a list
  mutate(outliers = map(outliers, ~na_if(.x, "") %>% as.numeric))  # Convert to numeric, handling NA


df <- df %>%
  group_by(extracted_model, level) %>%
  slice(1) %>%
  ungroup()

df_outliers <- df %>%
  select(extracted_model, level, outliers) %>%
  unnest(outliers, keep_empty = TRUE) %>%
  drop_na(outliers)  # Drop rows where outliers are NA

df$extracted_model <- factor(df$extracted_model, levels = c('simpleAE', 'simpleVAE', 'priorVAE' , 'beta_simpleVAE', 'beta_priorVAE'))


p <- ggplot(df, aes(x = extracted_model, y = median)) + 
  geom_boxplot(aes(lower = q1, middle = median, upper = q3, ymin = min_val, ymax = max_val),
               stat = "identity") +
  geom_jitter(data = df_outliers, aes(y = outliers), color = "red", width = 0.2) +
  theme_minimal() +
  facet_wrap(~ level, scales = "fixed") +
  labs(y = "Correlation") + theme(axis.title.x = element_blank() ,axis.text.x = element_text(angle = 90, vjust=0.5, hjust=1))

pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/finalized_plots/boxplot_three_levles.pdf', width = 8, height = 3)
print(p)
dev.off()
