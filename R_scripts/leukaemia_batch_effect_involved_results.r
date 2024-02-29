.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
# Load limma package
library(limma)
library(hgu133plus2.db)
library(ggplot2)
library(reshape2)


## Gene level


exprs <- read.csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/microarray_input_gene_level.csv", header = TRUE )
print(nrow(exprs))
print(ncol(exprs))



anno <- read.csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/annotation_list.csv", col.names = c("Disease", "Batch_info"), header = TRUE)
group <- anno$Disease 
batch <- factor(anno$Batch_info)
print(length(group))
batch_levels_clean <- make.names(levels(batch))
levels(batch) <- batch_levels_clean

design <- model.matrix(~ factor(group) + batch)
colnames(design)[2] <- "AML_vs_Normal"

fit <- lmFit(exprs, design)

contrast.matrix <- makeContrasts(AML_vs_Normal = AML_vs_Normal, levels = design)  # Adjust contrast as needed

fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

res <- topTable(fit2, coef="AML_vs_Normal", number=Inf)

print(sum(res$adj.P.Val < 0.05))

write.csv(res, "/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/deg_results_limma_gene_level.csv")

# The plot
exprs_long <- melt(exprs)
exprs_long$Condition <- rep(group, each=nrow(exprs))
pdf("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/density_plot_gene_level.pdf")
print(ggplot(exprs_long, aes(x=value, fill=Condition)) + 
  geom_density(alpha=0.75) + 
  scale_fill_manual(values=c("blue", "red")) + 
  theme_minimal() + 
  labs(x="Expression Level", y="Density", fill="Condition") + 
  theme(legend.position="bottom"))
dev.off()


## Transcript level

exprs <- read.csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/microarray_input_transcript_level.csv", header = TRUE )
print(nrow(exprs))
print(ncol(exprs))



anno <- read.csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/annotation_list.csv", col.names = c("Disease", "Batch_info"), header = TRUE)
group <- anno$Disease 
batch <- factor(anno$Batch_info)
print(length(group))
batch_levels_clean <- make.names(levels(batch))
levels(batch) <- batch_levels_clean

design <- model.matrix(~ factor(group) + batch)
colnames(design)[2] <- "AML_vs_Normal"

fit <- lmFit(exprs, design)

contrast.matrix <- makeContrasts(AML_vs_Normal = AML_vs_Normal, levels = design)  # Adjust contrast as needed

fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

res <- topTable(fit2, coef="AML_vs_Normal", number=Inf)

print(sum(res$adj.P.Val < 0.05))

gene_symbols <- mapIds(hgu133plus2.db, keys=res$X, column="SYMBOL", keytype="PROBEID", multiVals="first")

res$Symbol <- gene_symbols[match(res$X, names(gene_symbols))]


write.csv(res, "/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/deg_results_limma_transcript_level.csv")

# The plot
exprs_long <- melt(exprs)
exprs_long$Condition <- rep(group, each=nrow(exprs))
pdf("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/density_plot_transcript_level.pdf")
print(ggplot(exprs_long, aes(x=value, fill=Condition)) + 
  geom_density(alpha=0.75) + 
  scale_fill_manual(values=c("blue", "red")) + 
  theme_minimal() + 
  labs(x="Expression Level", y="Density", fill="Condition") + 
  theme(legend.position="bottom"))
dev.off()

