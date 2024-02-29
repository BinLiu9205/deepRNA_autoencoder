.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
library(enrichR)
setEnrichrSite("Enrichr")

dbs <- listEnrichrDbs()
websiteLive <- TRUE
if (is.null(dbs)) websiteLive <- FALSE
if (websiteLive) head(dbs)

df1 <- read.csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/highest_relative_error_gene.csv", header = F, col.names = F)
df2 <- read.csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/highest_absolute_error_gene.csv", header = F, col.names = F)
dbs <- c( 'GO_Biological_Process_2021', 'GO_Cellular_Component_2021', 'GO_Molecular_Function_2021', 'KEGG_2019_Human', 'WikiPathways_2019_Human')

runEnrichR <- function(deg, dbs_names){
  if (length(deg) > 0){
    enriched <- enrichr(deg, dbs_names)
    goRes1 <- enriched["GO_Biological_Process_2021"][[1]]
    goRes2 <- enriched["KEGG_2019_Human"][[1]]
    goRes3 <- enriched["WikiPathways_2019_Human"][[1]]
    goRes4 <- enriched["GO_Cellular_Component_2021"][[1]]
    goRes5 <- enriched["GO_Molecular_Function_2021"][[1]]
  }
  res_all <- list(`GO_Biological_Process_2021` = goRes1, `GO_Cellular_Component_2021` = goRes4, 
                           `GO_Molecular_Function_2021` = goRes5, `KEGG_2019_Human` = goRes2, 
                           `WikiPathways_2019_Human` = goRes3)
  return(res_all)
}


library(writexl)
res_all1 <- runEnrichR(df1[,1], dbs_names = dbs)
write_xlsx(res_all1,"/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/highest_relative_error_gene_GSEA_results.xlsx")

res_all2 <- runEnrichR(df2[,1], dbs_names = dbs)
write_xlsx(res_all1,"/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/highest_absolute_error_gene_GSEA_results.xlsx")
 
         

