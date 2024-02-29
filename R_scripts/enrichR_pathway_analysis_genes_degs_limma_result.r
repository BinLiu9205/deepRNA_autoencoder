.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
library(enrichR)
setEnrichrSite("Enrichr")

dbs <- listEnrichrDbs()
websiteLive <- TRUE
if (is.null(dbs)) websiteLive <- FALSE
if (websiteLive) head(dbs)

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

folder_array <- c('degs_analysis_cancer_health', 'degs_analysis_nsclc_sclc', 'degs_analysis_input_leukaemia_multiple_geos' , 'degs_analysis_input_leukaemia')

for (folder in folder_array){
df1 <- read.csv(paste0('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/', folder, '/deg_results_limma_gene_level.csv'))
res_all1 <- runEnrichR(df1[df1$adj.P.Val < 0.05, 'X'], dbs_names = dbs)
write_xlsx(res_all1,paste0('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/', folder, '/gsea_results_based_on_limma_gene_level.xlsx'))

df2 <- read.csv(paste0('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/', folder, '/deg_results_limma_transcript_level.csv'))
res_all2 <- runEnrichR(df2[df2$adj.P.Val < 0.05, 'Symbol'], dbs_names = dbs)
write_xlsx(res_all2,paste0('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/', folder, '/gsea_results_based_on_limma_transcript_level.xlsx'))

}

