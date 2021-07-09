library(splatter)
params <- newSplatParams(batchCells = 2000, nGenes = 5000, 
                         group.prob = c(0.26,0.09,0.1,0.07,0.06,0.04,0.04,0.06,0.05,0.03,0.03,0.05,0.03,0.03,0.03,0.03), 
                         de.prob = 0.25, 
                         path.from = c(0,1,1,1,4,2,2,2,3,3,3,4,5,5,5,5),
                         de.facScale = 0.25, 
                         dropout.mid = 0.8, 
                         dropout.shape = -0.8, 
                         seed = 11)

sim <- splatSimulatePaths(params, verbose = FALSE)
simedcounts <- counts(sim)
idents <- sim$Group
