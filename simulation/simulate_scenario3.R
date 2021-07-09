library(splatter)
params <- newSplatParams(batchCells = 2000, nGenes = 5000, 
                         group.prob = c(0.04,0.06,0.03,0.07,0.08,0.1,0.06,0.04,0.04,0.06,0.07,0.07,0.08,0.1,0.1), 
                         dropout.mid = 0.8,
                         dropout.shape = -0.8,
                         de.prob = 0.3,
                         de.facScale = c(0.11,0.12,0.11,0.11,0.11,0.15,0.12,0.11,0.13,0.12,0.15,0.13,0.13,0.12,0.11),
                         seed = 8)

sim <- splatSimulateGroups(params, verbose = FALSE)
simcounts <- counts(sim)
idents <- sim$Group